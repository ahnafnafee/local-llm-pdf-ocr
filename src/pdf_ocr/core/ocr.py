"""
OCRProcessor - LLM-based OCR processing.

Keeps prompts and OCR response cleanup independent of the inference server.
Configure LLM_API_BASE/LLM_MODEL or inject a VisionClient.
"""

import asyncio
import logging
import re

from pdf_ocr.inference import (
    LLMCallError,  # compatibility re-export
    IncompleteCompletionError,
    LLMConfig,
    ModelNotLoadedError,  # compatibility re-export
    OpenAICompatibleClient,
    VisionClient,
    ensure_model_available,
)
from pdf_ocr.utils.image import crop_for_ocr
from pdf_ocr.core.profiles import resolve_profile


PAGE_PROMPT = (
    "Transcribe all text in this document image in reading order. "
    "Preserve line breaks and use Markdown for headings and tables. "
    "Preserve the original spelling and abbreviations, including errors. "
    "Do not correct or expand any source word. Read the visible characters "
    "rather than replacing them with a more likely word. "
    "Return only the transcription, without commentary."
)


# Canonical OlmOCR-2 prompt (the model was RL-trained on this exact string).
# Source: github.com/allenai/olmocr olmocr/prompts/prompts.py
# :func:`build_no_anchoring_v4_yaml_prompt`.
OLMOCR_PAGE_PROMPT = (
    "Attached is one page of a document that you must process. Just return "
    "the plain text representation of this document as if you were reading it "
    "naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following "
    "markdown syntax ![Alt text describing the contents of the figure]"
    "(page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top "
    "specifying values for the primary_language, is_rotation_valid, "
    "rotation_correction, is_table, and is_diagram parameters."
)

# Prompt for cropped box regions — we want raw text only, no metadata
# (the YAML front matter is nonsensical for a single line/region).
CROP_PROMPT = (
    "Transcribe the text in this image exactly as it appears. "
    "Preserve the original spelling and abbreviations, including errors. "
    "Return only the plain text on a single line when possible, "
    "with no metadata, no markdown, no formatting, no explanation."
)


# Phrases the model emits as a fallback when it can't read the crop —
# usually because the crop is blank, decorative, or otherwise non-text.
# We strip them so they don't pollute the searchable text layer.
_HALLUCINATION_PATTERNS = (
    "the quick brown fox jumps over the lazy dog",  # OlmOCR-2 pangram fallback
    "lorem ipsum",
)


def _is_fallback_response(text: str) -> bool:
    """
    True if ``text`` is essentially one of the known LLM fallback phrases.

    A substring match would over-trigger: a real document might contain
    "lorem ipsum" as quoted placeholder text, or the pangram as an
    example sentence. We require the response to *be* the fallback
    after light normalization (case-fold, strip whitespace, drop
    surrounding punctuation/quotes) — i.e. the fallback occupies the
    entire crop response, not just part of it.
    """
    _trim = ".!?\"'`)([]{}<>“”‘’ \t"
    normalized = text.strip().lower().strip(_trim)
    return normalized in _HALLUCINATION_PATTERNS


class OCRLines(list):
    """Transcribed lines with an optional layout hint for hybrid alignment."""

    def __init__(self, lines, *, requires_region_ocr=False):
        super().__init__(lines)
        self.requires_region_ocr = requires_region_ocr


class OCRProcessor:
    """LLM-based OCR processor over an OpenAI-compatible async client.

    Local VLMs occasionally fall into runaway-generation loops on dense
    or unusual pages — we bound both the per-call timeout and the
    response token budget so a single bad page can't hang the pipeline
    indefinitely. Tuned per-call (full-page vs single-line crop): a page
    can legitimately take longer than a crop, and warrants a higher token
    budget for paragraph-level content.
    """

    # Page-level OCR (full image): up to ~4 minutes, ~6k tokens of output.
    # Dense handwritten pages with tables can easily produce 2-3k tokens
    # of markdown, so 6k leaves headroom without enabling endless loops.
    PAGE_TIMEOUT_S: float = 240.0
    PAGE_MAX_TOKENS: int = 6144

    # Crop-level OCR (single box): a sentence at most. Capping much
    # tighter prevents a confused model from emitting a whole-page worth
    # of hallucinated text into one bbox during the refine stage.
    CROP_TIMEOUT_S: float = 60.0
    CROP_MAX_TOKENS: int = 256
    # Subclasses may implement their own recognition methods without a client.
    crop_canvas_size: int | None = None

    def __init__(
        self, api_base: str | None = None, model: str | None = None, *,
        api_key: str | None = None, client: VisionClient | None = None,
        page_prompt: str | None = None,
        crop_prompt: str | None = None,
        profile: str | None = None,
    ):
        config = LLMConfig.from_env(
            api_base=api_base, model=model, api_key=api_key,
            require_endpoint=client is None,
        )
        self.api_base = config.api_base
        self.model = config.model
        self.profile = resolve_profile(self.model, profile)
        self.temperature = self.profile.temperature
        # A bounded page canvas is an alternate view for format failures,
        # not the primary crop: padding every region loses small details.
        self.crop_canvas_size = self.profile.crop_canvas_size
        self._owned_client: OpenAICompatibleClient | None = None
        if client is None:
            client = self._owned_client = OpenAICompatibleClient(
                config, image_first=self.profile.image_first,
                generation_options=dict(self.profile.sampling),
            )
        self.client = client
        native_prompt = self.profile.page_prompt
        if native_prompt is None:
            native_prompt = OLMOCR_PAGE_PROMPT if self.profile.name == "olmocr" else PAGE_PROMPT
        self.page_prompt = page_prompt if page_prompt is not None else native_prompt
        native_crop = self.profile.crop_prompt
        self.crop_prompt = crop_prompt if crop_prompt is not None else (
            native_crop if native_crop is not None else CROP_PROMPT
        )

    async def ensure_model_available(self) -> None:
        """Verify the server advertises the configured model ID."""
        await ensure_model_available(self.client, self.model)

    async def ensure_model_loaded(self) -> None:
        """Compatibility alias; availability does not imply memory residency."""
        await self.ensure_model_available()

    async def aclose(self) -> None:
        """Close an internally created adapter; injected clients stay open."""
        if self._owned_client is not None:
            await self._owned_client.aclose()

    async def perform_ocr(self, image_base64: str) -> list[str]:
        """
        OCR a full page image. Returns a list of non-empty lines in reading order.

        YAML front matter emitted by OlmOCR (rotation/language/is_table flags)
        is stripped before returning. Runaway repetition (the model getting
        stuck emitting the same line over and over) is detected and clipped
        — this happens occasionally on dense handwritten pages even with
        max_tokens set, and pollutes downstream alignment with junk lines.
        """
        text = await self._chat(
            self.page_prompt, image_base64,
            timeout=self.PAGE_TIMEOUT_S,
            max_tokens=self.PAGE_MAX_TOKENS,
        )
        if not text:
            return []
        body = _strip_yaml_front_matter(text)
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        # Linear alignment cannot bind table cells or diagram labels reliably.
        # Read only the front matter, never similarly named text in the body.
        front_matter = re.match(r"\s*---\s*\n(.*?)\n---(?:\s*\n|$)", text, re.DOTALL)
        region_layout = bool(front_matter and re.search(
            r"^\s*is_(?:table|diagram):\s*(?:true|yes)\s*$",
            front_matter.group(1), re.IGNORECASE | re.MULTILINE,
        ))
        return OCRLines(_strip_runaway_repetition(lines), requires_region_ocr=region_layout)

    async def perform_ocr_on_crop(self, image_base64: str) -> str:
        """
        OCR a single cropped box region. Returns a single whitespace-joined
        string (the crop is small, so we don't try to preserve line structure).
        Empty-string for blank/uncertain crops (filtered hallucination).
        """
        try:
            text = await self._chat(
                self.crop_prompt, image_base64,
                timeout=self.CROP_TIMEOUT_S, max_tokens=self.CROP_MAX_TOKENS,
            )
        except IncompleteCompletionError:
            # A line crop cannot legitimately contain hundreds of repeated
            # tokens. Retry using the page-trained prompt, then leave an
            # unreadable region empty rather than embedding a cut-off loop.
            logging.warning("Crop transcription hit its token limit; retrying with the page prompt.")
            try:
                text = await self._chat(
                    self.page_prompt, image_base64,
                    timeout=self.CROP_TIMEOUT_S, max_tokens=self.CROP_MAX_TOKENS,
                )
            except IncompleteCompletionError:
                logging.warning("Crop retry also hit its token limit; region remains unrecognized.")
                return ""
        if not text:
            return ""
        body = _strip_yaml_front_matter(text)
        result = " ".join(line.strip() for line in body.split("\n") if line.strip())
        if _is_fallback_response(result):
            return ""
        return result

    async def perform_ocr_on_region(self, page_image: str, bbox, *, mask_boxes=None) -> str:
        """Read a native region, with one alternate view for format failures.

        OlmOCR can interpret narrow handwriting as LaTeX even when asked for
        literal text. Retry that response on a paper canvas made from source
        pixels. Ordinary crops retain their original detail and aspect ratio.
        The total deadline includes truncation retries and the alternate view.
        """
        async def recognize():
            image = await asyncio.to_thread(crop_for_ocr, page_image, bbox, mask_boxes=mask_boxes)
            if image is None:
                return ""
            text = await self.perform_ocr_on_crop(image)
            formatted = re.search(r"\\[([]|(?<![\w:])\\[A-Za-z]+|\$\$|```", text)
            if self.crop_canvas_size is not None and formatted:
                alternate = await asyncio.to_thread(
                    crop_for_ocr, page_image, bbox, mask_boxes=mask_boxes,
                    canvas_size=self.crop_canvas_size,
                )
                if alternate is not None:
                    logging.debug("Retrying formatted region transcription on a source-pixel canvas.")
                    text = await self.perform_ocr_on_crop(alternate) or text
            return text

        budget = 2 * self.CROP_TIMEOUT_S
        try:
            return await asyncio.wait_for(recognize(), timeout=budget)
        except asyncio.TimeoutError as exc:
            raise LLMCallError(f"Region OCR exceeded its {budget:g}s deadline, including retries") from exc

    async def _chat(
        self,
        prompt: str,
        image_base64: str,
        *,
        timeout: float,
        max_tokens: int,
    ) -> str:
        return await self.client.complete(
            prompt, image_base64, model=self.model,
            timeout=timeout, max_tokens=max_tokens, temperature=self.temperature,
        )


def _strip_runaway_repetition(lines: list[str], max_repeat: int = 20) -> list[str]:
    """
    Drop pathological repetition from LLM output.

    Local VLMs occasionally fall into an output loop on dense or unusual
    pages — the same line is emitted dozens or hundreds of times in a row
    until max_tokens cuts the response off. The repeated junk pollutes
    every box the DP then tries to assign it to. We cap any single string
    at ``max_repeat`` total occurrences across the response: large enough
    to admit legitimate repeated structure (table row tags, separators)
    but small enough that runaway loops are clipped to a handful of lines
    and the rest is dropped.

    A warning is emitted if any clipping happened so the user knows the
    OCR layer for that page may be incomplete.
    """
    counts: dict[str, int] = {}
    out: list[str] = []
    truncated = 0
    for line in lines:
        c = counts.get(line, 0) + 1
        counts[line] = c
        if c <= max_repeat:
            out.append(line)
        else:
            truncated += 1
    if truncated > 0:
        worst = max(counts.items(), key=lambda kv: kv[1])
        logging.warning(
            "LLM OCR output had %d runaway-repetition lines clipped "
            "(worst offender: %r occurred %d times). The model likely "
            "got stuck on this page; output may be incomplete. "
            "Try lowering --max-image-dim or switching --model.",
            truncated, worst[0][:60], worst[1],
        )
    return out


def _strip_yaml_front_matter(text: str) -> str:
    """
    If the response begins with a YAML front matter block (--- ... ---),
    return the body after it. Otherwise return the input unchanged.
    Robust to models that ignore the front-matter instruction.
    """
    t = text.lstrip()
    if not t.startswith("---"):
        return text
    # Find the closing fence on its own line, after the opening fence.
    rest = t[3:]
    close_idx = rest.find("\n---")
    if close_idx == -1:
        return text  # malformed; return as-is
    body = rest[close_idx + len("\n---"):]
    # Trim the newline directly after the closing fence.
    return body.lstrip("\n").strip()
