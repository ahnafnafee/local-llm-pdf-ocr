"""
OCRProcessor - LLM-based OCR processing.

Uses a local vision LLM (OlmOCR via LM Studio by default; any OpenAI-compatible
endpoint works, including GLM OCR via Ollama — set LLM_API_BASE/LLM_MODEL or
pass --api-base/--model).
"""

import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class LLMCallError(RuntimeError):
    """Raised when a call to the local LLM OCR endpoint fails.

    Wraps the underlying exception (connection refused, model not loaded,
    timeout, auth, ...) with a message that names the api-base and model
    so the user can diagnose without digging through a stack trace.
    """


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

    def __init__(self, api_base: str | None = None, model: str | None = None):
        self.api_base = api_base or os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
        self.model = model or os.getenv("LLM_MODEL", "allenai/olmocr-2-7b")
        self.client = AsyncOpenAI(base_url=self.api_base, api_key="lm-studio")

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
            OLMOCR_PAGE_PROMPT, image_base64,
            timeout=self.PAGE_TIMEOUT_S,
            max_tokens=self.PAGE_MAX_TOKENS,
        )
        if not text:
            return []
        body = _strip_yaml_front_matter(text)
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        return _strip_runaway_repetition(lines)

    async def perform_ocr_on_crop(self, image_base64: str) -> str:
        """
        OCR a single cropped box region. Returns a single whitespace-joined
        string (the crop is small, so we don't try to preserve line structure).
        Empty-string for blank/uncertain crops (filtered hallucination).
        """
        text = await self._chat(
            CROP_PROMPT, image_base64,
            timeout=self.CROP_TIMEOUT_S,
            max_tokens=self.CROP_MAX_TOKENS,
        )
        if not text:
            return ""
        body = _strip_yaml_front_matter(text)
        result = " ".join(line.strip() for line in body.split("\n") if line.strip())
        if _is_fallback_response(result):
            return ""
        return result

    async def _chat(
        self,
        prompt: str,
        image_base64: str,
        *,
        timeout: float,
        max_tokens: int,
    ) -> str:
        try:
            response = await self.client.with_options(
                timeout=timeout
            ).chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise LLMCallError(
                f"LLM OCR call failed against {self.api_base} "
                f"(model={self.model!r}): {type(e).__name__}: {e}\n"
                f"  - Is your local LLM server (LM Studio / Ollama / vLLM) running at "
                f"{self.api_base}?\n"
                f"  - Is model {self.model!r} loaded and serving vision inputs?\n"
                f"  - Did the model run away on this page? "
                f"Try reducing --max-image-dim or capping the page complexity.\n"
                f"  - Override endpoint with LLM_API_BASE / LLM_MODEL in .env, "
                f"or pass --api-base / --model."
            ) from e


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
