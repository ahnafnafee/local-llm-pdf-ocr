"""
Grounded OCR — backends that emit text WITH bounding boxes in one call.

When a VLM can natively ground its output (Qwen2.5-VL, Qwen3-VL, Florence-2,
MinerU, Z.AI hosted GLM-OCR, etc.) the whole Surya-detect → LLM-transcribe →
DP-align → refine dance collapses to a single call. The model returns a list
of `(bbox, text)` pairs already bound together, and we just render the page
background and embed them.

This module defines:

- `GroundedBlock`, `GroundedResponse` — normalized shape every backend returns
- `GroundedOCRBackend` — async Protocol (one method: `ocr_document`)
- `PromptedGroundedOCR` — default backend for OpenAI-compat VLMs that emit
  `{"bbox_2d": [...], "content": "..."}` JSON when prompted (Qwen-VL family,
  any vLLM deployment of a grounding-capable vision model)
- `parse_zai_response` / `parse_glm_layout_details` — parsers for two common
  JSON shapes
- `ZAIHostedOCR` — skeleton REST client for the hosted service

The pipeline picks this path automatically when `grounded_backend` is passed
to `OCRPipeline`; otherwise it falls back to the hybrid Surya+LLM+DP flow.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

from pdf_ocr.inference import (
    DEFAULT_GROUNDED_MODEL, IncompleteCompletionError, LLMCallError, LLMConfig,
    OpenAICompatibleClient, VisionClient,
    ensure_model_available,
)
from pdf_ocr.utils.pages import select_pages
from pdf_ocr.core.profiles import resolve_profile


ProgressCallback = Callable[[str, int, int, str], Awaitable[None]]


@dataclass
class GroundedBlock:
    bbox: list[float]       # normalized [nx0, ny0, nx1, ny1] in 0..1
    text: str
    page_index: int
    label: str = "text"     # filter: keep "text", drop "image"/"figure"


@dataclass
class GroundedResponse:
    blocks: list[GroundedBlock]
    page_sizes: list[tuple[int, int]] = field(default_factory=list)  # (w, h) per page
    # Source indices corresponding to page_sizes, including pages with no
    # recognized blocks. None preserves the legacy contiguous-page contract.
    page_indices: list[int] | None = None
    page_errors: dict[int, str] = field(default_factory=dict)


class GroundedOCRBackend(Protocol):
    """Backends that return text WITH layout in one shot (no Surya needed).

    `progress` is optional; callers that don't care about per-page updates
    can omit it. Backends SHOULD emit the `"ocr"` stage with (current,
    total) set to pages-completed / total-pages so the pipeline's progress
    adapter stays aligned with the documented stage set.
    `pages` is an optional 1-indexed range; select before rendering/inference
    and return original source indices in blocks and response.page_indices.
    """

    async def ocr_document(
        self,
        pdf_path: str,
        progress: Optional[ProgressCallback] = None,
        *,
        pages: str | None = None,
    ) -> GroundedResponse: ...


# Labels we treat as *non-content* — structural regions that aren't meant
# to carry selectable text. Newer grounded responses emit labels like
# "title", "list_item", "form_field", "diagram_node" etc. alongside "text";
# the old handwritten fixture was pure "text" + "image". Instead of allow-
# listing content labels (brittle across schema versions) we deny-list the
# structural ones.
_NON_CONTENT_LABELS = frozenset({
    "image",
    "empty_line",      # unfilled underline fields
    "signature_line",  # form signature placeholder
    "list_marker",     # lone bullet/dash glyphs
})


def _parse_lighton_grounding(text: str, page_index: int) -> list[GroundedBlock]:
    """Parse native LightOnOCR-3 block markers on its fixed 0–1000 grid.

    Image/chart descriptions are generated visual interpretations, not OCR.
    Keep their boundaries when splitting text but exclude them from output.
    """
    markers = list(re.finditer(
        r"!\[([a-z_]+\+?)\]\((\d+(?:\s*,\s*\d+){3})\)", text,
    ))
    if not markers:
        raise ValueError("LightOnOCR returned no native grounding markers")
    if text[:markers[0].start()].strip():
        raise ValueError("Unexpected text before LightOnOCR grounding markers")
    blocks = []
    for index, marker in enumerate(markers):
        label = marker[1].removesuffix("+")
        coordinates = [int(value) for value in marker[2].split(",")]
        x0, y0, x1, y1 = coordinates
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("Invalid LightOnOCR bounding box")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        content = text[marker.end():end].strip()
        if re.search(r"!\[[a-z_]+\+?\]\(", content):
            raise ValueError("Malformed LightOnOCR grounding marker")
        if label in {"image", "header_image", "footer_image", "chart"} or not content:
            continue
        blocks.append(GroundedBlock(
            bbox=[value / 1000 for value in coordinates], text=content,
            page_index=page_index, label=label,
        ))
    return blocks


def _parse_paddle_spotting(text: str, page_index: int) -> list[GroundedBlock]:
    """Convert PaddleOCR-VL's eight location tokens into enclosing boxes."""
    tokens = list(re.finditer(r"<\|LOC_(\d+)\|>", text))
    if not tokens:
        raise LLMCallError("Paddle spotting returned no location tokens; the server must "
                           "preserve special tokens for Spotting mode.")
    if len(tokens) % 8:
        raise ValueError("Incomplete Paddle spotting polygon")
    wrappers = r"<\|(?:TEXT_START|TEXT_END|LOC_BEGIN|LOC_END)\|>"
    blocks = []
    end = 0
    for start in range(0, len(tokens), 8):
        group = tokens[start:start + 8]
        content = re.sub(wrappers, "", text[end:group[0].start()]).strip()
        polygon = [int(token[1]) for token in group]
        between = text[group[0].start():group[-1].end()]
        if re.sub(r"<\|LOC_\d+\|>|\s+", "", between):
            raise ValueError("Text interrupts a Paddle spotting polygon")
        end = group[-1].end()
        x0, y0 = min(polygon[::2]), min(polygon[1::2])
        x1, y1 = max(polygon[::2]), max(polygon[1::2])
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("Invalid Paddle spotting polygon")
        if content:
            blocks.append(GroundedBlock(
                [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000], content, page_index,
            ))
    if re.sub(wrappers, "", text[end:]).strip():
        raise ValueError("Paddle spotting has trailing text without a polygon")
    return blocks


# --- parsers ---------------------------------------------------------------


def parse_zai_response(payload: dict[str, Any]) -> GroundedResponse:
    """
    Parse the Z.AI hosted OCR API response (the shape returned by
    `https://ocr.z.ai` / z-ai-audio) into normalized blocks.

    Handles both bbox axis conventions (`[x0,y0,x1,y1]` and `[y0,x0,y1,x1]`)
    — auto-detected per-response by the aspect-ratio heuristic in
    :func:`_detect_axis_order_zxyxy`. Accepts either the full envelope
    `{"code": 200, "data": {...}}` or just the inner `data` object.
    """
    d = payload.get("data", payload)
    pages = d.get("data_info", {}).get("pages", [])
    page_sizes = [(int(p["width"]), int(p["height"])) for p in pages]

    raw_items = [
        b for b in d.get("layout", [])
        if b.get("block_label", "text") not in _NON_CONTENT_LABELS
    ]
    swap = _detect_axis_order_zxyxy([b["bbox"] for b in raw_items]) == "yxyx"

    blocks: list[GroundedBlock] = []
    for b in raw_items:
        pidx = b.get("page_index", 0)
        if pidx >= len(page_sizes):
            continue
        pw, ph = page_sizes[pidx]
        bbox = b["bbox"]
        if swap:
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        x0, y0, x1, y1 = bbox
        content = (b.get("block_content") or "").strip()
        if not content:
            continue
        blocks.append(GroundedBlock(
            bbox=[_clamp(x0 / pw), _clamp(y0 / ph),
                  _clamp(x1 / pw), _clamp(y1 / ph)],
            text=content,
            page_index=pidx,
            label=b.get("block_label", "text"),
        ))
    return GroundedResponse(blocks=blocks, page_sizes=page_sizes)


def _detect_axis_order_zxyxy(raw_boxes: list[list[float]]) -> str:
    """Return "xyxy" or "yxyx" based on whether boxes look portrait as xyxy.

    Kept as a tiny self-contained helper (rather than importing from
    evaluation.py) to keep this module's surface independent of tests.
    """
    portrait, counted = 0, 0
    for b in raw_boxes:
        if len(b) != 4:
            continue
        w_xy = abs(b[2] - b[0])
        h_xy = abs(b[3] - b[1])
        if w_xy <= 0 or h_xy <= 0:
            continue
        counted += 1
        if h_xy > 1.5 * w_xy:
            portrait += 1
    if counted == 0:
        return "xyxy"
    return "yxyx" if portrait > counted / 2 else "xyxy"


def parse_glm_layout_details(payload_or_json: Any, page_index: int = 0) -> GroundedResponse:
    """
    Parse `layout_details` emitted by GLM-OCR via vLLM / self-hosted server,
    where each block has `bbox_2d: [x0, y0, x1, y1]` in pixel coords relative
    to the rendered page image.

    Accepts either the full JSON object or a pre-parsed dict. `page_index`
    specifies which page the blocks belong to (single-page calls).
    """
    if isinstance(payload_or_json, str):
        payload_or_json = json.loads(payload_or_json)
    d = payload_or_json

    pages = d.get("data_info", {}).get("pages", [])
    page_sizes = [(int(p["width"]), int(p["height"])) for p in pages]
    if not page_sizes:
        raise ValueError("parse_glm_layout_details: missing data_info.pages")

    # layout_details can be list[list[block]] (per page) or flat list.
    raw = d.get("layout_details", [])
    if raw and isinstance(raw[0], list):
        raw_blocks = raw[page_index] if page_index < len(raw) else []
    else:
        raw_blocks = raw

    blocks: list[GroundedBlock] = []
    pw, ph = page_sizes[page_index]
    for b in raw_blocks:
        if b.get("label") != "text":
            continue
        content = (b.get("content") or "").strip()
        if not content:
            continue
        x0, y0, x1, y1 = b["bbox_2d"]
        blocks.append(GroundedBlock(
            bbox=[_clamp(x0 / pw), _clamp(y0 / ph),
                  _clamp(x1 / pw), _clamp(y1 / ph)],
            text=content,
            page_index=page_index,
        ))
    return GroundedResponse(blocks=blocks, page_sizes=page_sizes)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _text_ink_coverage(image, blocks: list[GroundedBlock]) -> float | None:
    """Conservative evidence for a response confined to a small page fragment.

    Count dark source pixels under the reported boxes, allowing a small margin
    for imperfect coordinates. This is not an OCR confidence score: pictures
    also contain ink. It is only used to request a second view after a severely
    incomplete full-page response, never to infer words from the pixels itself.
    Uniform backgrounds and pages with too little dark ink provide no evidence.
    """
    from PIL import Image, ImageChops, ImageDraw

    gray = image.convert("L")
    histogram = gray.histogram()
    target = gray.width * gray.height * 0.9
    count = 0
    background = 255
    for value, frequency in enumerate(histogram):
        count += frequency
        if count >= target:
            background = value
            break
    threshold = min(160, background - 40)
    if threshold <= 0:
        return None
    ink = gray.point(lambda value: 255 if value < threshold else 0)
    total = ink.histogram()[255]
    if total < max(64, gray.width * gray.height * 0.0001):
        return None
    covered = Image.new("L", gray.size, 0)
    draw = ImageDraw.Draw(covered)
    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        draw.rectangle((
            max(0, int((x0 - 0.01) * gray.width)),
            max(0, int((y0 - 0.01) * gray.height)),
            min(gray.width - 1, int((x1 + 0.01) * gray.width)),
            min(gray.height - 1, int((y1 + 0.01) * gray.height)),
        ), fill=255)
    return ImageChops.multiply(ink, covered).histogram()[255] / total


# --- reference backend: Z.AI hosted (HTTP, async task polling) -------------


class ZAIHostedOCR:
    """
    Reference skeleton for Z.AI's hosted OCR REST service.

    Z.AI hasn't published the exact async-task endpoint path publicly — the
    response shape you've seen comes from `ocr.z.ai`. Fill in `SUBMIT_PATH`
    and `TASK_PATH` once you have the real routes (Network tab on ocr.z.ai
    or a Z.AI support ticket will give them). The parse + polling logic is
    complete; only the two endpoint strings and the `submit` request body
    format may need tweaking.

    Expected flow:
        1. POST {base_url}{SUBMIT_PATH} with the PDF  → {task_id}
        2. GET  {base_url}{TASK_PATH}/{task_id}       → {status, data:{layout,...}}
           repeat until status == "completed"
        3. parse_zai_response(data) → GroundedResponse

    Usage:
        backend = ZAIHostedOCR(api_key=os.environ["ZAI_API_KEY"])
        pipeline = OCRPipeline(grounded_backend=backend, pdf_handler=PDFHandler())
        await pipeline.run("in.pdf", "out.pdf")
    """

    # TODO: confirm these against the live service.
    SUBMIT_PATH = "/api/paas/v4/ocr/submit"
    TASK_PATH = "/api/paas/v4/ocr/task"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.z.ai",
        poll_interval_s: float = 2.0,
        timeout_s: float = 300.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    async def ocr_document(
        self,
        pdf_path: str,
        progress: Optional[ProgressCallback] = None,
        *,
        pages: str | None = None,
    ) -> GroundedResponse:
        import asyncio
        import httpx

        if pages:
            raise NotImplementedError("ZAIHostedOCR does not support page selection.")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=60) as client:
            # 1. Submit
            with open(pdf_path, "rb") as f:
                resp = await client.post(
                    self.base_url + self.SUBMIT_PATH,
                    headers=headers,
                    files={"file": (pdf_path.rsplit("/", 1)[-1], f, "application/pdf")},
                )
            resp.raise_for_status()
            task_id = resp.json()["data"]["task_id"]

            # 2. Poll until completed
            elapsed = 0.0
            while elapsed < self.timeout_s:
                await asyncio.sleep(self.poll_interval_s)
                elapsed += self.poll_interval_s
                r = await client.get(
                    f"{self.base_url}{self.TASK_PATH}/{task_id}",
                    headers=headers,
                )
                r.raise_for_status()
                payload = r.json()
                status = payload.get("data", {}).get("status")
                if status == "completed":
                    return parse_zai_response(payload)
                if status in ("failed", "error"):
                    raise RuntimeError(f"Z.AI OCR task failed: {payload}")
            raise TimeoutError(f"Z.AI OCR task {task_id} did not complete in {self.timeout_s}s")


# --- prompted grounded backend (works with Qwen2.5-VL / Qwen3-VL / etc.) ---


DEFAULT_GROUNDING_PROMPT = (
    "Read all text in the image, including small labels and handwritten notes. "
    "Return a compact JSON array with one entry per visual text line: "
    '{"bbox_2d":[x1,y1,x2,y2],"content":"text"}. '
    "Use coordinates from 0 to 1000 relative to this image, with each box "
    "tightly enclosing its line. Preserve separate lines, transcribe faithfully "
    "with the original spelling and abbreviations, including errors, "
    "in reading order, and output each line once. Do not describe figures or "
    "invent text in blank regions. Return only compact JSON, with no indentation "
    "or whitespace outside strings."
)

_GROUNDING_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["blocks"],
    "properties": {"blocks": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["bbox_2d", "content"], "properties": {
            "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4,
                        "items": {"type": "number"}},
            "content": {"type": "string"},
        },
    }}},
}


class PromptedGroundedOCR:
    """
    Grounded backend built on an OpenAI-compatible vision LLM endpoint.

    Works with any VLM that emits `{bbox_2d:[...], content:"..."}` when asked —
    confirmed for Qwen2.5-VL (line-level, wrapped in fences) and Qwen3-VL
    (line-level, bare JSON). Should also work for MiniCPM-V, InternVL, etc.

    The backend rasterizes each selected PDF page itself and retries
    truncated or malformed responses on smaller source regions, so it
    doubles as its own PDF handler — the OCRPipeline will use its
    `ocr_document(pdf_path)` method directly.

    Usage:
        backend = PromptedGroundedOCR(
            api_base="http://localhost:8000/v1",
            model="your-vision-model",
        )
        pipe = OCRPipeline(pdf_handler=PDFHandler(), grounded_backend=backend)
        await pipe.run("in.pdf", "out.pdf")
    """

    def __init__(
        self,
        api_base: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_image_dim: int = 1024,
        dpi: int = 150,
        prompt: str | None = None,
        timeout_s: float = 240.0,
        max_tokens: int | None = None,
        concurrency: int = 1,
        *,
        client: VisionClient | None = None,
        profile: str | None = None,
    ):
        config = LLMConfig.from_env(
            api_base=api_base, model=model, api_key=api_key,
            require_endpoint=client is None,
            default_model=DEFAULT_GROUNDED_MODEL,
        )
        self.api_base = config.api_base
        self.model = config.model
        self.profile = resolve_profile(self.model, profile)
        self._owned_client: OpenAICompatibleClient | None = None
        self.temperature = self.profile.grounded_temperature
        if client is None:
            client = self._owned_client = OpenAICompatibleClient(
                config, generation_options=dict(self.profile.grounded_sampling),
                image_first=self.profile.image_first,
            )
        self.client = client
        self.max_image_dim = max_image_dim
        self.dpi = dpi
        self.prompt = prompt if prompt is not None else (
            self.profile.grounding_prompt or DEFAULT_GROUNDING_PROMPT
        )
        self.grounding_format = self.profile.grounding_format
        self._check_full_page_coverage = prompt is None and self.grounding_format == "json"
        self.coordinate_space = "auto" if prompt else "normalized_1000"
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens if max_tokens is not None else self.profile.grounded_max_tokens
        self.concurrency = concurrency

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

    async def ocr_document(
        self,
        pdf_path: str,
        progress: Optional[ProgressCallback] = None,
        *,
        pages: str | None = None,
    ) -> GroundedResponse:
        import fitz
        from PIL import Image

        from pdf_ocr.core.pdf import _is_image_path

        # 1. Rasterize selected pages, remembering original indices and sizes.
        # For image inputs (JPEG/PNG/TIFF) skip the PDF round-trip and read
        # pixels directly — this both saves work and supports multi-frame TIFF.
        page_imgs: dict[int, tuple[str, int, int]] = {}  # source index -> (b64, w, h)
        source_imgs: dict[int, Image.Image] = {}

        def _emit(page_idx: int, img: Image.Image) -> None:
            img = img.convert("RGB")
            source_imgs[page_idx] = img.copy()
            img.thumbnail((self.max_image_dim, self.max_image_dim))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            page_imgs[page_idx] = (
                base64.b64encode(buf.getvalue()).decode(), img.width, img.height
            )

        if _is_image_path(pdf_path):
            with Image.open(pdf_path) as src:
                for page_idx in select_pages(pages, getattr(src, "n_frames", 1)):
                    src.seek(page_idx)
                    _emit(page_idx, src.copy())
        else:
            doc = fitz.open(pdf_path)
            try:
                for page_idx in select_pages(pages, len(doc)):
                    page = doc[page_idx]
                    pix = page.get_pixmap(dpi=self.dpi)
                    _emit(page_idx, Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            finally:
                doc.close()

        # 2. Call the VLM per page, streaming progress and collecting failures
        # as empty pages. The pipeline validates all pages before writing output.
        sem = asyncio.Semaphore(max(1, self.concurrency))
        total_pages = len(page_imgs)
        page_errors: dict[int, str] = {}

        async def run_one(page_idx: int) -> tuple[int, list[GroundedBlock]]:
            b64, w, h = page_imgs[page_idx]

            async def recognize(image_b64, width, height, *, depth=0):
                complete_json = (getattr(self.client, "complete_json", None)
                                 if self.grounding_format == "json" else None)
                # Optional capability keeps the small injectable client
                # contract usable on endpoints without constrained decoding.
                complete = complete_json or self.client.complete
                text = await complete(
                    (self.prompt + '\nWrap the array in an object with a "blocks" key.'
                     if complete_json else self.prompt), image_b64, model=self.model,
                    timeout=self.timeout_s, max_tokens=min(self.max_tokens, 2048) if depth == 2 else self.max_tokens,
                    temperature=self.temperature,
                    **({"schema": _GROUNDING_SCHEMA} if complete_json else {}),
                )
                if self.grounding_format == "lighton":
                    blocks = _parse_lighton_grounding(text, page_idx)
                elif self.grounding_format == "paddle":
                    blocks = _parse_paddle_spotting(text, page_idx)
                else:
                    blocks = _parse_grounded_json(
                        text, page_idx, width, height,
                        coordinate_space=self.coordinate_space, strict=True,
                    )
                if not blocks and not (depth and _is_empty_grounded_json(text)):
                    raise LLMCallError("model returned no usable grounded text")
                if depth == 0 and self._check_full_page_coverage:
                    coverage = _text_ink_coverage(source_imgs[page_idx], blocks)
                    if coverage is not None and coverage < 0.20:
                        raise ValueError(
                            f"Grounded boxes cover only {coverage:.1%} of source ink; "
                            "retrying source regions to check for omitted text"
                        )
                return blocks

            source = source_imgs[page_idx]

            async def recognize_region(bounds, depth=0):
                left, top, right, bottom = bounds
                if depth == 0:
                    image_b64, width, height = b64, w, h
                else:
                    region = source.crop(bounds)
                    region.thumbnail((self.max_image_dim, self.max_image_dim))
                    buf = io.BytesIO()
                    region.save(buf, format="PNG")
                    image_b64 = base64.b64encode(buf.getvalue()).decode()
                    width, height = region.size
                try:
                    blocks = await recognize(image_b64, width, height, depth=depth)
                    for block in blocks:
                        x0, y0, x1, y1 = block.bbox
                        block.bbox = [
                            (left + x0 * (right - left)) / source.width,
                            (top + y0 * (bottom - top)) / source.height,
                            (left + x1 * (right - left)) / source.width,
                            (top + y1 * (bottom - top)) / source.height,
                        ]
                    return blocks
                except (IncompleteCompletionError, ValueError) as exc:
                    if depth >= 2:
                        raise
                    logging.warning(
                        "Grounded page %d region needs subdivision at depth %d (%s).",
                        page_idx + 1, depth, exc,
                    )
                # Split the longer edge into overlapping thirds, using source
                # pixels. A second split can isolate dense columns. Accept only
                # centers owned by each core, avoiding duplicate seam lines.
                axis = 0 if right - left > bottom - top else 1
                start, end = bounds[axis], bounds[axis + 2]
                span = end - start
                recovered = []
                for part in range(3):
                    core_start = start + part * span / 3
                    core_end = start + (part + 1) * span / 3
                    child = list(bounds)
                    child[axis] = max(start, int(core_start - 0.025 * span))
                    child[axis + 2] = min(end, int(core_end + 0.025 * span))
                    blocks = await recognize_region(tuple(child), depth + 1)
                    for block in blocks:
                        center = (block.bbox[axis] + block.bbox[axis + 2]) / 2
                        center *= source.size[axis]
                        if core_start <= center < core_end or (part == 2 and center == end):
                            recovered.append(block)
                return recovered

            async def recognize_page():
                blocks = await recognize_region((0, 0, source.width, source.height))
                if not blocks:
                    raise LLMCallError("model returned no usable grounded text")
                return blocks

            async with sem:
                try:
                    # Bound the whole page, including recovery and clients
                    # that do not enforce their own request timeout.
                    return page_idx, await asyncio.wait_for(recognize_page(), self.timeout_s)
                except Exception as e:
                    # Retain successful pages in the response for callers that
                    # inspect partial results; the pipeline rejects empty pages.
                    reason = (f"Page exceeded its {self.timeout_s:g}s deadline, including recovery."
                              if isinstance(e, asyncio.TimeoutError)
                              else f"{type(e).__name__}: {e}")
                    page_errors[page_idx] = reason
                    logging.warning("grounded OCR failed for page %d: %s", page_idx + 1, reason)
                    return page_idx, []

        tasks = [asyncio.create_task(run_one(i)) for i in page_imgs]
        blocks_by_page: dict[int, list[GroundedBlock]] = {}
        completed = 0
        try:
            if progress is not None:
                await progress("ocr", 0, total_pages, f"Grounded OCR (0/{total_pages})...")
            for fut in asyncio.as_completed(tasks):
                page_idx, blocks = await fut
                blocks_by_page[page_idx] = blocks
                completed += 1
                if progress is not None:
                    await progress(
                        "ocr", completed, total_pages,
                        f"Grounded OCR ({completed}/{total_pages})",
                    )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten in page order for a stable, deterministic output.
        flat_blocks: list[GroundedBlock] = []
        for page_idx in page_imgs:
            flat_blocks.extend(blocks_by_page.get(page_idx, []))
        return GroundedResponse(
            blocks=flat_blocks,
            page_sizes=[(w, h) for (_, w, h) in page_imgs.values()],
            page_indices=list(page_imgs),
            page_errors=page_errors,
        )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", re.IGNORECASE)
_BARE_ARRAY = re.compile(r"(\[[\s\S]*\])")

# Grounding VLMs use two bbox coordinate conventions: absolute pixels of
# the supplied image (Qwen2.5-VL, GLM-OCR) and a fixed 0-1000 grid
# independent of image size (Qwen3-VL). Dividing grid coordinates by the
# pixel dimensions silently misplaces every box, so the convention is
# detected per page before normalizing.
_GRID_MAX = 1000.0


def _bbox_scale_denominators(
    raw_boxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> tuple[float, float, bool]:
    """Pick the divisors mapping raw bbox coords to 0..1 normalized space.

    Returns ``(den_x, den_y, is_grid)`` where ``is_grid`` says the page's
    coordinates were judged to be on the 0-1000 grid rather than pixels:

    - any coordinate beyond 1000 must be pixels;
    - x beyond the image width or y beyond the image height cannot be
      pixels, so the page is on the grid;
    - every coordinate within 1000 on an image much larger than the grid
      in BOTH axes would mean full-page text confined to the top-left
      1000px corner — implausible, treat as grid.

    Images near the grid size (the default ``max_image_dim=1024``
    thumbnail) are inherently ambiguous, but there the two readings
    differ by under 3% so either divisor places boxes acceptably.
    """
    if not raw_boxes:
        return float(img_w), float(img_h), False
    max_x = max(max(b[0], b[2]) for b in raw_boxes)
    max_y = max(max(b[1], b[3]) for b in raw_boxes)
    if max_x > _GRID_MAX + 0.5 or max_y > _GRID_MAX + 0.5:
        return float(img_w), float(img_h), False
    if max_x > img_w + 2 or max_y > img_h + 2:
        return _GRID_MAX, _GRID_MAX, True
    if img_w >= 1.4 * _GRID_MAX and img_h >= 1.4 * _GRID_MAX:
        return _GRID_MAX, _GRID_MAX, True
    return float(img_w), float(img_h), False


def _is_empty_grounded_json(text: str) -> bool:
    """Accept explicit empty recovery bands, independently of JSON formatting."""
    raw = text.strip()
    fence = _JSON_FENCE.search(raw)
    if fence:
        raw = fence.group(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False
    blocks = payload.get("blocks") if isinstance(payload, dict) else payload
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            return False
        values = [block[key] for key in ("content", "text", "text_content") if key in block]
        if not values or any(not isinstance(value, str) or value.strip() for value in values):
            return False
    return True


def _parse_grounded_json(
    text: str, page_idx: int, img_w: int, img_h: int,
    *, coordinate_space: str = "auto", strict: bool = False,
) -> list[GroundedBlock]:
    """
    Extract a JSON array of `{bbox_2d, content}` blocks from a VLM response.

    Handles three observed response shapes:
      1. Bare JSON array (Qwen3-VL)
      2. JSON wrapped in ```json ... ``` fence (Qwen2.5-VL)
      3. JSON with preamble prose before the array
    """
    raw = text.strip()
    if not raw:
        return []

    # Strip code fence if present.
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    elif raw.startswith("```"):
        # Defensive: open fence but closing dropped by truncation.
        raw = raw.lstrip("`").lstrip("json").lstrip().rstrip("`").rstrip()

    # Try a direct parse; fall back to greediest array substring.
    data: Any
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m2 = _BARE_ARRAY.search(raw)
        if not m2:
            logging.debug(f"grounded parse: no array in response: {raw[:200]!r}")
            return []
        try:
            data = json.loads(m2.group(1))
        except json.JSONDecodeError as e:
            if strict:
                raise ValueError("model returned malformed grounded JSON") from e
            logging.debug(f"grounded parse failed: {e} — raw={raw[:200]!r}")
            return []

    if isinstance(data, dict):
        # Some models wrap the array in {"results": [...]} or similar.
        for key in ("results", "blocks", "layout", "layout_details", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]  # single object → one-element list

    if not isinstance(data, list):
        return []
    parsed: list[tuple[tuple[float, float, float, float], str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d") or item.get("bbox")
        content = item.get("content") or item.get("text") or ""
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        content = str(content).strip()
        if not content:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        parsed.append(((x0, y0, x1, y1), content))

    # Convention detection needs the whole page's boxes, so normalization
    # happens after the collection pass.
    if coordinate_space == "normalized_1000":
        den_x, den_y, is_grid = _GRID_MAX, _GRID_MAX, False
    else:
        den_x, den_y, is_grid = _bbox_scale_denominators(
            [p[0] for p in parsed], img_w, img_h,
        )
    if is_grid:
        logging.warning(
            f"grounded parse: page {page_idx} bbox coordinates fit the "
            f"0-1000 grid but not the {img_w}x{img_h} image — treating "
            f"bbox_2d as thousandths of the page (Qwen3-VL convention) "
            f"instead of pixels"
        )

    return [
        GroundedBlock(
            bbox=[_clamp(x0 / den_x), _clamp(y0 / den_y),
                  _clamp(x1 / den_x), _clamp(y1 / den_y)],
            text=content,
            page_index=page_idx,
        )
        for (x0, y0, x1, y1), content in parsed
    ]
