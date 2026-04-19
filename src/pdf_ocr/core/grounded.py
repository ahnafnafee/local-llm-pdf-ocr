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


class GroundedOCRBackend(Protocol):
    """Backends that return text WITH layout in one shot (no Surya needed).

    `progress` is optional; callers that don't care about per-page updates
    can omit it. Backends SHOULD emit the `"ocr"` stage with (current,
    total) set to pages-completed / total-pages so the pipeline's progress
    adapter stays aligned with the documented stage set.
    """

    async def ocr_document(
        self,
        pdf_path: str,
        progress: Optional[ProgressCallback] = None,
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
    ) -> GroundedResponse:
        import asyncio
        import httpx

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
    "You are an exhaustive OCR engine. Output a JSON array covering EVERY "
    "text region on this page: headers, form labels, field names, body "
    "paragraphs, numbered items, signatures, footnotes — all of it. "
    "Each element must have this exact shape: "
    "{\"bbox_2d\": [x1, y1, x2, y2], \"content\": \"<text>\"} "
    "where bbox_2d is pixel coordinates in the image (x1<x2, y1<y2). "
    "Do not skip small labels. Do not summarize. Do not merge distant lines. "
    "No markdown fences, no prose — only the raw JSON array."
)


class PromptedGroundedOCR:
    """
    Grounded backend built on an OpenAI-compatible vision LLM endpoint.

    Works with any VLM that emits `{bbox_2d:[...], content:"..."}` when asked —
    confirmed for Qwen2.5-VL (line-level, wrapped in fences) and Qwen3-VL
    (line-level, bare JSON). Should also work for MiniCPM-V, InternVL, etc.

    The backend rasterizes each PDF page itself (one call per page), so it
    doubles as its own PDF handler — the OCRPipeline will use its
    `ocr_document(pdf_path)` method directly.

    Usage:
        backend = PromptedGroundedOCR(
            api_base="http://localhost:1234/v1",
            model="qwen/qwen3-vl-8b",
        )
        pipe = OCRPipeline(pdf_handler=PDFHandler(), grounded_backend=backend)
        await pipe.run("in.pdf", "out.pdf")
    """

    def __init__(
        self,
        api_base: str = "http://localhost:1234/v1",
        model: str = "qwen/qwen3-vl-8b",
        api_key: str = "lm-studio",
        max_image_dim: int = 1024,
        dpi: int = 150,
        prompt: str | None = None,
        timeout_s: float = 240.0,
        max_tokens: int = 8192,
        concurrency: int = 1,
    ):
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
        self.max_image_dim = max_image_dim
        self.dpi = dpi
        self.prompt = prompt or DEFAULT_GROUNDING_PROMPT
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.concurrency = concurrency

    async def ocr_document(
        self,
        pdf_path: str,
        progress: Optional[ProgressCallback] = None,
    ) -> GroundedResponse:
        import fitz
        from PIL import Image, ImageSequence
        from openai import AsyncOpenAI

        from src.pdf_ocr.core.pdf import _is_image_path

        # 1. Rasterize every page, remembering dimensions.
        # For image inputs (JPEG/PNG/TIFF) skip the PDF round-trip and read
        # pixels directly — this both saves work and supports multi-frame TIFF.
        page_imgs: list[tuple[str, int, int]] = []  # (b64, width, height)

        def _emit(img: Image.Image) -> None:
            img = img.convert("RGB")
            img.thumbnail((self.max_image_dim, self.max_image_dim))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            page_imgs.append(
                (base64.b64encode(buf.getvalue()).decode(), img.width, img.height)
            )

        if _is_image_path(pdf_path):
            with Image.open(pdf_path) as src:
                for frame in ImageSequence.Iterator(src):
                    _emit(frame.copy())
        else:
            doc = fitz.open(pdf_path)
            try:
                for page in doc:
                    pix = page.get_pixmap(dpi=self.dpi)
                    _emit(Image.open(io.BytesIO(pix.tobytes("jpg", jpg_quality=70))))
            finally:
                doc.close()

        # 2. Call the VLM per page, streaming progress and isolating failures
        # so one bad page doesn't tank a multi-page document.
        client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key, timeout=self.timeout_s)
        sem = asyncio.Semaphore(max(1, self.concurrency))
        total_pages = len(page_imgs)

        async def run_one(page_idx: int) -> tuple[int, list[GroundedBlock]]:
            b64, w, h = page_imgs[page_idx]
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=self.model, temperature=0.0, max_tokens=self.max_tokens,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                }},
                            ],
                        }],
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    return page_idx, _parse_grounded_json(text, page_idx, w, h)
                except Exception as e:
                    # Per-page isolation: log the failure and return zero blocks
                    # for this page so surviving pages still land in the output.
                    logging.warning(
                        f"grounded OCR failed for page {page_idx}: "
                        f"{type(e).__name__}: {e}"
                    )
                    return page_idx, []

        tasks = [asyncio.create_task(run_one(i)) for i in range(total_pages)]
        blocks_by_page: dict[int, list[GroundedBlock]] = {}
        completed = 0
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

        # Flatten in page order for a stable, deterministic output.
        flat_blocks: list[GroundedBlock] = []
        for page_idx in range(total_pages):
            flat_blocks.extend(blocks_by_page.get(page_idx, []))
        return GroundedResponse(
            blocks=flat_blocks,
            page_sizes=[(w, h) for (_, w, h) in page_imgs],
        )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", re.IGNORECASE)
_BARE_ARRAY = re.compile(r"(\[[\s\S]*\])")


def _parse_grounded_json(
    text: str, page_idx: int, img_w: int, img_h: int,
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

    blocks: list[GroundedBlock] = []
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
        blocks.append(GroundedBlock(
            bbox=[_clamp(x0 / img_w), _clamp(y0 / img_h),
                  _clamp(x1 / img_w), _clamp(y1 / img_h)],
            text=content,
            page_index=page_idx,
        ))
    return blocks
