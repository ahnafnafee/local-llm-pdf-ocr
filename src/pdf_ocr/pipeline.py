"""
OCRPipeline - Shared orchestration for CLI and web entry points.

Pipeline phases:
    1. convert: rasterize the PDF to base64 images            (PDFHandler)
    2. detect:  batch-detect layout boxes                     (aligner)
    3. ocr:     LLM OCR + DP line-to-box alignment, per page  (OCRProcessor + aligner)
    4. refine:  per-box crop re-OCR for low-confidence boxes  (OCRProcessor, optional)
    5. embed:   emit the output file                          (output_writer)

Components are injected so extensions can swap any phase:
    - aligner: `get_detected_boxes_batch(list[bytes]) -> list[list[BBox]]`
      and `align_text(structured, llm_text) -> list[(BBox, str)]`
    - ocr_processor: async `perform_ocr(image_base64) -> list[str]` and
      `perform_ocr_on_crop(image_base64) -> str`
    - pdf_handler: `convert_to_images(path, dpi) -> dict[int, b64]`
    - output_writer: callable(input_path, output_path, pages_data, dpi) -> None
      (defaults to `pdf_handler.embed_structured_text`)

Progress reporting via optional async callback:
    async def progress(stage: str, current: int, total: int, message: str) -> None
    stages: "convert" | "detect" | "ocr" | "refine" | "embed"
"""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
from typing import Awaitable, Callable, Optional

from src.pdf_ocr.core.grounded import GroundedOCRBackend
from src.pdf_ocr.utils.image import crop_for_ocr

ProgressCallback = Callable[[str, int, int, str], Awaitable[None]]
OutputWriter = Callable[[str, str, dict, int], None]


def parse_page_range(page_str: str, total_pages: int) -> list[int]:
    """Parse a 1-indexed range like '1-3,5,7-9' into sorted 0-indexed pages."""
    pages: set[int] = set()
    for part in page_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            for p in range(start, end + 1):
                if 1 <= p <= total_pages:
                    pages.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                pages.add(p - 1)
    return sorted(pages)


class OCRPipeline:
    def __init__(
        self,
        aligner=None,
        ocr_processor=None,
        pdf_handler=None,
        output_writer: Optional[OutputWriter] = None,
        grounded_backend: Optional[GroundedOCRBackend] = None,
    ):
        """
        When `grounded_backend` is provided, `run()` skips Surya detection,
        LLM OCR, DP alignment, and the refine stage — the backend returns
        `(bbox, text)` pairs directly. `aligner` and `ocr_processor` are
        only required for the hybrid (default) path.
        """
        self.aligner = aligner
        self.ocr_processor = ocr_processor
        self.pdf_handler = pdf_handler
        self.grounded_backend = grounded_backend
        if pdf_handler is None:
            raise ValueError("pdf_handler is required (used for output writing)")
        self.output_writer = output_writer or pdf_handler.embed_structured_text

    async def run(
        self,
        input_path: str,
        output_path: str,
        *,
        dpi: int = 200,
        pages: Optional[str] = None,
        concurrency: int = 1,
        refine: bool = True,
        max_image_dim: int = 1024,
        dense_threshold: int = 60,
        dense_mode: str = "auto",
        progress: Optional[ProgressCallback] = None,
    ) -> dict[int, list[str]]:
        """
        Execute the full pipeline and return raw LLM text per processed page.

        If a `grounded_backend` is configured, runs the grounded path:
            backend.ocr_document(pdf) → embed (bbox, text) tuples directly.
            `concurrency`/`refine`/`max_image_dim` are ignored; `dpi` is
            still forwarded to `output_writer` (used by the default
            `PDFHandler.embed_structured_text` to rasterize the page
            background at the requested resolution).
        Otherwise runs the hybrid path (Surya + LLM + DP + optional refine).

        Args:
            refine: when True (default), re-OCR sizeable boxes that the DP
                aligner could not populate, by cropping them and sending
                each crop to the LLM. Covers table/multi-column/figure
                cases where line-to-box DP leaves gaps.
            max_image_dim: longest-edge cap (px) for page images sent to the
                LLM. Drop to ~640 for small local VLMs (e.g. GLM-OCR:1.1B)
                that crash on larger inputs.
            dense_threshold: in `dense_mode="auto"`, pages with more than
                this many detected Surya boxes use per-box OCR instead of
                full-page OCR. Full-page OCR fails on dense handwritten
                pages because the LLM hallucinates / loops; per-box bypasses
                the issue at the cost of N more LLM calls per page.
            dense_mode: ``"auto"`` (default) picks per-box OCR when a page
                exceeds ``dense_threshold`` boxes; ``"always"`` forces
                per-box for every page; ``"never"`` keeps the original
                full-page path even on dense content.
        """
        if dense_mode not in ("auto", "always", "never"):
            raise ValueError(
                f"dense_mode must be one of 'auto', 'always', 'never'; got {dense_mode!r}"
            )

        if self.grounded_backend is not None:
            return await self._run_grounded(input_path, output_path, dpi=dpi, progress=progress)

        if self.aligner is None or self.ocr_processor is None:
            raise ValueError(
                "Hybrid pipeline requires both `aligner` and `ocr_processor`. "
                "Pass a `grounded_backend=...` instead to use the grounded path."
            )

        await _notify(progress, "convert", 0, 1, "Converting PDF to images...")
        images_dict = await asyncio.to_thread(
            self.pdf_handler.convert_to_images, input_path, dpi, max_image_dim
        )
        page_nums = sorted(images_dict.keys())
        total_pages = len(page_nums)

        if pages:
            selected = set(parse_page_range(pages, total_pages))
            page_nums = [p for p in page_nums if p in selected]
        await _notify(progress, "convert", 1, 1, f"Converted {total_pages} pages.")

        # --- Phase 1: batch layout detection ---
        await _notify(progress, "detect", 0, 1, f"Detecting layout for {len(page_nums)} pages...")
        image_bytes = [base64.b64decode(images_dict[p]) for p in page_nums]
        batch_boxes = await asyncio.to_thread(
            self.aligner.get_detected_boxes_batch, image_bytes
        )
        pages_structured: dict[int, list] = {
            p: [(box, "") for box in batch_boxes[i]] for i, p in enumerate(page_nums)
        }
        await _notify(progress, "detect", 1, 1, "Layout detection complete.")

        # Decide per-box vs full-page OCR per page. Per-box is more
        # accurate on dense content but costs N times the LLM calls.
        per_box_pages: set[int] = set()
        for p_num in page_nums:
            n_boxes = len(pages_structured[p_num])
            if dense_mode == "always" or (
                dense_mode == "auto" and n_boxes > dense_threshold
            ):
                per_box_pages.add(p_num)

        # --- Phase 2: concurrent OCR (per-page strategy) ---
        pages_text: dict[int, list[str]] = {}
        semaphore = asyncio.Semaphore(max(1, concurrency))
        total = len(page_nums)

        async def process_page(p_num: int):
            if p_num in per_box_pages:
                # _ocr_per_box manages its own concurrency by acquiring
                # `semaphore` once per box. Wrapping the call in another
                # `async with semaphore:` here would deadlock when
                # concurrency=1 (outer holder waits for inner acquire).
                aligned = await self._ocr_per_box(
                    images_dict[p_num], pages_structured[p_num], semaphore
                )
                # No "raw lines" in per-box mode — each box's text IS the answer.
                llm_lines = [t for _, t in aligned if t]
                return p_num, llm_lines, aligned
            async with semaphore:
                llm_lines = await self.ocr_processor.perform_ocr(images_dict[p_num])
                if llm_lines:
                    aligned = await asyncio.to_thread(
                        self.aligner.align_text, pages_structured[p_num], llm_lines
                    )
                else:
                    aligned = pages_structured[p_num]
                return p_num, llm_lines, aligned

        completed = 0
        ocr_label = (
            "OCR" if not per_box_pages
            else f"OCR ({len(per_box_pages)} dense / "
                 f"{total - len(per_box_pages)} sparse)"
        )
        await _notify(progress, "ocr", 0, total, f"{ocr_label} (0/{total})...")
        for coro in asyncio.as_completed([process_page(p) for p in page_nums]):
            p_num, llm_lines, aligned = await coro
            pages_text[p_num] = llm_lines
            pages_structured[p_num] = aligned
            completed += 1
            await _notify(
                progress, "ocr", completed, total, f"{ocr_label} ({completed}/{total})"
            )

        # --- Phase 3: per-box crop re-OCR for low-confidence boxes ---
        # Skip refine for pages that already went through per-box OCR —
        # every box on those pages was already individually OCR'd.
        if refine:
            sparse_structured = {
                p: pages_structured[p] for p in page_nums if p not in per_box_pages
            }
            if sparse_structured:
                await self._refine_uncertain(
                    sparse_structured, images_dict, semaphore, progress
                )

        # --- Phase 4: write output ---
        await _notify(progress, "embed", 0, 1, "Writing output...")
        await asyncio.to_thread(
            self.output_writer, input_path, output_path, pages_structured, dpi
        )
        await _notify(progress, "embed", 1, 1, "Done.")
        return pages_text

    async def _run_grounded(
        self,
        input_path: str,
        output_path: str,
        *,
        dpi: int,
        progress: Optional[ProgressCallback],
    ) -> dict[int, list[str]]:
        """
        Grounded path: the backend returns (bbox, text) pairs directly.
        No Surya, no DP, no refine — the model already knows where the text is.

        Emits the `"ocr"` progress stage (not a separate `"grounded"` stage)
        so downstream progress adapters — e.g. `server._STAGE_WEIGHTS` —
        map cleanly without a dedicated weight entry. The backend is
        responsible for per-page tick emission; we forward the callback
        directly so the user sees granular progress instead of a 0 → 100
        jump during multi-page grounded runs.
        """
        response = await self.grounded_backend.ocr_document(input_path, progress=progress)

        pages_data: dict[int, list] = defaultdict(list)
        pages_text: dict[int, list[str]] = defaultdict(list)
        for block in response.blocks:
            pages_data[block.page_index].append((block.bbox, block.text))
            pages_text[block.page_index].append(block.text)

        await _notify(progress, "embed", 0, 1, "Writing output...")
        await asyncio.to_thread(
            self.output_writer, input_path, output_path, dict(pages_data), dpi
        )
        await _notify(progress, "embed", 1, 1, "Done.")
        return dict(pages_text)

    async def _ocr_per_box(
        self,
        image_b64: str,
        structured: list[tuple[list[float], str]],
        semaphore: asyncio.Semaphore,
    ) -> list[tuple[list[float], str]]:
        """
        OCR every detected box on a page individually.

        Used in dense-mode for pages where full-page OCR is unreliable
        (the LLM loops, hallucinates, or just gets confused by the volume
        of content). Each box becomes a small focused crop the model can
        transcribe accurately.

        Two filters apply *before* the LLM call:
          - :func:`_is_refinable` rejects thin rules / tiny decorations
            (same guard the refine stage uses — Surya emits boxes for
            section dividers etc. on dense handwritten pages, and
            sending those to the LLM produces hallucinated text).
          - :func:`crop_for_ocr` decodes the page once, crops the padded
            region, runs a stddev-based blank check on the *same* padded
            crop, and returns ``None`` for near-uniform regions.

        Returns ``[(bbox, text), ...]`` in the same order as ``structured``.
        Boxes that come back blank or filtered get an empty string.
        """
        async def ocr_one(idx: int, bbox: list[float]):
            async with semaphore:
                if not _is_refinable(bbox):
                    return idx, ""
                crop_b64 = await asyncio.to_thread(crop_for_ocr, image_b64, bbox)
                if crop_b64 is None:
                    return idx, ""
                text = await self.ocr_processor.perform_ocr_on_crop(crop_b64)
                return idx, text

        tasks = [
            asyncio.create_task(ocr_one(i, bbox))
            for i, (bbox, _) in enumerate(structured)
        ]
        results: dict[int, str] = {}
        for fut in asyncio.as_completed(tasks):
            idx, text = await fut
            results[idx] = text.strip()
        return [(bbox, results.get(i, "")) for i, (bbox, _) in enumerate(structured)]

    async def _refine_uncertain(
        self,
        pages_structured: dict[int, list],
        images_dict: dict[int, str],
        semaphore: asyncio.Semaphore,
        progress: Optional[ProgressCallback],
    ) -> None:
        """
        Re-OCR boxes the DP aligner couldn't populate.

        A box is flagged for refinement when:
          - Its aligned text is empty, AND
          - It is large enough to plausibly contain readable text
            (filters out thin rule lines, tiny decorative boxes).

        The crop+LLM call is done concurrently under the same semaphore as
        the page-level OCR.
        """
        targets: list[tuple[int, int, list[float]]] = []
        for p_num, aligned in pages_structured.items():
            for idx, (bbox, text) in enumerate(aligned):
                if not text.strip() and _is_refinable(bbox):
                    targets.append((p_num, idx, bbox))

        if not targets:
            return

        total = len(targets)
        await _notify(progress, "refine", 0, total, f"Refining {total} uncertain boxes...")

        async def refine_one(p_num: int, idx: int, bbox: list[float]):
            async with semaphore:
                # crop_for_ocr decodes once and runs the blank check on
                # the same padded crop the LLM would receive — returns
                # None for near-uniform regions (notebook background,
                # margins) so we skip the LLM call without polluting
                # the text layer with the model's pangram fallback.
                crop_b64 = await asyncio.to_thread(
                    crop_for_ocr, images_dict[p_num], bbox
                )
                if crop_b64 is None:
                    return p_num, idx, ""
                text = await self.ocr_processor.perform_ocr_on_crop(crop_b64)
                return p_num, idx, text

        completed = 0
        refined_indices: dict[int, set[int]] = defaultdict(set)
        for coro in asyncio.as_completed(
            [refine_one(p, i, b) for p, i, b in targets]
        ):
            p_num, idx, text = await coro
            bbox_cur, _ = pages_structured[p_num][idx]
            pages_structured[p_num][idx] = (bbox_cur, text.strip())
            refined_indices[p_num].add(idx)
            completed += 1
            await _notify(
                progress, "refine", completed, total,
                f"Refining boxes ({completed}/{total})",
            )

        # Post-refine dedup: refine sometimes produces text already present
        # in a vertically-nearby matched box, e.g. when Surya emits two
        # overlapping bboxes for the same content, or when the DP misaligned
        # a line via skip_line attachment and refine then re-OCR'd the
        # original location. Without dedup the OCR text layer ends up with
        # the same line twice, which surfaces as duplicated lines on
        # copy-paste from the output PDF.
        for p_num, idxs in refined_indices.items():
            _drop_refined_duplicates(pages_structured[p_num], idxs)


# --- helpers ----------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    """Lowercased, whitespace-collapsed form for substring comparison."""
    return " ".join(text.lower().split())


def _drop_refined_duplicates(
    page_boxes: list[tuple[list[float], str]],
    refined_indices: set[int],
    *,
    radius: int = 4,
) -> None:
    """
    Mutate ``page_boxes`` in place: clear the text of any refined box
    whose content already appears (as a substring or exact match) in a
    non-refined matched box within ``radius`` index positions.

    Why refined-only one-way: matched text came from the DP and reflects
    the LLM's reading-order emission for the page; refined text came
    from a per-box crop and is more vulnerable to neighbor-content
    bleed-through (Surya bboxes can overlap, and small crops give the
    VLM less context). When the two collide, the matched version is the
    safer keep.

    Comparison is case-insensitive and whitespace-collapsed so trivial
    formatting differences don't defeat the dedup.
    """
    for r_idx in sorted(refined_indices):
        r_bbox, r_text = page_boxes[r_idx]
        if not r_text:
            continue
        r_norm = _normalize_for_dedup(r_text)
        if not r_norm:
            continue
        lo = max(0, r_idx - radius)
        hi = min(len(page_boxes), r_idx + radius + 1)
        for o_idx in range(lo, hi):
            if o_idx == r_idx or o_idx in refined_indices:
                continue
            _, o_text = page_boxes[o_idx]
            if not o_text:
                continue
            o_norm = _normalize_for_dedup(o_text)
            if r_norm in o_norm:
                page_boxes[r_idx] = (r_bbox, "")
                break


def _is_refinable(bbox: list[float]) -> bool:
    """
    Only trigger per-box re-OCR for boxes large enough to plausibly contain
    readable text. Cutoffs are in normalized (0..1) page coordinates, tuned
    so horizontal rules and tiny marks get skipped.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    # At ~200 DPI Letter this is roughly 40pt wide x 6pt tall — a comfortable
    # floor for "could be a word". Adjust if docs use very small type.
    return width > 0.03 and height > 0.008


async def _notify(cb: Optional[ProgressCallback], stage, current, total, message):
    if cb is not None:
        await cb(stage, current, total, message)
