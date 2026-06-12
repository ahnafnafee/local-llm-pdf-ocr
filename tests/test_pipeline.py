"""
OCRPipeline orchestration tests with fully stubbed dependencies.

These validate wiring, concurrency, progress callbacks, and the refine
fallback without loading Surya or contacting the LLM.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from pdf_ocr.pipeline import (
    OCRPipeline,
    _drop_overlap_duplicates,
    _drop_refined_duplicates,
    _is_refinable,
    _page_is_dense_print,
    parse_page_range,
)


def _make_tiny_b64_image() -> str:
    # Paint dark stripes so any cropped sub-region has enough pixel variance
    # to pass the refine-stage blank-crop guard. A pure-white image trips
    # is_blank_crop and short-circuits the refine path under test.
    from PIL import ImageDraw
    img = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(img)
    for y in range(0, 300, 20):
        draw.rectangle([0, y, 300, y + 5], fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _StubAligner:
    def __init__(self, boxes_per_page=None, alignment=None):
        self.boxes = boxes_per_page or [
            [0.1, 0.1, 0.9, 0.15],
            [0.1, 0.2, 0.9, 0.25],
            [0.1, 0.3, 0.9, 0.35],
        ]
        self.alignment = alignment  # callable or None (identity)

    def get_detected_boxes_batch(self, images):
        return [list(self.boxes) for _ in images]

    def align_text(self, structured, lines):
        if self.alignment:
            return self.alignment(structured, lines)
        # Default: populate every box with the corresponding line (padding/truncating).
        out = []
        for i, (box, _) in enumerate(structured):
            out.append((box, lines[i] if i < len(lines) else ""))
        return out


class _StubPDF:
    def __init__(self, n_pages: int = 2):
        self.n_pages = n_pages
        self.last_pages = None

    def convert_to_images(self, path, dpi=150, max_image_dim=1024):
        return {i: _make_tiny_b64_image() for i in range(self.n_pages)}

    def embed_structured_text(self, inp, out, pages, dpi):
        self.last_pages = dict(pages)


class TestMinBoxConfidence:
    """Opt-in pre-alignment confidence cut. Off by default; boxes with
    no confidence attribute (plain lists, custom aligners) always pass."""

    def _boxes(self):
        from pdf_ocr.core.geometry import OrientedBox
        return [
            OrientedBox([0.1, 0.1, 0.9, 0.15], confidence=1.0),
            OrientedBox([0.1, 0.2, 0.9, 0.25], confidence=0.05),
            [0.1, 0.3, 0.9, 0.35],
        ]

    def _recording_aligner(self, seen: list):
        aligner = _StubAligner(boxes_per_page=self._boxes())
        inner = aligner.align_text

        def align(structured, lines):
            seen.append(len(structured))
            return inner(structured, lines)

        aligner.align_text = align
        return aligner

    async def test_low_confidence_boxes_dropped_before_alignment(self, stub_ocr):
        seen: list[int] = []
        pdf = _StubPDF(n_pages=1)
        pipeline = OCRPipeline(
            aligner=self._recording_aligner(seen),
            ocr_processor=stub_ocr,
            pdf_handler=pdf,
        )
        await pipeline.run(
            "in.pdf", "out.pdf", refine=False, min_box_confidence=0.2,
        )
        assert seen == [2]
        confs = [getattr(b, "confidence", None) for b, _ in pdf.last_pages[0]]
        assert confs == [1.0, None]  # 0.05 dropped; confidence-less kept

    async def test_off_by_default_keeps_all_boxes(self, stub_ocr):
        seen: list[int] = []
        pdf = _StubPDF(n_pages=1)
        pipeline = OCRPipeline(
            aligner=self._recording_aligner(seen),
            ocr_processor=stub_ocr,
            pdf_handler=pdf,
        )
        await pipeline.run("in.pdf", "out.pdf", refine=False)
        assert seen == [3]


class TestParsePageRange:
    def test_single_page(self):
        assert parse_page_range("3", 10) == [2]

    def test_range(self):
        assert parse_page_range("1-3", 10) == [0, 1, 2]

    def test_mixed(self):
        assert parse_page_range("1-3,5,7-9", 10) == [0, 1, 2, 4, 6, 7, 8]

    def test_out_of_range_clipped(self):
        assert parse_page_range("8-12", 5) == []  # none in range

    def test_duplicates_collapsed(self):
        assert parse_page_range("1,1,2-3,3", 5) == [0, 1, 2]


class TestDropRefinedDuplicates:
    """Post-refine dedup pass: refined text that already exists in a
    nearby matched box gets dropped, so the OCR text layer doesn't
    contain the same line twice."""

    def _box(self, idx: int) -> list[float]:
        # Generate a benign distinct bbox for each index so list ops
        # treat them as separate entries. Coordinates don't matter for
        # the dedup logic — it's index-based.
        return [0.1, 0.05 * idx, 0.9, 0.05 * idx + 0.04]

    def test_drops_exact_duplicate_in_adjacent_matched_box(self):
        # Refined text equals the matched neighbor — drop refined.
        boxes = [
            (self._box(0), "HEALTH INTAKE FORM"),       # matched
            (self._box(1), "HEALTH INTAKE FORM"),       # refined (dup)
        ]
        _drop_refined_duplicates(boxes, refined_indices={1})
        assert boxes[0][1] == "HEALTH INTAKE FORM"      # matched kept
        assert boxes[1][1] == ""                        # refined dropped

    def test_drops_substring_of_concatenated_neighbor(self):
        # The DP can attach a skip_line to a matched box, producing a
        # concatenated string. Refine then re-OCRs the lost line's box
        # and produces the substring — drop the refined copy.
        boxes = [
            (self._box(0), "HEALTH INTAKE FORM Please fill out the form."),
            (self._box(1), "HEALTH INTAKE FORM"),  # refined (substring)
        ]
        _drop_refined_duplicates(boxes, refined_indices={1})
        assert boxes[1][1] == ""

    def test_keeps_distinct_text_in_adjacent_box(self):
        # Two real entries that just happen to look similar — keep both.
        boxes = [
            (self._box(0), "Vyvanse (25mg) daily for attention"),    # matched
            (self._box(1), "Vyranse (25mg) daily for attention"),    # refined, real 2nd entry
        ]
        _drop_refined_duplicates(boxes, refined_indices={1})
        assert boxes[1][1] == "Vyranse (25mg) daily for attention"

    def test_case_and_whitespace_normalized(self):
        boxes = [
            (self._box(0), "Date: 9/14/19"),                # matched
            (self._box(1), "  date:    9/14/19  "),         # refined, sloppy
        ]
        _drop_refined_duplicates(boxes, refined_indices={1})
        assert boxes[1][1] == ""

    def test_does_not_compare_two_refined_against_each_other(self):
        # Both boxes are refined: each could be a real recovery, even if
        # they happen to OCR identically. Don't drop either — that's the
        # caller's job at a higher layer if it matters.
        boxes = [
            (self._box(0), "same content"),
            (self._box(1), "same content"),
        ]
        _drop_refined_duplicates(boxes, refined_indices={0, 1})
        assert boxes[0][1] == "same content"
        assert boxes[1][1] == "same content"

    def test_respects_search_radius(self):
        # Refined box is far from the matching box — no dedup.
        boxes = [(self._box(i), "filler") for i in range(20)]
        boxes[0] = (self._box(0), "the line")
        boxes[15] = (self._box(15), "the line")
        _drop_refined_duplicates(boxes, refined_indices={15}, radius=4)
        assert boxes[15][1] == "the line"  # too far, not deduped

    def test_empty_refined_text_skipped(self):
        # Refined returned empty (e.g. blank-crop short circuit) — leave it.
        boxes = [
            (self._box(0), "real content"),
            (self._box(1), ""),
        ]
        _drop_refined_duplicates(boxes, refined_indices={1})
        assert boxes[1][1] == ""


class TestDropOverlapDuplicates:
    """Per-box OCR dedup: detectors sometimes emit both a paragraph/row
    box and its constituent line/cell boxes for the same region. Per-box
    OCR transcribes each independently, so the same content lands in the
    text layer twice at the same location — and text extraction then
    interleaves the copies into muddle. Dedup requires BOTH spatial
    overlap AND text containment; never fuzzy similarity (dense forms
    legitimately repeat near-identical labels at different positions)."""

    def test_contained_text_cleared_keeps_superset(self):
        # Fee-table row: narrow cell box inside a wide row-level box that
        # read the whole row. The cell's text is a prefix of the row's.
        boxes = [
            (
                [0.12, 0.740, 0.41, 0.750],
                "For each printed enquiry numbered in the form",
            ),
            (
                [0.12, 0.744, 0.83, 0.762],
                "For each printed enquiry numbered in the form For any and "
                "each further enquiry added by solicitors",
            ),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[0][1] == ""
        assert boxes[1][1] != ""

    def test_identical_text_keeps_tighter_box(self):
        # Paragraph box and its first-line box read the same content; the
        # tighter line box is the more precise carrier — clear the big one.
        line = [0.05, 0.356, 0.94, 0.368]
        para = [0.05, 0.362, 0.94, 0.419]
        boxes = [
            (line, "Under arrangements made between the Councils"),
            (para, "Under arrangements made between the Councils"),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[0][1] != ""
        assert boxes[1][1] == ""

    def test_identical_boxes_keep_exactly_one(self):
        bbox = [0.1, 0.1, 0.9, 0.15]
        boxes = [
            (list(bbox), "duplicate detection"),
            (list(bbox), "duplicate detection"),
        ]
        _drop_overlap_duplicates(boxes)
        kept = [t for _, t in boxes if t]
        assert kept == ["duplicate detection"]

    def test_distinct_overlapping_texts_kept(self):
        # Adjacent lines whose boxes overlap but carry different content.
        boxes = [
            ([0.05, 0.360, 0.94, 0.370], "alpha beta gamma delta"),
            ([0.05, 0.365, 0.94, 0.375], "epsilon zeta eta theta"),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[0][1] and boxes[1][1]

    def test_identical_text_without_overlap_kept(self):
        # Repeated form labels at different page positions are real
        # content, not duplicates.
        boxes = [
            ([0.05, 0.17, 0.46, 0.19], "NAME AND ADDRESS (IN BLOCK LETTERS)"),
            ([0.16, 0.79, 0.49, 0.81], "NAME AND ADDRESS (IN BLOCK LETTERS)"),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[0][1] and boxes[1][1]

    def test_below_overlap_threshold_kept(self):
        # Contained text but only a sliver of spatial overlap: vertically
        # adjacent boxes brushing each other are not the same region.
        boxes = [
            ([0.1, 0.10, 0.9, 0.20], "short text here"),
            ([0.1, 0.19, 0.9, 0.30], "short text here and much more after"),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[0][1] and boxes[1][1]

    def test_short_contained_text_kept(self):
        # Tiny strings ("£ p" column headers) coincidentally appear inside
        # any row text — never clear them.
        boxes = [
            ([0.40, 0.744, 0.46, 0.754], "£ p"),
            ([0.12, 0.744, 0.83, 0.762], "fees table £ p row content here"),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[0][1] == "£ p"

    def test_empty_text_ignored(self):
        boxes = [
            ([0.1, 0.1, 0.9, 0.15], ""),
            ([0.1, 0.12, 0.9, 0.17], "real content here"),
        ]
        _drop_overlap_duplicates(boxes)
        assert boxes[1][1] == "real content here"


class _SeqCropOCR:
    """Stub whose per-crop responses come from a queue (call order)."""

    def __init__(self, crop_texts: list[str]):
        self.crop_texts = list(crop_texts)
        self.page_calls = 0
        self.crop_calls = 0

    async def perform_ocr(self, image_base64: str) -> list[str]:
        self.page_calls += 1
        return ["unused"]

    async def perform_ocr_on_crop(self, image_base64: str) -> str:
        self.crop_calls += 1
        return self.crop_texts.pop(0) if self.crop_texts else ""


class TestPerBoxOverlapDedupWiring:
    async def test_dense_path_drops_contained_overlap_duplicate(self):
        # Two overlapping detections for one table row: a narrow cell box
        # and a wide row box. Per-box OCR fills both; the dedup must clear
        # the contained copy before output.
        narrow = [0.1, 0.70, 0.45, 0.75]
        wide = [0.1, 0.72, 0.85, 0.80]
        ocr = _SeqCropOCR([
            "left cell line",
            "left cell line plus the right cell content",
        ])
        pdf = _StubPDF(n_pages=1)
        pipe = OCRPipeline(
            _StubAligner(boxes_per_page=[narrow, wide]), ocr, pdf
        )
        result = await pipe.run(
            "in.pdf", "out.pdf",
            concurrency=1, refine=False, dense_mode="always",
        )
        assert ocr.crop_calls == 2
        texts = [t for _, t in pdf.last_pages[0]]
        assert texts == ["", "left cell line plus the right cell content"]
        # The page's returned lines reflect the dedup too.
        assert result[0] == ["left cell line plus the right cell content"]


class TestPageIsDensePrint:
    """Masking gate: many short, straight lines = dense machine print."""

    def test_dense_print_page_qualifies(self):
        boxes = [[0.1, 0.1 + i * 0.012, 0.9, 0.11 + i * 0.012] for i in range(40)]
        assert _page_is_dense_print(boxes)

    def test_taller_handwriting_lines_rejected(self):
        # Median height at handwriting scale (~1.6% of the page).
        boxes = [[0.1, 0.1 + i * 0.02, 0.9, 0.116 + i * 0.02] for i in range(40)]
        assert not _page_is_dense_print(boxes)

    def test_tilted_lines_rejected(self):
        from pdf_ocr.core.geometry import OrientedBox

        boxes = [
            OrientedBox(
                [0.1, 0.1 + i * 0.012, 0.9, 0.11 + i * 0.012],
                angle=5.0 if i % 4 == 0 else 0.0,  # 25% tilted
            )
            for i in range(40)
        ]
        assert not _page_is_dense_print(boxes)

    def test_empty_page_rejected(self):
        assert not _page_is_dense_print([])


# Print-like page: 10 short straight rows (median height 0.01, no tilt)
# so the masking gate fires. Heights stay above _is_refinable's 0.008
# floor.
_PRINT_BOXES = [[0.1, 0.1 + i * 0.05, 0.9, 0.11 + i * 0.05] for i in range(10)]


class TestMaskableNeighbors:
    """Containment-aware mask selection: a paragraph box must still see
    its own line boxes (and read all its content), while each line box
    masks the paragraph's remainder and reads only itself — the dedup
    then collapses the contained copy."""

    PARA = [0.05, 0.36, 0.94, 0.42]
    LINE = [0.05, 0.356, 0.94, 0.368]   # ~half inside PARA
    OTHER = [0.05, 0.50, 0.94, 0.51]    # disjoint line elsewhere

    def test_contained_neighbor_left_visible_to_its_parent(self):
        from pdf_ocr.pipeline import _maskable_neighbors

        boxes = [self.PARA, self.LINE, self.OTHER]
        masks = _maskable_neighbors(0, boxes)
        assert self.LINE not in masks      # parent reads its own content
        assert self.OTHER in masks         # unrelated line still masked

    def test_parent_still_masked_for_the_contained_box(self):
        from pdf_ocr.pipeline import _maskable_neighbors

        boxes = [self.PARA, self.LINE, self.OTHER]
        masks = _maskable_neighbors(1, boxes)
        # Asymmetric: only a sliver of PARA sits inside LINE, so LINE's
        # crop paints PARA's remainder out and reads only itself.
        assert self.PARA in masks
        assert self.OTHER in masks

    def test_adjacent_boxes_mask_each_other(self):
        from pdf_ocr.pipeline import _maskable_neighbors

        a = [0.1, 0.10, 0.9, 0.11]
        b = [0.1, 0.112, 0.9, 0.122]
        assert _maskable_neighbors(0, [a, b]) == [b]
        assert _maskable_neighbors(1, [a, b]) == [a]


class TestNeighborMaskWiring:
    """On dense-print pages both crop paths must tell crop_for_ocr about
    the page's other boxes so neighbouring lines can be painted out of
    each crop; handwriting-like pages must NOT mask (stroke extents
    exceed the detected boxes, so masking amputates the target's ink)."""

    def _recording(self, monkeypatch):
        from pdf_ocr import pipeline as pl

        seen: list[dict] = []
        real = pl.crop_for_ocr

        def recording(image_b64, bbox, **kw):
            seen.append({
                "bbox": list(bbox),
                "masks": [list(m) for m in (kw.get("mask_boxes") or [])],
            })
            return real(image_b64, bbox, **kw)

        monkeypatch.setattr(pl, "crop_for_ocr", recording)
        return seen

    async def test_per_box_path_masks_other_boxes_on_print(
        self, monkeypatch, make_stub_ocr,
    ):
        seen = self._recording(monkeypatch)
        pipe = OCRPipeline(
            _StubAligner(boxes_per_page=_PRINT_BOXES),
            make_stub_ocr(), _StubPDF(n_pages=1),
        )
        await pipe.run(
            "in.pdf", "out.pdf", refine=False, dense_mode="always", concurrency=1,
        )
        assert len(seen) == len(_PRINT_BOXES)
        for call in seen:
            assert len(call["masks"]) == len(_PRINT_BOXES) - 1
            assert call["bbox"] not in call["masks"]

    async def test_per_box_path_skips_masks_on_handwriting(
        self, monkeypatch, make_stub_ocr,
    ):
        seen = self._recording(monkeypatch)
        # Default stub boxes are 0.05 tall — handwriting scale.
        pipe = OCRPipeline(_StubAligner(), make_stub_ocr(), _StubPDF(n_pages=1))
        await pipe.run(
            "in.pdf", "out.pdf", refine=False, dense_mode="always", concurrency=1,
        )
        assert len(seen) == 3
        assert all(call["masks"] == [] for call in seen)

    async def test_refine_path_masks_other_boxes_on_print(
        self, monkeypatch, make_stub_ocr,
    ):
        seen = self._recording(monkeypatch)
        ocr = make_stub_ocr(page_lines=["only one line"], crop_text="from crop")

        def alignment_with_gap(structured, lines):
            return [
                (b, lines[0] if i == 0 else "")
                for i, (b, _) in enumerate(structured)
            ]

        pipe = OCRPipeline(
            _StubAligner(
                boxes_per_page=_PRINT_BOXES[:3], alignment=alignment_with_gap,
            ),
            ocr, _StubPDF(n_pages=1),
        )
        await pipe.run("in.pdf", "out.pdf", concurrency=1, refine=True)
        # Boxes 1 and 2 were refined; each call masks the other two boxes.
        assert len(seen) == 2
        for call in seen:
            assert len(call["masks"]) == 2
            assert call["bbox"] not in call["masks"]

    async def test_refine_path_skips_masks_on_handwriting(
        self, monkeypatch, make_stub_ocr,
    ):
        seen = self._recording(monkeypatch)
        ocr = make_stub_ocr(page_lines=["only one line"], crop_text="from crop")

        def alignment_with_gap(structured, lines):
            return [
                (b, lines[0] if i == 0 else "")
                for i, (b, _) in enumerate(structured)
            ]

        pipe = OCRPipeline(
            _StubAligner(alignment=alignment_with_gap), ocr, _StubPDF(n_pages=1)
        )
        await pipe.run("in.pdf", "out.pdf", concurrency=1, refine=True)
        assert len(seen) == 2
        assert all(call["masks"] == [] for call in seen)


class TestRefinableGate:
    def test_accepts_medium_boxes(self):
        assert _is_refinable([0.1, 0.1, 0.5, 0.2])

    def test_rejects_thin_rule_lines(self):
        assert not _is_refinable([0.1, 0.1, 0.9, 0.105])

    def test_rejects_tiny_decorations(self):
        assert not _is_refinable([0.1, 0.1, 0.11, 0.11])


class TestOCRPipeline:
    async def test_basic_e2e(self, stub_ocr):
        aligner = _StubAligner()
        pdf = _StubPDF(n_pages=2)
        pipe = OCRPipeline(aligner, stub_ocr, pdf)

        result = await pipe.run("in.pdf", "out.pdf", concurrency=2, refine=False)

        assert set(result.keys()) == {0, 1}
        assert pdf.last_pages is not None
        assert set(pdf.last_pages.keys()) == {0, 1}
        # Every page got written with 3 boxes (as defined by StubAligner).
        for p in pdf.last_pages.values():
            assert len(p) == 3

    async def test_refine_fills_empty_boxes(self, make_stub_ocr):
        ocr = make_stub_ocr(page_lines=["only one line"], crop_text="from crop")

        def alignment_with_gap(structured, lines):
            # Populate box 0, leave 1 and 2 empty.
            out = []
            for i, (b, _) in enumerate(structured):
                out.append((b, lines[0] if i == 0 else ""))
            return out

        aligner = _StubAligner(alignment=alignment_with_gap)
        pdf = _StubPDF(n_pages=1)
        pipe = OCRPipeline(aligner, ocr, pdf)

        await pipe.run("in.pdf", "out.pdf", concurrency=2, refine=True)

        # 2 empty boxes per page × 1 page = 2 crop calls.
        assert ocr.crop_calls == 2
        texts = [t for _, t in pdf.last_pages[0]]
        assert texts[0] == "only one line"
        assert texts[1] == "from crop"
        assert texts[2] == "from crop"

    async def test_refine_skips_when_disabled(self, make_stub_ocr):
        ocr = make_stub_ocr(page_lines=["single"])
        aligner = _StubAligner(alignment=lambda s, l: [(s[0][0], l[0])] + [(b, "") for b, _ in s[1:]])
        pdf = _StubPDF(n_pages=1)
        pipe = OCRPipeline(aligner, ocr, pdf)

        await pipe.run("in.pdf", "out.pdf", refine=False)
        assert ocr.crop_calls == 0

    async def test_dense_mode_always_uses_per_box_ocr(self, make_stub_ocr):
        # dense_mode="always" should bypass full-page OCR entirely and OCR
        # every detected box individually via perform_ocr_on_crop.
        ocr = make_stub_ocr(page_lines=["fullpage line"], crop_text="per-box")
        aligner = _StubAligner()  # 3 boxes per page
        pdf = _StubPDF(n_pages=2)
        pipe = OCRPipeline(aligner, ocr, pdf)

        await pipe.run(
            "in.pdf", "out.pdf",
            concurrency=3, refine=False, dense_mode="always",
        )

        # 3 boxes × 2 pages = 6 per-box OCR calls. No full-page calls.
        assert ocr.page_calls == 0
        assert ocr.crop_calls == 6
        # Every box got the per-box text, not the (unused) full-page line.
        for page_boxes in pdf.last_pages.values():
            assert all(t == "per-box" for _, t in page_boxes)

    async def test_dense_mode_never_keeps_full_page(self, make_stub_ocr):
        ocr = make_stub_ocr(page_lines=["fullpage line"], crop_text="per-box")
        aligner = _StubAligner()
        pdf = _StubPDF(n_pages=1)
        pipe = OCRPipeline(aligner, ocr, pdf)

        await pipe.run(
            "in.pdf", "out.pdf", refine=False, dense_mode="never",
        )
        assert ocr.page_calls == 1
        assert ocr.crop_calls == 0

    async def test_dense_mode_auto_picks_per_box_for_dense_pages(self, make_stub_ocr):
        ocr = make_stub_ocr(page_lines=["fullpage line"], crop_text="per-box")
        # 7 rows × 10 cols = 70 boxes, each 0.09 wide × 0.13 tall — well
        # above _is_refinable's 0.03×0.008 floor so every box reaches the
        # LLM. The stub page image's horizontal stripes ensure each crop
        # has enough pixel variance to pass the blank check.
        many_boxes = [
            [c * 0.10, r * 0.13, c * 0.10 + 0.09, r * 0.13 + 0.13]
            for r in range(7) for c in range(10)
        ]
        aligner = _StubAligner(boxes_per_page=many_boxes)
        pdf = _StubPDF(n_pages=1)
        pipe = OCRPipeline(aligner, ocr, pdf)

        await pipe.run(
            "in.pdf", "out.pdf",
            concurrency=5, refine=False,
            dense_mode="auto", dense_threshold=60,
        )
        # 70 boxes > 60 threshold → per-box; full-page OCR was NOT called.
        assert ocr.page_calls == 0
        assert ocr.crop_calls == 70

    async def test_dense_mode_auto_keeps_full_page_for_sparse_pages(self, make_stub_ocr):
        ocr = make_stub_ocr(page_lines=["fullpage line"], crop_text="per-box")
        aligner = _StubAligner()  # 3 boxes per page (sparse)
        pdf = _StubPDF(n_pages=1)
        pipe = OCRPipeline(aligner, ocr, pdf)

        await pipe.run(
            "in.pdf", "out.pdf", refine=False,
            dense_mode="auto", dense_threshold=60,
        )
        assert ocr.page_calls == 1
        assert ocr.crop_calls == 0

    async def test_dense_mode_invalid_raises(self, stub_ocr):
        pipe = OCRPipeline(_StubAligner(), stub_ocr, _StubPDF(n_pages=1))
        with pytest.raises(ValueError, match="dense_mode"):
            await pipe.run("in.pdf", "out.pdf", dense_mode="invalid")

    async def test_progress_stages_all_fire(self, stub_ocr):
        aligner = _StubAligner(alignment=lambda s, l: [(b, "") for b, _ in s])
        pipe = OCRPipeline(aligner, stub_ocr, _StubPDF(n_pages=1))

        stages_seen = []

        async def cb(stage, cur, tot, msg):
            stages_seen.append(stage)

        await pipe.run("in.pdf", "out.pdf", progress=cb, refine=True)
        # All five pipeline stages should appear when refinement actually runs.
        assert set(stages_seen) == {"convert", "detect", "ocr", "refine", "embed"}

    async def test_progress_skips_refine_when_no_targets(self, stub_ocr):
        # StubAligner default fills every box, so refine has nothing to do.
        aligner = _StubAligner()
        pipe = OCRPipeline(aligner, stub_ocr, _StubPDF(n_pages=1))

        stages_seen = []

        async def cb(stage, cur, tot, msg):
            stages_seen.append(stage)

        await pipe.run("in.pdf", "out.pdf", progress=cb, refine=True)
        # Nothing to refine — the stage just doesn't emit.
        assert "refine" not in stages_seen
        assert {"convert", "detect", "ocr", "embed"}.issubset(stages_seen)

    async def test_concurrency_parameter_is_respected(self, make_stub_ocr):
        ocr = make_stub_ocr()
        pipe = OCRPipeline(_StubAligner(), ocr, _StubPDF(n_pages=4))
        await pipe.run("in.pdf", "out.pdf", concurrency=2, refine=False)
        assert ocr.page_calls == 4  # one per page

    async def test_custom_output_writer(self, stub_ocr):
        captured = {}

        def custom_writer(inp, out, pages, dpi):
            captured["called"] = True
            captured["pages"] = dict(pages)
            captured["dpi"] = dpi

        pipe = OCRPipeline(_StubAligner(), stub_ocr, _StubPDF(n_pages=1), output_writer=custom_writer)
        await pipe.run("in.pdf", "out.pdf", dpi=250, refine=False)

        assert captured.get("called") is True
        assert captured["dpi"] == 250
        assert 0 in captured["pages"]


class _FakeAlignment(list):
    """Stands in for aligner.AlignmentResult without importing the
    aligner module (which would pull Surya into these stub-only tests).
    The pipeline reads `match_count` via getattr, so any list carrying
    the attribute exercises the retry path."""

    def __init__(self, pairs, match_count: int):
        super().__init__(pairs)
        self.match_count = match_count


class TestDpConfidenceRetry:
    """A sparse page whose DP alignment matched too few boxes gets one
    per-box retry in auto mode (the form-page failure mode: many label
    boxes with near-identical match costs)."""

    BOXES = [[0.05, 0.05 + i * 0.1, 0.6, 0.12 + i * 0.1] for i in range(8)]

    @staticmethod
    def _low_confidence(structured, lines):
        return _FakeAlignment([(b, "guess") for b, _ in structured], match_count=1)

    @staticmethod
    def _high_confidence(structured, lines):
        return _FakeAlignment(
            [(b, "good") for b, _ in structured],
            match_count=len(structured),
        )

    async def test_low_match_rate_triggers_per_box_redo(self, stub_ocr):
        pdf = _StubPDF(n_pages=1)
        aligner = _StubAligner(boxes_per_page=self.BOXES, alignment=self._low_confidence)
        pipe = OCRPipeline(aligner, stub_ocr, pdf)
        await pipe.run("in.pdf", "out.pdf", dense_mode="auto")
        assert stub_ocr.page_calls == 1            # full-page attempt happened
        assert stub_ocr.crop_calls == len(self.BOXES)  # then per-box redo
        texts = [t for _b, t in pdf.last_pages[0]]
        assert texts == ["recovered"] * len(self.BOXES)

    async def test_high_match_rate_keeps_dp_result(self, stub_ocr):
        pdf = _StubPDF(n_pages=1)
        aligner = _StubAligner(boxes_per_page=self.BOXES, alignment=self._high_confidence)
        pipe = OCRPipeline(aligner, stub_ocr, pdf)
        await pipe.run("in.pdf", "out.pdf", dense_mode="auto")
        assert stub_ocr.crop_calls == 0
        assert [t for _b, t in pdf.last_pages[0]] == ["good"] * len(self.BOXES)

    async def test_small_pages_never_retry(self, stub_ocr):
        # Below the box floor a low match rate is noise, not evidence.
        boxes = self.BOXES[:3]

        def low(structured, lines):
            return _FakeAlignment([(b, "guess") for b, _ in structured], match_count=0)

        pipe = OCRPipeline(
            _StubAligner(boxes_per_page=boxes, alignment=low), stub_ocr,
            _StubPDF(n_pages=1),
        )
        await pipe.run("in.pdf", "out.pdf", dense_mode="auto", refine=False)
        assert stub_ocr.crop_calls == 0

    async def test_never_mode_disables_retry(self, stub_ocr):
        pipe = OCRPipeline(
            _StubAligner(boxes_per_page=self.BOXES, alignment=self._low_confidence),
            stub_ocr, _StubPDF(n_pages=1),
        )
        await pipe.run("in.pdf", "out.pdf", dense_mode="never", refine=False)
        assert stub_ocr.crop_calls == 0

    async def test_plain_list_alignment_skips_retry(self, stub_ocr):
        # Custom aligners that return plain lists (no match_count) keep
        # the pre-retry behavior.
        def plain(structured, lines):
            return [(b, "guess") for b, _ in structured]

        pipe = OCRPipeline(
            _StubAligner(boxes_per_page=self.BOXES, alignment=plain),
            stub_ocr, _StubPDF(n_pages=1),
        )
        await pipe.run("in.pdf", "out.pdf", dense_mode="auto", refine=False)
        assert stub_ocr.crop_calls == 0
