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

from src.pdf_ocr.pipeline import OCRPipeline, _is_refinable, parse_page_range


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
