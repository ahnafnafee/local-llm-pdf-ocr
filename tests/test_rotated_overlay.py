"""Rotated-text overlay behavior across both writers, plus the writers'
word-granularity readiness.

PDF assertions go through real PyMuPDF text extraction: the extracted
line's `dir` vector is the ground truth for whether the invisible text
actually runs along the detected baseline angle.
"""

from __future__ import annotations

import math
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from pdf_ocr.core.geometry import OrientedBox
from pdf_ocr.core.html import HTMLHandler, MODE_FULL_HEIGHT, MODE_LETTER_SPACING
from pdf_ocr.core.pdf import PDFHandler


def _synth_png(path: Path, size=(800, 1000)) -> Path:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).rectangle([40, 40, 760, 200], fill="lightgray")
    img.save(path, format="PNG")
    return path


def _oriented_box(angle_deg: float, page_w=800.0, page_h=1000.0,
                  run=400.0, h=40.0) -> OrientedBox:
    """Rotated-rect OrientedBox like the aligner builds from Surya."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -math.sin(a), math.cos(a)
    p0 = (100.0, 200.0)
    pts = [
        p0,
        (p0[0] + run * dx, p0[1] + run * dy),
        (p0[0] + run * dx + h * px, p0[1] + run * dy + h * py),
        (p0[0] + h * px, p0[1] + h * py),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    quad = []
    for x, y in pts:
        quad.extend((x / page_w, y / page_h))
    return OrientedBox(
        [min(xs) / page_w, min(ys) / page_h, max(xs) / page_w, max(ys) / page_h],
        quad=quad, angle=angle_deg,
    )


def _extract_lines(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    try:
        lines = []
        for blk in doc[0].get_text("dict")["blocks"]:
            lines.extend(blk.get("lines", []))
        return lines
    finally:
        doc.close()


class TestPdfRotatedText:
    def test_rotated_box_embeds_text_along_the_angle(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.pdf"
        PDFHandler().embed_structured_text(
            str(src), str(out),
            {0: [(_oriented_box(10.0), "tilted handwriting line")]},
        )
        lines = _extract_lines(out)
        assert len(lines) == 1
        text = "".join(s["text"] for s in lines[0]["spans"])
        assert "tilted handwriting line" in text
        # Extraction dir is the writing direction in page coords (y
        # down): +10 deg means the line descends left-to-right.
        dx, dy = lines[0]["dir"]
        assert dx == pytest.approx(math.cos(math.radians(10)), abs=0.02)
        assert dy == pytest.approx(math.sin(math.radians(10)), abs=0.02)

    def test_negative_angle_ascends(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.pdf"
        PDFHandler().embed_structured_text(
            str(src), str(out),
            {0: [(_oriented_box(-8.0), "uphill line")]},
        )
        (line,) = _extract_lines(out)
        assert line["dir"][1] == pytest.approx(
            math.sin(math.radians(-8)), abs=0.02
        )

    def test_sub_threshold_angle_stays_horizontal(self, tmp_path: Path):
        # Min-area-rect noise on straight text must not tilt the layer.
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.pdf"
        PDFHandler().embed_structured_text(
            str(src), str(out),
            {0: [(_oriented_box(2.0), "actually straight")]},
        )
        (line,) = _extract_lines(out)
        assert line["dir"] == (1.0, 0.0)

    def test_plain_list_box_unaffected(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.pdf"
        PDFHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.2, 0.6, 0.25], "plain box")]},
        )
        (line,) = _extract_lines(out)
        assert line["dir"] == (1.0, 0.0)


class TestHtmlRotatedSpans:
    def test_rotated_box_emits_rotate_variable(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [(_oriented_box(10.0), "tilted line")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "--rotate:10deg;" in html
        # Anchored at the quad's top-left corner: 100/800 and 200/1000.
        assert "left:12.5%" in html
        assert "top:20%" in html
        # The stylesheet swings the span via the variables.
        assert "transform: rotate(var(--rotate, 0deg)) scaleX(var(--scale-x, 1));" in html
        assert "transform-origin: 0 0;" in html

    def test_sub_threshold_angle_emits_no_rotate(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [(_oriented_box(2.0), "straight enough")]},
        )
        assert "--rotate:" not in out.read_text(encoding="utf-8")

    def test_plain_list_box_emits_no_rotate(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.2, 0.6, 0.25], "plain box")]},
        )
        assert "--rotate:" not in out.read_text(encoding="utf-8")


class TestMeasuredScaleX:
    def test_scaled_mode_emits_measure_script(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out), {0: [([0.1, 0.1, 0.6, 0.15], "hello")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "measureText" in html
        assert "--scale-x" in html

    @pytest.mark.parametrize("mode", [MODE_LETTER_SPACING, MODE_FULL_HEIGHT])
    def test_other_modes_do_not_measure(self, mode, tmp_path: Path):
        # letter-spacing already stretches via spacing (scaleX would
        # double-stretch); full-height keeps natural width by design.
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler(mode=mode).embed_structured_text(
            str(src), str(out), {0: [([0.1, 0.1, 0.6, 0.15], "hello")]},
        )
        assert "measureText" not in out.read_text(encoding="utf-8")

    def test_measure_script_runs_after_fit_script(self, tmp_path: Path):
        # Fitting rewrites page widths; measurement must read post-fit
        # layout, so emission order is load-bearing.
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out), {0: [([0.1, 0.1, 0.6, 0.15], "hello")]},
        )
        html = out.read_text(encoding="utf-8")
        assert html.index("clientWidth") < html.index("measureText")


class TestWordGranularityReadiness:
    """The writers are granularity-agnostic: word-level boxes from a
    future word-level detector must render one unit per word with
    per-word geometry, with no writer changes."""

    WORDS = [
        ([0.10, 0.10, 0.22, 0.14], "alpha"),
        ([0.25, 0.10, 0.37, 0.14], "beta"),
        ([0.40, 0.10, 0.55, 0.14], "gamma"),
    ]

    def test_html_renders_one_span_per_word_box(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(str(src), str(out), {0: self.WORDS})
        html = out.read_text(encoding="utf-8")
        assert html.count('class="line"') == 3
        assert html.index("alpha") < html.index("beta") < html.index("gamma")

    def test_pdf_places_each_word_inside_its_box(self, tmp_path: Path):
        src = _synth_png(tmp_path / "src.png")
        out = tmp_path / "out.pdf"
        PDFHandler().embed_structured_text(str(src), str(out), {0: self.WORDS})
        doc = fitz.open(str(out))
        try:
            page = doc[0]
            words = {w[4]: w for w in page.get_text("words")}
            assert set(words) == {"alpha", "beta", "gamma"}
            for bbox, token in self.WORDS:
                x0_pt = words[token][0]
                # Word starts inside its own box (normalized x range).
                assert bbox[0] - 0.01 <= x0_pt / page.rect.width <= bbox[2]
        finally:
            doc.close()
