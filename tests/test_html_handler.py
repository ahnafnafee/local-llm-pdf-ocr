"""
End-to-end tests for `pdf_ocr.core.html.HTMLHandler`.

These run *without* the LLM or Surya: ground-truth fixtures in
`tests/fixtures/ground_truth_*.json` provide real `(bbox, text)` data,
which we feed straight into the writer. That covers the writer's full
contract — input dispatch, page rendering, span emission — without
paying for a model call.

Three input modes are exercised:
    * raw image (synthesized PNG)
    * single-frame PDF (`examples/digital.pdf`)
    * multi-frame TIFF (synthesized in tmp_path)

Plus targeted behavioral tests for the three sizing modes and the two
edge cases (full-page fallback and `'\n'`-joined multi-line bboxes).
"""

from __future__ import annotations

import re
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pdf_ocr.core.html import (
    DEFAULT_MODE,
    HTMLHandler,
    MODE_FULL_HEIGHT,
    MODE_LETTER_SPACING,
    MODE_SCALED,
)
from pdf_ocr.evaluation import load_ground_truth


# --- helpers --------------------------------------------------------------


def _pages_data_from_fixture(fixture_path: Path) -> dict[int, list]:
    """Group GTBlock entries by page_index into the writer's input shape."""
    blocks, _ = load_ground_truth(fixture_path)
    out: dict[int, list] = defaultdict(list)
    for b in blocks:
        out[b.page_index].append((b.bbox, b.text))
    return dict(out)


def _make_synth_image(path: Path, size=(800, 1000)) -> Path:
    """Synthesize a small PNG with some shapes so the embedded JPEG isn't
    pure white (matters for completeness — the file is still rendered as
    a background, even if the test doesn't assert on the rendered pixels)."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 750, 250], fill="lightgray")
    draw.rectangle([50, 400, 750, 600], outline="black", width=2)
    img.save(path, format="PNG")
    return path


def _make_multiframe_tiff(path: Path, n: int = 3, size=(600, 800)) -> Path:
    frames = []
    for i in range(n):
        frame = Image.new("RGB", size, "white")
        ImageDraw.Draw(frame).text((40, 40), f"Page {i+1}", fill="black")
        frames.append(frame)
    frames[0].save(path, format="TIFF", save_all=True, append_images=frames[1:])
    return path


# --- structural assertions ------------------------------------------------


class TestHtmlStructure:
    def test_emits_doctype_and_self_closing_html(self, tmp_path: Path):
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.1, 0.5, 0.15], "hello")]},
        )
        html = out.read_text(encoding="utf-8")
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert "<title>src</title>" in html

    def test_one_page_div_per_image_frame(self, tmp_path: Path):
        # Multi-frame TIFF → one page div per frame.
        src = _make_multiframe_tiff(tmp_path / "scan.tiff", n=3)
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.1, 0.5, 0.15], "p1")],
             1: [([0.1, 0.1, 0.5, 0.15], "p2")],
             2: [([0.1, 0.1, 0.5, 0.15], "p3")]},
        )
        html = out.read_text(encoding="utf-8")
        assert html.count('class="page"') == 3
        for marker in ("p1", "p2", "p3"):
            assert marker in html

    def test_data_url_inlined_per_page_when_opted_in(self, tmp_path: Path):
        # Multi-frame TIFFs need rasterization regardless; with
        # `inline_images=True` each frame becomes a data: URL.
        src = _make_multiframe_tiff(tmp_path / "scan.tiff", n=2)
        out = tmp_path / "out.html"
        HTMLHandler(inline_images=True).embed_structured_text(
            str(src), str(out), {},
        )
        html = out.read_text(encoding="utf-8")
        urls = re.findall(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", html)
        assert len(urls) == 2
        assert all(len(u) > 100 for u in urls)
        # No sidecar files written in inline mode.
        sidecars = sorted(p.name for p in tmp_path.glob("out_p*.jpg"))
        assert sidecars == []

    def test_external_default_emits_sidecar_jpegs_for_multipage_input(
        self, tmp_path: Path,
    ):
        # Multi-frame TIFF → one sidecar JPEG per frame, no data: URLs.
        src = _make_multiframe_tiff(tmp_path / "scan.tiff", n=3)
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(str(src), str(out), {})
        html = out.read_text(encoding="utf-8")
        assert "data:image/jpeg;base64" not in html
        sidecars = sorted(p.name for p in tmp_path.glob("out_p*.jpg"))
        assert sidecars == ["out_p1.jpg", "out_p2.jpg", "out_p3.jpg"]
        for name in sidecars:
            assert f"url('{name}')" in html
            assert (tmp_path / name).stat().st_size > 0

    def test_external_default_references_browser_native_image_directly(
        self, tmp_path: Path,
    ):
        # Single-frame PNG input: no rasterization, HTML references the
        # input file directly via a relative URL.
        src = _make_synth_image(tmp_path / "scan.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.1, 0.5, 0.15], "hello")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "data:image/jpeg;base64" not in html
        assert "url('scan.png')" in html
        # No sidecars written for direct-reference inputs.
        assert sorted(tmp_path.glob("out_p*.jpg")) == []

    def test_skips_empty_text_boxes(self, tmp_path: Path):
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [
                ([0.05, 0.05, 0.5, 0.10], "kept"),
                ([0.05, 0.15, 0.5, 0.20], ""),       # skip — empty
                ([0.05, 0.25, 0.5, 0.30], "   "),    # skip — whitespace only
            ]},
        )
        html = out.read_text(encoding="utf-8")
        # Exactly one span emitted (the "kept" one).
        assert html.count('class="line"') == 1
        assert "kept" in html

    def test_page_div_uses_container_relative_sizing(self, tmp_path: Path):
        # The .page div must declare --page-w / --page-h CSS variables
        # and use container queries so spans scale with the page at any
        # zoom level. A small fit-to-viewport script must also be
        # included so the default view doesn't overflow narrow viewports.
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.1, 0.9, 0.15], "hello world")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "container-type: inline-size" in html
        assert "aspect-ratio: var(--page-w) / var(--page-h)" in html
        assert "--page-w:" in html and "--page-h:" in html
        # Spans must use container-relative units, not raw pixels.
        assert "left:10%" in html
        assert "cqw" in html
        # Inline fit-to-viewport script must be present so zoom-in works
        # symmetrically with zoom-out.
        assert "document.body.clientWidth" in html
        assert "querySelectorAll('div.page')" in html

    def test_html_special_chars_are_escaped(self, tmp_path: Path):
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.05, 0.05, 0.5, 0.10], "<script>&amp;")]},
        )
        html = out.read_text(encoding="utf-8")
        # The literal text must NOT appear unescaped — that would break parsing.
        assert "<script>&amp;" not in html.replace("&lt;script&gt;", "")
        assert "&lt;script&gt;" in html


# --- mode behavior --------------------------------------------------------


class TestSizingModes:
    @pytest.fixture
    def synth(self, tmp_path: Path) -> Path:
        return _make_synth_image(tmp_path / "src.png")

    def test_default_mode_is_scaled(self):
        # Defends the milahu-requested default: scaled keeps text
        # legible at any zoom, letter-spacing can render characters as
        # overlapping smears at small bbox widths.
        assert DEFAULT_MODE == MODE_SCALED
        assert HTMLHandler().mode == MODE_SCALED

    def test_default_does_not_emit_letter_spacing(
        self, synth: Path, tmp_path: Path,
    ):
        out = tmp_path / "default.html"
        HTMLHandler().embed_structured_text(
            str(synth), str(out),
            {0: [([0.05, 0.05, 0.95, 0.10], "abcdef")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "letter-spacing:" not in html
        assert "font-size:" in html

    def test_letter_spacing_opt_in_emits_letter_spacing_style(
        self, synth: Path, tmp_path: Path,
    ):
        out = tmp_path / "ls.html"
        HTMLHandler(mode=MODE_LETTER_SPACING).embed_structured_text(
            str(synth), str(out),
            {0: [([0.05, 0.05, 0.95, 0.10], "abcdef")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "letter-spacing:" in html

    def test_full_height_does_not_emit_letter_spacing(
        self, synth: Path, tmp_path: Path,
    ):
        out = tmp_path / "fh.html"
        HTMLHandler(mode=MODE_FULL_HEIGHT).embed_structured_text(
            str(synth), str(out),
            {0: [([0.05, 0.05, 0.95, 0.10], "abcdef")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "letter-spacing:" not in html
        assert "font-size:" in html

    def test_scaled_does_not_emit_letter_spacing(
        self, synth: Path, tmp_path: Path,
    ):
        out = tmp_path / "sc.html"
        HTMLHandler(mode=MODE_SCALED).embed_structured_text(
            str(synth), str(out),
            {0: [([0.05, 0.05, 0.95, 0.10], "abcdef")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "letter-spacing:" not in html
        assert "font-size:" in html

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown HTML mode"):
            HTMLHandler(mode="not-a-mode")


# --- edge cases routed through layout helpers -----------------------------


class TestLayoutEdgeCases:
    def test_full_page_fallback_renders_each_line_separately(self, tmp_path: Path):
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.0, 0.0, 1.0, 1.0], "fallback A\nfallback B\nfallback C")]},
        )
        html = out.read_text(encoding="utf-8")
        # Three spans, one per line. Each present in the output.
        assert html.count('class="line"') == 3
        for marker in ("fallback A", "fallback B", "fallback C"):
            assert marker in html

    def test_multi_line_real_bbox_splits_to_multiple_spans(self, tmp_path: Path):
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.2, 0.9, 0.4], "line one\nline two")]},
        )
        html = out.read_text(encoding="utf-8")
        assert html.count('class="line"') == 2
        assert "line one" in html
        assert "line two" in html

    def test_zero_dimension_bbox_skipped(self, tmp_path: Path):
        src = _make_synth_image(tmp_path / "src.png")
        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [
                ([0.5, 0.5, 0.5, 0.5], "zero box"),
                ([0.1, 0.1, 0.5, 0.15], "kept"),
            ]},
        )
        html = out.read_text(encoding="utf-8")
        # Zero-width-and-height box dropped; only the second emits.
        assert html.count('class="line"') == 1
        assert "kept" in html
        assert "zero box" not in html


# --- input dispatch -------------------------------------------------------


class TestInputDispatch:
    def test_pdf_input_writes_one_sidecar_per_page(
        self, example_pdfs, tmp_path: Path,
    ):
        out = tmp_path / "out.html"
        pages_data = {0: [([0.1, 0.1, 0.6, 0.15], "DIGITALMARKER")]}
        HTMLHandler().embed_structured_text(
            str(example_pdfs["digital.pdf"]),
            str(out),
            pages_data,
            dpi=100,
        )
        html = out.read_text(encoding="utf-8")
        assert html.count('class="page"') == 1
        assert "DIGITALMARKER" in html
        assert "data:image/jpeg;base64," not in html
        sidecar = tmp_path / "out.jpg"
        assert sidecar.exists() and sidecar.stat().st_size > 0
        assert "url('out.jpg')" in html

    def test_pdf_input_inline_mode_emits_data_url(
        self, example_pdfs, tmp_path: Path,
    ):
        out = tmp_path / "out.html"
        HTMLHandler(inline_images=True).embed_structured_text(
            str(example_pdfs["digital.pdf"]),
            str(out),
            {0: [([0.1, 0.1, 0.6, 0.15], "INLINEMARKER")]},
            dpi=100,
        )
        html = out.read_text(encoding="utf-8")
        assert "INLINEMARKER" in html
        assert "data:image/jpeg;base64," in html
        assert sorted(tmp_path.glob("out_p*.jpg")) == []

    def test_image_input_handles_avif(self, tmp_path: Path):
        # AVIF is in the browser-native allowlist, so the HTML
        # references the input file directly with no rasterization.
        img = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(img).rectangle([50, 50, 350, 250], fill="lightgray")
        src = tmp_path / "scan.avif"
        img.save(src, format="AVIF", quality=80)

        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.1, 0.5, 0.15], "AVIFMARKER")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "AVIFMARKER" in html
        assert "data:image/jpeg;base64," not in html
        assert "url('scan.avif')" in html
        assert sorted(tmp_path.glob("out_p*.jpg")) == []

    def test_bmp_input_falls_back_to_sidecar(self, tmp_path: Path):
        # BMP isn't in the browser-native allowlist, so the writer
        # rasterizes to a sidecar JPEG even though it's a single image.
        img = Image.new("RGB", (200, 150), "white")
        ImageDraw.Draw(img).rectangle([20, 20, 180, 130], fill="lightgray")
        src = tmp_path / "scan.bmp"
        img.save(src, format="BMP")

        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(src), str(out),
            {0: [([0.1, 0.1, 0.5, 0.2], "BMPMARKER")]},
        )
        html = out.read_text(encoding="utf-8")
        assert "BMPMARKER" in html
        assert "url('out.jpg')" in html
        assert (tmp_path / "out.jpg").exists()


# --- end-to-end via fixtures ----------------------------------------------


class TestEndToEndFromFixture:
    """Drive the writer with real layout data from the GT fixtures."""

    def test_digital_fixture_round_trip(self, example_pdfs, tmp_path: Path):
        fixture = Path("tests/fixtures/ground_truth_digital.json")
        pages_data = _pages_data_from_fixture(fixture)
        assert 0 in pages_data, "fixture must have at least page 0"

        out = tmp_path / "out.html"
        HTMLHandler().embed_structured_text(
            str(example_pdfs["digital.pdf"]),
            str(out), pages_data, dpi=100,
        )
        html = out.read_text(encoding="utf-8")

        # Every non-empty fixture text must appear in the HTML body
        # (the writer escapes `<>&` but those don't appear in this fixture).
        n_emitted_texts = 0
        for items in pages_data.values():
            for _, text in items:
                if text.strip():
                    n_emitted_texts += 1

        # Spans are emitted per visual line, so n_spans >= n_emitted_texts
        # (multi-line text in one bbox produces >1 span). Loose bound.
        assert html.count('class="line"') >= n_emitted_texts
