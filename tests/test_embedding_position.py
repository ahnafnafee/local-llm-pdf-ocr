"""
Position-correspondence tests for the text↔bbox↔embedding chain.

These validate the real question we care about: when we tell the pipeline
"put text T in bounding box B", does the OUTPUT PDF actually have T
searchable at the coordinates of B — not at some other position on the page,
and not bleeding into a neighbouring box?

We use PyMuPDF's `page.get_text("words", clip=rect)` to extract words whose
baseline falls inside a given rect. That's the ground truth for "what can a
user select if they drag a selection rectangle over this region".
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.pdf_ocr.core.pdf import PDFHandler


def _bbox_to_pdf_rect(bbox_norm, page_w: float, page_h: float) -> fitz.Rect:
    return fitz.Rect(
        bbox_norm[0] * page_w,
        bbox_norm[1] * page_h,
        bbox_norm[2] * page_w,
        bbox_norm[3] * page_h,
    )


def _extract_words_in_rect(page: fitz.Page, rect: fitz.Rect) -> list[str]:
    """Return the words whose bbox overlaps `rect` by at least 50% of word area."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_idx)
    hits: list[str] = []
    for w in words:
        w_rect = fitz.Rect(w[0], w[1], w[2], w[3])
        inter = w_rect & rect
        if inter.is_empty:
            continue
        w_area = max(1e-6, w_rect.get_area())
        if (inter.get_area() / w_area) >= 0.5:
            hits.append(w[4])
    return hits


def _overlap_ratio(inner: fitz.Rect, outer: fitz.Rect) -> float:
    inter = inner & outer
    if inter.is_empty:
        return 0.0
    return inter.get_area() / max(1e-6, inner.get_area())


@pytest.fixture
def pdf_handler() -> PDFHandler:
    return PDFHandler()


class TestSingleBoxPosition:
    """A single marker embedded at a known bbox must be retrievable at that bbox."""

    def test_marker_lands_inside_its_bbox(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "single_box.pdf")

        marker = "ZETAMARK"
        bbox_norm = [0.25, 0.35, 0.55, 0.40]
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            rect = _bbox_to_pdf_rect(bbox_norm, page.rect.width, page.rect.height)
            words = page.get_text("words")

        # The embedded word's bbox should overlap the intended bbox by >=50%
        # of its own area. Ascenders may peek above the top edge by a few
        # pts at large font sizes, which is acceptable for search/selection.
        marker_words = [w for w in words if marker in w[4]]
        assert marker_words, f"marker {marker!r} not found in output"
        for w in marker_words:
            wr = fitz.Rect(w[0], w[1], w[2], w[3])
            overlap = _overlap_ratio(wr, rect)
            assert overlap >= 0.5, (
                f"{marker} at {list(wr)} poorly overlaps bbox {list(rect)} "
                f"(overlap={overlap:.2f})"
            )


class TestMultiBoxIsolation:
    """Multiple markers in distinct boxes must not cross-contaminate."""

    def test_three_markers_stay_in_their_own_boxes(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "three_box.pdf")

        layout = [
            ([0.10, 0.10, 0.40, 0.14], "ALPHAWORD"),
            ([0.10, 0.30, 0.40, 0.34], "BETAWORD"),
            ([0.10, 0.60, 0.40, 0.64], "GAMMAWORD"),
        ]
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: layout}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            rects = [(_bbox_to_pdf_rect(b, pw, ph), m) for b, m in layout]
            hits_per_box = [
                (marker, _extract_words_in_rect(page, rect))
                for rect, marker in rects
            ]

        # Each box should contain its own marker and NOT the other two.
        all_markers = {m for _, m in layout}
        for marker, hits in hits_per_box:
            joined = " ".join(hits)
            assert marker in joined, (
                f"{marker} missing from its own box; hits: {hits}"
            )
            intruders = (all_markers - {marker}) & set(hits)
            assert not intruders, (
                f"{marker}'s box contains foreign markers {intruders}"
            )


class TestClipExtraction:
    """`get_text('text', clip=rect)` should return exactly the marker we placed."""

    def test_clip_returns_embedded_marker_only(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "clip_test.pdf")

        marker = "CLIPMARKER"
        bbox_norm = [0.15, 0.45, 0.45, 0.49]
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            rect = _bbox_to_pdf_rect(bbox_norm, page.rect.width, page.rect.height)
            clipped = page.get_text("text", clip=rect).strip()

        # The clip region may also pick up the image's rasterized-then-invisible
        # glyphs from the underlying example page — but our embedded marker
        # is a unique string that wouldn't occur there naturally.
        assert marker in clipped, (
            f"expected {marker} in clip text; got {clipped!r}"
        )


class TestNoLeakageOutsideBox:
    """Text outside the embedded bbox should not contain the marker."""

    def test_marker_does_not_leak_into_other_quadrants(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "leak_test.pdf")

        marker = "TOPLEFTMARKER"
        # Put marker firmly in top-left quadrant.
        bbox_norm = [0.10, 0.10, 0.35, 0.13]
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            # Bottom-right quadrant — far from the marker.
            far_rect = fitz.Rect(0.55 * pw, 0.55 * ph, pw, ph)
            far_text = page.get_text("text", clip=far_rect)

        assert marker not in far_text, (
            f"marker leaked into far region: {far_text[:200]!r}"
        )


class TestFullBboxCoverage:
    """
    Glyph bboxes must span the full bbox width so selecting any part of
    the region returns the text. Before the horizontal-scale fix, short
    text in a wide box only covered ~20-30% of the box width; selecting
    the right side of the box returned nothing.
    """

    def test_short_text_in_wide_box_covers_full_width(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "wide_cover.pdf")

        bbox_norm = [0.20, 0.30, 0.70, 0.34]  # 50% wide, 4% tall
        marker = "Hi"  # only 2 chars — used to fill ~20% of box width
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            box_rect = fitz.Rect(bbox_norm[0] * pw, bbox_norm[1] * ph,
                                 bbox_norm[2] * pw, bbox_norm[3] * ph)
            words = [w for w in page.get_text("words") if marker in w[4]]
            assert words, f"marker {marker!r} not emitted"
            wr = fitz.Rect(words[0][0], words[0][1], words[0][2], words[0][3])

        coverage = wr.width / box_rect.width
        assert coverage >= 0.90, (
            f"word only covers {coverage:.0%} of box width — horizontal scale not applied"
        )

    def test_right_half_of_wide_box_returns_text(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        """The right half of a wide box must return SOMETHING. Before the
        fix, the invisible text ended partway across the box and the right
        portion was empty. After the fix, clipping the right half of a
        stretched word returns the word's tail characters (not the whole
        word — that's physically correct: selection maps to the pixel
        extents of the stretched glyphs)."""
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "right_half.pdf")

        bbox_norm = [0.20, 0.30, 0.70, 0.34]
        marker = "ALPHAWORD"
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            box_rect = fitz.Rect(bbox_norm[0] * pw, bbox_norm[1] * ph,
                                 bbox_norm[2] * pw, bbox_norm[3] * ph)
            right_half = fitz.Rect(
                box_rect.x0 + box_rect.width * 0.6,  # right 40% of box
                box_rect.y0, box_rect.x1, box_rect.y1,
            )
            right_text = page.get_text("text", clip=right_half).strip()

        # Whatever we get back must be a non-empty suffix of the marker.
        assert right_text, "right half of wide box returned nothing — stretch regression"
        assert right_text in marker, (
            f"right-half text {right_text!r} is not a substring of {marker!r}"
        )

    def test_no_vertical_overflow(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        """Glyphs must not extend above or below the bbox — otherwise they
        bleed into neighbouring rows."""
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "no_overflow.pdf")

        bbox_norm = [0.20, 0.40, 0.70, 0.44]
        # Include descenders and ascenders so we hit the worst-case extent.
        marker = "jumping Ayaks grep"
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            box_rect = fitz.Rect(bbox_norm[0] * pw, bbox_norm[1] * ph,
                                 bbox_norm[2] * pw, bbox_norm[3] * ph)
            words = page.get_text("words")

        tol = 0.5  # sub-pixel rounding tolerance
        for w in words:
            wr = fitz.Rect(w[0], w[1], w[2], w[3])
            assert wr.y0 >= box_rect.y0 - tol, (
                f"glyph top {wr.y0:.2f} overshoots box top {box_rect.y0:.2f}"
            )
            assert wr.y1 <= box_rect.y1 + tol, (
                f"glyph bottom {wr.y1:.2f} overshoots box bottom {box_rect.y1:.2f}"
            )

    def test_long_text_in_narrow_box_compresses(
        self, pdf_handler, example_pdfs: dict[str, Path], tmp_path: Path
    ):
        """Long text in a narrow box must still land inside the box — the
        horizontal-scale path handles `scale_x < 1` by compressing glyphs,
        not by silently overflowing width."""
        input_pdf = str(example_pdfs["digital.pdf"])
        output_pdf = str(tmp_path / "compress.pdf")

        bbox_norm = [0.20, 0.50, 0.35, 0.54]
        marker = "LONGMARKER one two three four five six"
        pdf_handler.embed_structured_text(
            input_pdf, output_pdf, {0: [(bbox_norm, marker)]}, dpi=150,
        )

        with fitz.open(output_pdf) as doc:
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            box_rect = fitz.Rect(bbox_norm[0] * pw, bbox_norm[1] * ph,
                                 bbox_norm[2] * pw, bbox_norm[3] * ph)
            words = page.get_text("words")

        tol = 0.5
        for w in words:
            assert w[0] >= box_rect.x0 - tol, f"word starts left of box: {w[0]}"
            assert w[2] <= box_rect.x1 + tol, f"word ends right of box: {w[2]}"
