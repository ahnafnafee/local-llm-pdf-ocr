"""
Tests for `pdf_ocr.core.text.TextHandler` — the plain-text dump writer.

It's the simplest writer (no rasterization, no headers, no layout math),
so the tests cover the document shape, reading-order preservation, the
empty-box / empty-page short-circuits, and signature parity.
"""

from __future__ import annotations

from pathlib import Path

from pdf_ocr.core.text import TextHandler


class TestTextDump:
    def test_text_in_reading_order_within_a_page(self, tmp_path: Path):
        pages = {0: [
            ([0.1, 0.1, 0.5, 0.15], "first"),
            ([0.1, 0.2, 0.5, 0.25], "second"),
            ([0.1, 0.3, 0.5, 0.35], "third"),
        ]}
        out = tmp_path / "out.txt"
        TextHandler().embed_structured_text("fake.pdf", str(out), pages)
        txt = out.read_text(encoding="utf-8")
        assert txt == "first\nsecond\nthird\n"

    def test_pages_separated_by_blank_line_in_sorted_order(self, tmp_path: Path):
        # dict insertion order != page order — the writer must sort.
        pages = {
            1: [([0, 0, 1, 1], "page two")],
            0: [([0, 0, 1, 1], "page one")],
        }
        out = tmp_path / "out.txt"
        TextHandler().embed_structured_text("fake.pdf", str(out), pages)
        assert out.read_text(encoding="utf-8") == "page one\n\npage two\n"

    def test_empty_boxes_skipped_no_blank_runs(self, tmp_path: Path):
        pages = {0: [
            ([0, 0, 1, 1], "kept"),
            ([0, 0, 1, 1], ""),
            ([0, 0, 1, 1], "   "),       # whitespace only
            ([0, 0, 1, 1], "kept2"),
        ]}
        out = tmp_path / "out.txt"
        TextHandler().embed_structured_text("fake.pdf", str(out), pages)
        txt = out.read_text(encoding="utf-8")
        assert txt == "kept\nkept2\n"
        assert "\n\n\n" not in txt

    def test_empty_pages_drop_out(self, tmp_path: Path):
        # A middle page with no text must not leave a run of blank lines.
        pages = {
            0: [([0, 0, 1, 1], "a")],
            1: [([0, 0, 1, 1], "")],
            2: [([0, 0, 1, 1], "b")],
        }
        out = tmp_path / "out.txt"
        TextHandler().embed_structured_text("fake.pdf", str(out), pages)
        assert out.read_text(encoding="utf-8") == "a\n\nb\n"

    def test_fully_empty_doc_writes_empty_file(self, tmp_path: Path):
        out = tmp_path / "out.txt"
        TextHandler().embed_structured_text("fake.pdf", str(out), {0: [], 1: []})
        assert out.read_text(encoding="utf-8") == ""

    def test_dpi_parameter_accepted_for_signature_parity(self, tmp_path: Path):
        out = tmp_path / "out.txt"
        TextHandler().embed_structured_text(
            "fake.pdf", str(out), {0: [([0, 0, 1, 1], "ok")]}, dpi=300,
        )
        assert out.read_text(encoding="utf-8") == "ok\n"
