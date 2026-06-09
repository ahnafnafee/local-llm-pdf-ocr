"""
End-to-end tests for `pdf_ocr.core.markdown.MarkdownHandler`.

Markdown is the simplest writer (no rasterization, no layout math), so
the tests check the document shape, the page-header convention, the
reading-order preservation, and the empty-box short-circuit.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pdf_ocr.core.markdown import MarkdownHandler
from pdf_ocr.evaluation import load_ground_truth


def _pages_data_from_fixture(fixture_path: Path) -> dict[int, list]:
    blocks, _ = load_ground_truth(fixture_path)
    out: dict[int, list] = defaultdict(list)
    for b in blocks:
        out[b.page_index].append((b.bbox, b.text))
    return dict(out)


class TestMarkdownStructure:
    def test_h1_uses_input_filename(self, tmp_path: Path):
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text(
            "examples/digital.pdf", str(out), {0: []},
        )
        md = out.read_text(encoding="utf-8")
        assert md.startswith("# OCR output: digital.pdf")

    def test_one_h2_per_page(self, tmp_path: Path):
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text(
            "fake.pdf", str(out),
            {0: [], 1: [], 2: []},
        )
        md = out.read_text(encoding="utf-8")
        page_headers = re.findall(r"^## Page \d+$", md, flags=re.MULTILINE)
        assert page_headers == ["## Page 1", "## Page 2", "## Page 3"]

    def test_pages_emitted_in_sorted_numeric_order(self, tmp_path: Path):
        # dict insertion order != page order — the writer must sort.
        pages = {2: [], 0: [], 1: []}
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text("fake.pdf", str(out), pages)
        md = out.read_text(encoding="utf-8")
        positions = [md.index(h) for h in ("## Page 1", "## Page 2", "## Page 3")]
        assert positions == sorted(positions)

    def test_text_in_reading_order_within_a_page(self, tmp_path: Path):
        pages = {0: [
            ([0.1, 0.1, 0.5, 0.15], "first"),
            ([0.1, 0.2, 0.5, 0.25], "second"),
            ([0.1, 0.3, 0.5, 0.35], "third"),
        ]}
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text("fake.pdf", str(out), pages)
        md = out.read_text(encoding="utf-8")
        assert md.index("first") < md.index("second") < md.index("third")

    def test_empty_boxes_skipped(self, tmp_path: Path):
        pages = {0: [
            ([0.1, 0.1, 0.5, 0.15], "kept"),
            ([0.1, 0.2, 0.5, 0.25], ""),
            ([0.1, 0.3, 0.5, 0.35], "   "),     # whitespace only
            ([0.1, 0.4, 0.5, 0.45], "kept2"),
        ]}
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text("fake.pdf", str(out), pages)
        md = out.read_text(encoding="utf-8")
        # Two non-empty texts, no blank-paragraph artifacts between them.
        assert md.count("kept") == 2
        # No triple blank line (which would happen if empty boxes left
        # a "\n\n" each).
        assert "\n\n\n" not in md

    def test_dpi_parameter_accepted_for_signature_parity(self, tmp_path: Path):
        # The dpi kwarg exists only so MarkdownHandler matches the
        # OutputWriter signature shared by PDF / HTML writers.
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text(
            "fake.pdf", str(out), {0: [([0.1, 0.1, 0.5, 0.15], "ok")]}, dpi=300,
        )
        assert "ok" in out.read_text(encoding="utf-8")


class TestEndToEndFromFixture:
    def test_digital_fixture(self, tmp_path: Path):
        fixture = Path("tests/fixtures/ground_truth_digital.json")
        pages_data = _pages_data_from_fixture(fixture)
        out = tmp_path / "out.md"
        MarkdownHandler().embed_structured_text(
            "examples/digital.pdf", str(out), pages_data,
        )
        md = out.read_text(encoding="utf-8")

        # Top-level heading + at least one page heading
        assert "# OCR output: digital.pdf" in md
        assert "## Page 1" in md

        # Count: every non-empty fixture text should appear in the
        # output. We don't assert ordering across pages because the
        # fixture is single-page; within-page order is asserted in the
        # structural tests above.
        for items in pages_data.values():
            for _, text in items:
                text = text.strip()
                if text:
                    # The first 30 chars of the fixture text should be a
                    # substring of the output (loose match — markdown
                    # doesn't escape, so equality of the full block is
                    # also fine, but this is more robust to whitespace
                    # quirks).
                    snippet = text[:30].strip()
                    if snippet:
                        assert snippet in md, f"missing fixture text {snippet!r}"
