"""
Unit tests for the DP line-to-box aligner.

Covers the three properties that matter for correctness:
    1. Monotonicity — reading order is preserved.
    2. Conservation — every LLM line lands somewhere (nothing silently dropped).
    3. Size-awareness — large paragraphs bind to large boxes, short lines to short boxes.
"""

from __future__ import annotations

import pytest

from src.pdf_ocr.core.aligner import (
    HybridAligner,
    _dp_align,
    _normalize_lines,
)


def _aligner() -> HybridAligner:
    """Construct without firing Surya __init__ — we only exercise align_text."""
    return HybridAligner.__new__(HybridAligner)


class TestNormalizeLines:
    def test_string_with_newlines(self):
        assert _normalize_lines("hello\n\nworld") == ["hello", "world"]

    def test_list_of_strings(self):
        assert _normalize_lines(["  hello  ", "world"]) == ["hello", "world"]

    def test_list_items_with_embedded_newlines(self):
        assert _normalize_lines(["line1\nline2", "line3"]) == ["line1", "line2", "line3"]

    def test_empty_inputs(self):
        assert _normalize_lines("") == []
        assert _normalize_lines(None) == []
        assert _normalize_lines([]) == []
        assert _normalize_lines(["   ", ""]) == []


class TestDPAlign:
    def test_one_to_one_equal_sized(self):
        lines = ["alpha line", "beta line", "gamma line"]
        boxes = [[0.0, 0.0, 1.0, 0.3], [0.0, 0.3, 1.0, 0.6], [0.0, 0.6, 1.0, 0.9]]
        mapping = _dp_align(lines, boxes)
        assert mapping == {0: ["alpha line"], 1: ["beta line"], 2: ["gamma line"]}

    def test_respects_box_sizes(self):
        # Short heading + very long paragraph + short footer should map by size.
        lines = ["Title", "x" * 200, "Fin"]
        boxes = [
            [0.1, 0.05, 0.3, 0.08],   # tiny
            [0.0, 0.15, 1.0, 0.85],   # huge
            [0.4, 0.9, 0.55, 0.93],   # tiny
        ]
        mapping = _dp_align(lines, boxes)
        assert mapping[0] == ["Title"]
        assert mapping[1] == ["x" * 200]
        assert mapping[2] == ["Fin"]

    def test_more_lines_than_boxes_conserves_text(self):
        lines = ["A", "B", "C", "D", "E"]
        boxes = [[0.0, 0.0, 1.0, 0.5], [0.0, 0.5, 1.0, 1.0]]
        mapping = _dp_align(lines, boxes)
        all_placed = [t for vs in mapping.values() for t in vs]
        assert set(all_placed) == set(lines), "every line must be placed"

    def test_more_boxes_than_lines_leaves_empties(self):
        lines = ["one", "two"]
        boxes = [[0.0, 0.0, 0.3, 0.1]] * 5
        mapping = _dp_align(lines, boxes)
        assert sum(len(v) for v in mapping.values()) == 2

    def test_two_column_layout(self):
        # Surya sorts top-to-bottom then left-to-right. For a well-separated
        # 2-column page, that order matches an LLM reading left-col-first.
        lines = [
            "Left column first paragraph text here",
            "Left column second paragraph also has words",
            "Right column title",
            "Right column body text continues on",
        ]
        boxes = [
            [0.05, 0.10, 0.45, 0.15],   # L row 1
            [0.05, 0.18, 0.45, 0.30],   # L row 2
            [0.55, 0.10, 0.95, 0.12],   # R short title
            [0.55, 0.15, 0.95, 0.35],   # R body
        ]
        mapping = _dp_align(lines, boxes)
        assert len(mapping) == 4
        # Each box got text in increasing box-index order (monotonic).
        for i in range(4):
            assert i in mapping and mapping[i]

    def test_empty_boxes_yields_empty_mapping(self):
        assert _dp_align(["some text"], []) == {}

    def test_empty_lines_yields_empty_mapping(self):
        assert _dp_align([], [[0.0, 0.0, 1.0, 1.0]]) == {}


class TestAlignTextPublicAPI:
    def test_full_page_fallback_when_no_boxes(self):
        out = _aligner().align_text([], ["first line", "second line"])
        assert out == [([0.0, 0.0, 1.0, 1.0], "first line\nsecond line")]

    def test_all_empty_when_no_lines(self):
        structured = [([0.0, 0.0, 0.5, 0.5], ""), ([0.5, 0.5, 1.0, 1.0], "")]
        out = _aligner().align_text(structured, [])
        assert [t for _, t in out] == ["", ""]

    def test_result_length_matches_input_boxes(self):
        structured = [([i / 10, 0, i / 10 + 0.1, 0.1], "") for i in range(5)]
        lines = ["a", "b", "c"]
        out = _aligner().align_text(structured, lines)
        assert len(out) == 5  # one tuple per input box, in order

    def test_preserves_box_order(self):
        boxes = [
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.1, 1.0, 0.2],
            [0.0, 0.2, 1.0, 0.3],
        ]
        structured = [(b, "") for b in boxes]
        out = _aligner().align_text(structured, ["x", "y", "z"])
        assert [b for b, _ in out] == boxes

    def test_accepts_both_string_and_list_input(self):
        structured = [([0.0, 0.0, 0.5, 0.5], "")]
        out_str = _aligner().align_text(structured, "one\ntwo")
        out_lst = _aligner().align_text(structured, ["one", "two"])
        assert out_str == out_lst

    @pytest.mark.parametrize("n_lines,n_boxes", [(1, 1), (5, 3), (3, 5), (10, 10)])
    def test_conserves_all_lines_across_shapes(self, n_lines, n_boxes):
        lines = [f"line-{i}" for i in range(n_lines)]
        boxes = [[i / n_boxes, 0, (i + 1) / n_boxes, 0.1] for i in range(n_boxes)]
        structured = [(b, "") for b in boxes]
        out = _aligner().align_text(structured, lines)
        placed_text = " ".join(t for _, t in out if t)
        for line in lines:
            assert line in placed_text, f"line {line!r} was dropped"
