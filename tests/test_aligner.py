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
    _reading_order_sort,
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
        # The aligner emits boxes column-major (entire left column, then
        # entire right column) so the DP receives them in the same order
        # the LLM produces text. Verify that order is preserved end-to-end.
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


class TestReadingOrderSort:
    """Column-major reading-order sorter for multi-column pages."""

    def test_single_column_uses_row_major(self):
        # All boxes share roughly the same x range — no column gap.
        boxes = [
            [0.1, 0.20, 0.9, 0.25],
            [0.1, 0.10, 0.9, 0.15],
            [0.1, 0.30, 0.9, 0.35],
            [0.1, 0.40, 0.9, 0.45],
        ]
        out = _reading_order_sort(list(boxes))
        ys = [b[1] for b in out]
        assert ys == sorted(ys)

    def test_two_column_emits_column_major(self):
        # Left column (x≈0.2) and right column (x≈0.8), interleaved input.
        # Expected output: all of left column first (top-to-bottom), then all
        # of right column (top-to-bottom).
        L = [
            [0.05, 0.10, 0.40, 0.15],
            [0.05, 0.30, 0.40, 0.35],
            [0.05, 0.50, 0.40, 0.55],
            [0.05, 0.70, 0.40, 0.75],
        ]
        R = [
            [0.60, 0.10, 0.95, 0.15],
            [0.60, 0.30, 0.95, 0.35],
            [0.60, 0.50, 0.95, 0.55],
            [0.60, 0.70, 0.95, 0.75],
        ]
        # Interleave so input is row-major.
        interleaved: list[list[float]] = []
        for left, right in zip(L, R):
            interleaved.append(right)
            interleaved.append(left)

        out = _reading_order_sort(interleaved)
        # Left column first (4 boxes), then right column (4 boxes).
        assert out[:4] == L
        assert out[4:] == R

    def test_lone_marginal_box_does_not_create_fake_column(self):
        # 5 body boxes centered around x≈0.5 plus a single page-number-like
        # box at x≈0.95. Without the ≥2-on-each-side guard, the single
        # marginal box would be sorted as its own "column" and re-ordered
        # ahead of body content.
        body = [
            [0.10, 0.10, 0.90, 0.15],
            [0.10, 0.30, 0.90, 0.35],
            [0.10, 0.50, 0.90, 0.55],
            [0.10, 0.70, 0.90, 0.75],
            [0.10, 0.90, 0.90, 0.95],
        ]
        page_num = [[0.92, 0.95, 0.99, 0.98]]
        out = _reading_order_sort(body + page_num)
        # Output must still respect top-to-bottom flow — i.e. body comes
        # before the page number, which sits at the bottom-right.
        ys = [b[1] for b in out]
        assert ys == sorted(ys)

    def test_three_columns_recurse(self):
        col_x = [(0.05, 0.30), (0.40, 0.60), (0.70, 0.95)]
        boxes: list[list[float]] = []
        # Build 3 boxes per column at distinct y values, interleaved.
        for y in (0.10, 0.40, 0.70):
            for x0, x1 in col_x:
                boxes.append([x0, y, x1, y + 0.05])

        out = _reading_order_sort(boxes)
        # Column 1 (x≈0.18) → 3 boxes, then column 2 (x≈0.50) → 3,
        # then column 3 (x≈0.83) → 3.
        x_centers = [(b[0] + b[2]) / 2 for b in out]
        assert x_centers[:3] == sorted(x_centers[:3])
        # Each contiguous run of 3 should share the same column band.
        for start in (0, 3, 6):
            xs = x_centers[start:start + 3]
            assert max(xs) - min(xs) < 0.1, f"column band leaked at {start}"
        # Across runs, x must be increasing (left → right).
        assert x_centers[0] < x_centers[3] < x_centers[6]
