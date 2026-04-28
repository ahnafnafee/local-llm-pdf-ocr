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
        _, mapping = _dp_align(lines, boxes)
        assert mapping == {0: ["alpha line"], 1: ["beta line"], 2: ["gamma line"]}

    def test_respects_box_sizes(self):
        # Short heading + very long paragraph + short footer should map by size.
        lines = ["Title", "x" * 200, "Fin"]
        boxes = [
            [0.1, 0.05, 0.3, 0.08],   # tiny
            [0.0, 0.15, 1.0, 0.85],   # huge
            [0.4, 0.9, 0.55, 0.93],   # tiny
        ]
        _, mapping = _dp_align(lines, boxes)
        assert mapping[0] == ["Title"]
        assert mapping[1] == ["x" * 200]
        assert mapping[2] == ["Fin"]

    def test_more_lines_than_boxes_conserves_text(self):
        lines = ["A", "B", "C", "D", "E"]
        boxes = [[0.0, 0.0, 1.0, 0.5], [0.0, 0.5, 1.0, 1.0]]
        _, mapping = _dp_align(lines, boxes)
        all_placed = [t for vs in mapping.values() for t in vs]
        assert set(all_placed) == set(lines), "every line must be placed"

    def test_more_boxes_than_lines_leaves_empties(self):
        lines = ["one", "two"]
        boxes = [[0.0, 0.0, 0.3, 0.1]] * 5
        _, mapping = _dp_align(lines, boxes)
        assert sum(len(v) for v in mapping.values()) == 2

    def test_two_column_layout(self):
        # When boxes are passed in the same order the LLM emits text
        # (column-major here), the DP achieves a 1:1 monotonic match.
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
        _, mapping = _dp_align(lines, boxes)
        assert len(mapping) == 4
        # Each box got text in increasing box-index order (monotonic).
        for i in range(4):
            assert i in mapping and mapping[i]

    def test_cost_increases_with_order_mismatch(self):
        # Heterogeneous box sizes + line lengths so that the DP cost
        # depends on the box ordering. align_text relies on this.
        lines = ["tiny", "x" * 200, "x" * 200, "tiny"]
        boxes_aligned = [
            [0.0, 0.05, 0.10, 0.10],   # tiny
            [0.0, 0.15, 1.0, 0.45],    # huge
            [0.0, 0.50, 1.0, 0.80],    # huge
            [0.4, 0.85, 0.55, 0.90],   # tiny
        ]
        boxes_shuffled = [boxes_aligned[i] for i in (1, 0, 3, 2)]
        cost_aligned, _ = _dp_align(lines, boxes_aligned)
        cost_shuffled, _ = _dp_align(lines, boxes_shuffled)
        assert cost_shuffled > cost_aligned, (
            f"shuffled cost {cost_shuffled} must exceed aligned {cost_aligned}"
        )

    def test_empty_boxes_yields_empty_mapping(self):
        assert _dp_align(["some text"], []) == (0.0, {})

    def test_empty_lines_yields_empty_mapping(self):
        assert _dp_align([], [[0.0, 0.0, 1.0, 1.0]]) == (0.0, {})


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


# Two-column fixture for auto-detect tests. Heterogeneous box sizes so that
# different orderings produce different DP costs (uniform boxes cost-tie
# and can't discriminate between orderings). Left column tall, right short.
_AUTODETECT_L_BOXES = [
    [0.05, 0.05, 0.40, 0.20],
    [0.05, 0.25, 0.40, 0.40],
    [0.05, 0.45, 0.40, 0.60],
    [0.05, 0.65, 0.40, 0.80],
]
_AUTODETECT_R_BOXES = [
    [0.60, 0.05, 0.95, 0.08],
    [0.60, 0.25, 0.95, 0.28],
    [0.60, 0.45, 0.95, 0.48],
    [0.60, 0.65, 0.95, 0.68],
]
# Surya-style row-major input: L1, R1, L2, R2, L3, R3, L4, R4.
_AUTODETECT_BOXES_ROW_MAJOR = [
    box
    for pair in zip(_AUTODETECT_L_BOXES, _AUTODETECT_R_BOXES)
    for box in pair
]

_AUTODETECT_LEFT_TEXT = [
    f"L{i + 1} long body line with many words to fill a tall capacity box"
    for i in range(4)
]
_AUTODETECT_RIGHT_TEXT = [f"R{i + 1} short" for i in range(4)]


class TestAutoDetectReadingOrder:
    """align_text must place text in the right boxes regardless of which
    order the LLM emitted lines in. The DP cost itself is the signal —
    no per-model branching needed."""

    def test_column_major_emission_aligns_correctly(self):
        # OlmOCR-style: entire left column emitted first, then right column.
        lines = _AUTODETECT_LEFT_TEXT + _AUTODETECT_RIGHT_TEXT
        structured = [(b, "") for b in _AUTODETECT_BOXES_ROW_MAJOR]
        out = _aligner().align_text(structured, lines)
        actual = [t for _, t in out]
        for i in range(4):
            assert actual[i * 2] == _AUTODETECT_LEFT_TEXT[i]
            assert actual[i * 2 + 1] == _AUTODETECT_RIGHT_TEXT[i]

    def test_row_major_emission_aligns_correctly(self):
        # Row-major emission: L1 R1 L2 R2 L3 R3 L4 R4.
        lines = []
        for i in range(4):
            lines.append(_AUTODETECT_LEFT_TEXT[i])
            lines.append(_AUTODETECT_RIGHT_TEXT[i])
        structured = [(b, "") for b in _AUTODETECT_BOXES_ROW_MAJOR]
        out = _aligner().align_text(structured, lines)
        actual = [t for _, t in out]
        for i in range(4):
            assert actual[i * 2] == _AUTODETECT_LEFT_TEXT[i]
            assert actual[i * 2 + 1] == _AUTODETECT_RIGHT_TEXT[i]

    def test_single_column_unaffected(self):
        # Single-column page: row-major and column-major collapse to the
        # same order, so auto-detect is a no-op.
        boxes = [
            [0.10, 0.10, 0.90, 0.18],
            [0.10, 0.25, 0.90, 0.33],
            [0.10, 0.40, 0.90, 0.48],
            [0.10, 0.55, 0.90, 0.63],
        ]
        lines = ["first", "second", "third", "fourth"]
        structured = [(b, "") for b in boxes]
        out = _aligner().align_text(structured, lines)
        assert [t for _, t in out] == lines


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
