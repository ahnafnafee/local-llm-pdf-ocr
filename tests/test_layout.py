"""
Unit tests for `pdf_ocr.core._layout` helpers.

These helpers are shared by every output writer (PDF, HTML, …), so the
tests are thin and behavior-focused: full-page-fallback detection and
multi-line bbox vertical splitting.
"""

from __future__ import annotations

import pytest

from pdf_ocr.core._layout import (
    is_full_page_fallback,
    split_multi_line_bbox,
)


class TestIsFullPageFallback:
    def test_canonical_fallback_with_newline_is_true(self):
        # The exact aligner emission: [0,0,1,1] + multi-line text.
        assert is_full_page_fallback([0.0, 0.0, 1.0, 1.0], "line A\nline B")

    def test_almost_full_page_within_tolerance_is_true(self):
        # 0.001 / 0.999 are inside the tolerance window.
        assert is_full_page_fallback([0.0005, 0.0005, 0.9995, 0.9995], "a\nb")

    def test_full_page_without_newline_is_false(self):
        # Single-line text in a [0,0,1,1] bbox is unusual but not the
        # fallback case the helper guards against.
        assert not is_full_page_fallback([0.0, 0.0, 1.0, 1.0], "single line")

    def test_real_bbox_with_newline_is_false(self):
        # Genuine multi-line content in a real bbox must NOT be treated
        # as the full-page fallback (would shift it off-page).
        assert not is_full_page_fallback([0.1, 0.1, 0.9, 0.4], "line A\nline B")

    def test_outside_tolerance_is_false(self):
        # 0.01 / 0.99 are outside the 0.001 tolerance window — real bbox.
        assert not is_full_page_fallback([0.01, 0.01, 0.99, 0.99], "a\nb")

    def test_empty_text_is_false(self):
        assert not is_full_page_fallback([0.0, 0.0, 1.0, 1.0], "")


class TestSplitMultiLineBbox:
    def test_single_line_returns_one_entry(self):
        out = split_multi_line_bbox([0.1, 0.2, 0.9, 0.3], "hello world")
        assert len(out) == 1
        rect, text = out[0]
        assert rect == [0.1, 0.2, 0.9, 0.3]
        assert text == "hello world"

    def test_returns_a_copy_not_an_alias_of_the_input_list(self):
        # Mutating the returned rect must not affect the caller's input.
        original = [0.1, 0.2, 0.9, 0.3]
        out = split_multi_line_bbox(original, "x")
        out[0][0][0] = 0.99
        assert original == [0.1, 0.2, 0.9, 0.3]

    def test_empty_text_returns_empty_list(self):
        assert split_multi_line_bbox([0.1, 0.2, 0.9, 0.3], "") == []

    def test_whitespace_only_returns_empty(self):
        assert split_multi_line_bbox([0.1, 0.2, 0.9, 0.3], "   \n\t \n   ") == []

    def test_two_line_split_assigns_proportional_slices(self):
        out = split_multi_line_bbox([0.1, 0.2, 0.9, 0.4], "first\nsecond")
        assert len(out) == 2
        rect_a, text_a = out[0]
        rect_b, text_b = out[1]
        assert text_a == "first"
        assert text_b == "second"
        # Top half of [0.2, 0.4] is [0.2, 0.3]; bottom half is [0.3, 0.4].
        assert rect_a == pytest.approx([0.1, 0.2, 0.9, 0.3])
        assert rect_b == pytest.approx([0.1, 0.3, 0.9, 0.4])

    def test_three_line_split_assigns_proportional_slices(self):
        out = split_multi_line_bbox([0.0, 0.0, 1.0, 0.6], "a\nb\nc")
        assert len(out) == 3
        for i, (rect, text) in enumerate(out):
            assert text == "abc"[i]
            assert rect[1] == pytest.approx(i * 0.2)
            assert rect[3] == pytest.approx((i + 1) * 0.2)
            assert rect[0] == 0.0
            assert rect[2] == 1.0

    def test_empty_lines_dropped_during_split(self):
        # Trailing/embedded blank lines are stripped before counting.
        out = split_multi_line_bbox([0.0, 0.0, 1.0, 0.4], "a\n\nb\n   \n")
        assert len(out) == 2
        assert [text for _, text in out] == ["a", "b"]
        # After dropping empty lines we have 2 lines, so slice height is 0.2.
        assert out[0][0] == pytest.approx([0.0, 0.0, 1.0, 0.2])
        assert out[1][0] == pytest.approx([0.0, 0.2, 1.0, 0.4])

    def test_only_one_non_empty_line_after_strip_returns_one_entry(self):
        # "\n" present but only one non-empty line — falls through to
        # single-line case (caller treats this as no-split).
        out = split_multi_line_bbox([0.1, 0.2, 0.9, 0.3], "hello\n")
        assert len(out) == 1
        assert out[0][1] == "hello"
