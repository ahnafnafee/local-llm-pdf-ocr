"""Unit tests for OCRProcessor prompt/parsing concerns (no LLM calls)."""

from __future__ import annotations

from src.pdf_ocr.core.ocr import (
    CROP_PROMPT,
    OLMOCR_PAGE_PROMPT,
    _strip_runaway_repetition,
    _strip_yaml_front_matter,
)


class TestYAMLFrontMatter:
    def test_strips_canonical_olmocr_response(self):
        response = (
            "---\n"
            "primary_language: en\n"
            "is_rotation_valid: true\n"
            "rotation_correction: 0\n"
            "is_table: false\n"
            "is_diagram: false\n"
            "---\n"
            "# Document Title\n\nBody paragraph text.\n"
        )
        body = _strip_yaml_front_matter(response)
        assert "primary_language" not in body
        assert "Body paragraph text" in body
        assert body.startswith("# Document Title")

    def test_passthrough_when_no_front_matter(self):
        response = "Plain text response with no YAML.\nSecond line."
        assert _strip_yaml_front_matter(response) == response

    def test_malformed_front_matter_returned_unchanged(self):
        # Opening fence but no closing fence — don't guess; preserve text.
        response = "---\nprimary_language: en\nbody text with no closing fence"
        assert _strip_yaml_front_matter(response) == response

    def test_leading_whitespace_handled(self):
        response = "  \n---\nkey: val\n---\nbody"
        out = _strip_yaml_front_matter(response)
        assert out == "body"


class TestStripRunawayRepetition:
    def test_passes_short_unique_lines_through(self):
        lines = ["a", "b", "c", "d"]
        assert _strip_runaway_repetition(lines) == lines

    def test_admits_legitimate_table_repetition(self):
        # HTML table from OlmOCR's "tables to HTML" instruction — many <tr>
        # tags are expected and shouldn't be clipped.
        lines = (
            ["<table>"]
            + ["<tr>", "<td>x</td>", "</tr>"] * 10
            + ["</table>"]
        )
        out = _strip_runaway_repetition(lines, max_repeat=20)
        assert out == lines, "10x table repetition must survive"

    def test_clips_runaway_repetition(self):
        # Model stuck in a loop emitting the same line 100 times — should
        # leave only the first 20 occurrences.
        lines = ["unique header"] + ["LOOP"] * 100 + ["unique footer"]
        out = _strip_runaway_repetition(lines, max_repeat=20)
        assert out.count("LOOP") == 20
        assert out[0] == "unique header"
        # Lines after the loop are NOT preserved (they appear after the cap)
        # because we walk the list in order — but the cap is for the LOOP
        # string only, so non-LOOP lines that come after still pass through.
        assert "unique footer" in out

    def test_clips_multiple_runaway_lines_independently(self):
        lines = ["A"] * 50 + ["B"] * 50
        out = _strip_runaway_repetition(lines, max_repeat=10)
        assert out.count("A") == 10
        assert out.count("B") == 10

    def test_empty_input(self):
        assert _strip_runaway_repetition([]) == []


class TestPromptConstants:
    def test_olmocr_prompt_is_canonical(self):
        # Guard against accidental prompt drift — this string was lifted
        # verbatim from allenai/olmocr. If you change it, expect worse OCR.
        assert "Attached is one page of a document" in OLMOCR_PAGE_PROMPT
        assert "Convert equations to LateX and tables to HTML" in OLMOCR_PAGE_PROMPT
        assert "front matter section" in OLMOCR_PAGE_PROMPT

    def test_crop_prompt_is_minimal(self):
        # For crops we want plain text — no metadata/markdown ceremony.
        assert "no markdown" in CROP_PROMPT.lower()
        assert "plain text" in CROP_PROMPT.lower()
