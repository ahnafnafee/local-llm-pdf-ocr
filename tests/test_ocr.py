"""Unit tests for OCRProcessor prompt/parsing concerns (no LLM calls)."""

from __future__ import annotations

from src.pdf_ocr.core.ocr import (
    CROP_PROMPT,
    OLMOCR_PAGE_PROMPT,
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
