"""
Tests for `pdf_ocr.cli` argument parsing and output-path resolution.

Heavy LLM / pipeline behavior is tested elsewhere; this file is about
the CLI surface: the `--format` flag, `resolve_output_path` decisions,
and how explicit output paths interact with the format flag.
"""

from __future__ import annotations

import sys

import pytest

from pdf_ocr.cli import build_parser, resolve_output_path


class TestResolveOutputPath:
    def test_no_output_no_format_defaults_to_pdf(self):
        out = resolve_output_path("/tmp/in.pdf", None)
        assert out.endswith("in_ocr.pdf")

    def test_no_output_with_format_uses_format_suffix(self):
        assert resolve_output_path("/tmp/in.pdf", None, "html").endswith("in_ocr.html")
        assert resolve_output_path("/tmp/in.pdf", None, "md").endswith("in_ocr.md")
        assert resolve_output_path("/tmp/in.pdf", None, "pdf").endswith("in_ocr.pdf")

    def test_explicit_output_wins_over_format(self):
        # User provides an explicit `.md` output but also `--format html`.
        # Explicit output wins so the actual writer ends up Markdown.
        out = resolve_output_path("/tmp/in.pdf", "out.md", "html")
        assert out == "out.md"

    def test_explicit_output_no_format(self):
        out = resolve_output_path("/tmp/in.pdf", "out.html", None)
        assert out == "out.html"

    def test_image_input_with_format(self):
        out = resolve_output_path("/tmp/scan.png", None, "html")
        assert out.endswith("scan_ocr.html")


class TestParserFormatFlag:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_format_flag_accepts_pdf_html_md(self):
        for fmt in ("pdf", "html", "md"):
            ns = self._parse("--format", fmt)
            assert ns.format == fmt

    def test_format_flag_default_is_none(self):
        # When --format isn't given, the namespace value is None — the
        # caller (resolve_output_path) treats that as "default to pdf".
        ns = self._parse()
        assert ns.format is None

    def test_format_flag_rejects_unknown_value(self, capsys):
        with pytest.raises(SystemExit):
            self._parse("--format", "docx")
        # argparse prints to stderr; either captured stream is fine.
        err = capsys.readouterr().err
        assert "invalid choice" in err

    def test_explicit_output_path_still_parsed(self):
        ns = self._parse("custom.html")
        assert ns.input_pdf == "input.pdf"
        assert ns.output_pdf == "custom.html"
        assert ns.format is None


class TestHtmlImageFlags:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_html_inline_images_defaults_to_false(self):
        ns = self._parse()
        assert ns.html_inline_images is False

    def test_html_inline_images_flag_sets_true(self):
        ns = self._parse("--html-inline-images")
        assert ns.html_inline_images is True

    def test_html_mode_default_is_scaled(self):
        # milahu requested this default — letter-spacing was unreadable
        # when negative spacing made characters overlap.
        ns = self._parse()
        assert ns.html_mode == "scaled"

    def test_html_mode_letter_spacing_still_selectable(self):
        ns = self._parse("--html-mode", "letter-spacing")
        assert ns.html_mode == "letter-spacing"


class TestQuadFidelityFlags:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_min_box_confidence_default_is_none(self):
        # Filtering detector boxes changes recall — it must be opt-in.
        assert self._parse().min_box_confidence is None

    def test_min_box_confidence_parses_float(self):
        ns = self._parse("--min-box-confidence", "0.25")
        assert ns.min_box_confidence == 0.25

    def test_html_hover_text_defaults_off(self):
        # The text layer stays invisible unless explicitly opted in.
        assert self._parse().html_hover_text is False

    def test_html_hover_text_opt_in(self):
        assert self._parse("--html-hover-text").html_hover_text is True
