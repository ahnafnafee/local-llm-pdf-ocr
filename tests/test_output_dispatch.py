"""
Unit tests for `pdf_ocr.output` — the format-dispatch helpers shared by
the CLI and the FastAPI server.
"""

from __future__ import annotations

import pytest

from pdf_ocr.core.html import HTMLHandler
from pdf_ocr.core.markdown import MarkdownHandler
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.output import (
    SUPPORTED_FORMATS,
    format_from_path,
    media_type_for,
    resolve_output_writer,
    suffix_for_format,
)


class TestFormatFromPath:
    @pytest.mark.parametrize("path,expected", [
        ("out.pdf", "pdf"),
        ("OUT.PDF", "pdf"),
        ("dir/sub/out.pdf", "pdf"),
        ("out.html", "html"),
        ("OUT.HTML", "html"),
        ("out.htm", "html"),
        ("out.md", "md"),
        ("out.markdown", "md"),
        ("out.MARKDOWN", "md"),
    ])
    def test_known_extensions(self, path, expected):
        assert format_from_path(path) == expected

    def test_unknown_extension_defaults_to_pdf(self):
        assert format_from_path("out.txt") == "pdf"
        assert format_from_path("noextension") == "pdf"


class TestSuffixForFormat:
    @pytest.mark.parametrize("fmt,suffix", [
        ("pdf", ".pdf"),
        ("html", ".html"),
        ("md", ".md"),
    ])
    def test_canonical_suffixes(self, fmt, suffix):
        assert suffix_for_format(fmt) == suffix

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown output format"):
            suffix_for_format("docx")


class TestMediaTypeFor:
    @pytest.mark.parametrize("path,mt", [
        ("out.pdf", "application/pdf"),
        ("out.html", "text/html"),
        ("out.htm", "text/html"),
        ("out.md", "text/markdown"),
        ("out.markdown", "text/markdown"),
    ])
    def test_known_paths(self, path, mt):
        assert media_type_for(path) == mt

    def test_unknown_path_defaults_to_pdf_media(self):
        assert media_type_for("out.docx") == "application/pdf"


class TestResolveOutputWriter:
    def test_pdf_returns_pdf_writer(self):
        # The returned callable must be PDFHandler's bound method —
        # verify by checking the underlying class.
        writer = resolve_output_writer("out.pdf")
        assert getattr(writer, "__self__", None).__class__ is PDFHandler

    def test_html_returns_html_writer(self):
        writer = resolve_output_writer("out.html")
        assert getattr(writer, "__self__", None).__class__ is HTMLHandler

    def test_htm_extension_returns_html_writer(self):
        writer = resolve_output_writer("out.htm")
        assert getattr(writer, "__self__", None).__class__ is HTMLHandler

    def test_md_returns_markdown_writer(self):
        writer = resolve_output_writer("out.md")
        assert getattr(writer, "__self__", None).__class__ is MarkdownHandler

    def test_markdown_extension_returns_markdown_writer(self):
        writer = resolve_output_writer("out.markdown")
        assert getattr(writer, "__self__", None).__class__ is MarkdownHandler

    def test_unknown_extension_defaults_to_pdf_writer(self):
        writer = resolve_output_writer("out.docx")
        assert getattr(writer, "__self__", None).__class__ is PDFHandler

    def test_writer_callable_signature_matches_pipeline_contract(self):
        """The pipeline calls writer(input, output, pages_data, dpi).

        Verify we can pass that exact shape to the dispatched writer
        without TypeError. Doesn't check correctness of the output —
        just the signature contract.
        """
        # Arrange: a Markdown writer (cheapest — no rasterization).
        import tempfile
        writer = resolve_output_writer("out.md")
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_path = f.name
        try:
            writer("fake.pdf", output_path, {0: []}, 200)
        finally:
            import os
            os.unlink(output_path)

    def test_html_writer_default_is_external_images(self):
        writer = resolve_output_writer("out.html")
        handler = writer.__self__
        assert handler.inline_images is False

    def test_html_writer_inline_opt_in_via_kwarg(self):
        writer = resolve_output_writer("out.html", html_inline_images=True)
        handler = writer.__self__
        assert handler.inline_images is True

    def test_html_writer_default_mode_is_scaled(self):
        writer = resolve_output_writer("out.html")
        handler = writer.__self__
        assert handler.mode == "scaled"

    def test_html_writer_mode_override(self):
        writer = resolve_output_writer("out.html", html_mode="letter-spacing")
        handler = writer.__self__
        assert handler.mode == "letter-spacing"


class TestSupportedFormats:
    def test_canonical_order(self):
        # The CLI uses this for the --format flag's choices list, so the
        # order matters for the help-text output.
        assert SUPPORTED_FORMATS == ("pdf", "html", "md")
