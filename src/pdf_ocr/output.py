"""
Output-format dispatch.

Single source of truth for mapping an output path (or an explicit
format string) to the right writer + filename suffix + HTTP media type.
The CLI and the FastAPI server both consume this so adding a new
output format requires editing one place.

Three built-in formats:
    pdf  → searchable sandwich PDF (default)
    html → self-contained HTML with invisible-text overlay
    md   → Markdown text dump
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Same shape as `pdf_ocr.pipeline.OutputWriter` — kept here so callers
# can use the type hint without depending on the pipeline module.
OutputWriter = Callable[[str, str, dict, int], None]


_HTML_EXTS = frozenset({".html", ".htm"})
_MD_EXTS = frozenset({".md", ".markdown"})
_PDF_EXTS = frozenset({".pdf"})


_FORMAT_TO_SUFFIX = {
    "pdf": ".pdf",
    "html": ".html",
    "md": ".md",
}

_FORMAT_TO_MEDIA_TYPE = {
    "pdf": "application/pdf",
    "html": "text/html",
    "md": "text/markdown",
}

SUPPORTED_FORMATS = tuple(_FORMAT_TO_SUFFIX.keys())


def format_from_path(output_path: str) -> str:
    """Return the format name (`pdf` / `html` / `md`) matching the path's
    extension. Unknown extensions default to `pdf`."""
    ext = Path(output_path).suffix.lower()
    if ext in _HTML_EXTS:
        return "html"
    if ext in _MD_EXTS:
        return "md"
    return "pdf"


def suffix_for_format(fmt: str) -> str:
    """Return the canonical filename suffix for a format name.

    Raises:
        ValueError if `fmt` is not in :data:`SUPPORTED_FORMATS`.
    """
    if fmt not in _FORMAT_TO_SUFFIX:
        raise ValueError(
            f"unknown output format {fmt!r}; "
            f"expected one of {SUPPORTED_FORMATS}"
        )
    return _FORMAT_TO_SUFFIX[fmt]


def media_type_for(output_path: str) -> str:
    """Return the HTTP media type matching `output_path`'s extension."""
    return _FORMAT_TO_MEDIA_TYPE[format_from_path(output_path)]


def resolve_output_writer(
    output_path: str,
    *,
    html_mode: str | None = None,
    html_inline_images: bool = False,
    html_invert_dark: bool = False,
    html_hover_text: bool = False,
) -> OutputWriter:
    """Pick the writer matching `output_path`'s extension.

    Returns the writer's `embed_structured_text` bound method so the
    pipeline can call it as a plain `(input, output, pages, dpi)`
    callable.

    Args:
        output_path: drives format selection via extension.
        html_mode: optional sizing-strategy override for the HTML writer
            ("letter-spacing", "full-height", "scaled"). Ignored for
            non-HTML outputs.
        html_inline_images: when True, the HTML writer inlines page
            images as base64 ``data:`` URLs (single self-contained
            file). When False (default), page images are referenced as
            external files — the input file directly for single-frame
            browser-native images, or sidecar JPEGs written next to the
            output for PDFs and other inputs. Ignored for non-HTML
            outputs.
        html_invert_dark: when True, adds CSS that inverts page images
            in dark mode via ``filter: invert() hue-rotate(180deg)``.
            Useful for scanned white-background documents viewed at
            night. Ignored for non-HTML outputs.
        html_hover_text: when True, adds CSS that reveals a span's
            otherwise-invisible text on hover/focus (white on a dark
            backdrop), for inspecting which text is bound to which
            region. Ignored for non-HTML outputs.
    """
    # Late imports keep this module light — importing pdf_ocr.output
    # shouldn't pull in fitz or PIL unless the user actually needs a
    # writer.
    fmt = format_from_path(output_path)
    if fmt == "html":
        from pdf_ocr.core.html import HTMLHandler
        kwargs = {
            "inline_images": html_inline_images,
            "invert_dark": html_invert_dark,
            "hover_text": html_hover_text,
        }
        if html_mode is not None:
            kwargs["mode"] = html_mode
        return HTMLHandler(**kwargs).embed_structured_text
    if fmt == "md":
        from pdf_ocr.core.markdown import MarkdownHandler
        return MarkdownHandler().embed_structured_text
    from pdf_ocr.core.pdf import PDFHandler
    return PDFHandler().embed_structured_text
