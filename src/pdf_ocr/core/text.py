"""
TextHandler - emit OCR results as a plain ``.txt`` dump.

The fastest, simplest output: no rasterization, no positioning, no
markup. One block of text per page in reading order, pages separated by
a blank line. Pairs with the pipeline's text-only mode (full-page OCR,
no layout binding), but works with any ``(bbox, text)`` page data.
"""

from __future__ import annotations


class TextHandler:
    """Output writer mirroring ``PDFHandler.embed_structured_text``.

    ``input_path`` / ``dpi`` are accepted for signature parity with the
    other writers but are unused — a plain dump has no title header and
    no rasterization.
    """

    def embed_structured_text(
        self,
        input_path: str,  # noqa: ARG002  (signature parity; no header in a plain dump)
        output_path: str,
        pages_data: dict,
        dpi: int = 200,  # noqa: ARG002  (signature parity with other writers)
    ) -> None:
        pages: list[str] = []
        for page_num in sorted(pages_data.keys()):
            blocks = [
                (text or "").strip()
                for _, text in (pages_data.get(page_num) or [])
            ]
            page_text = "\n".join(b for b in blocks if b)
            if page_text:
                pages.append(page_text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(pages))
            if pages:
                f.write("\n")
