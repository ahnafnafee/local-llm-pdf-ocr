"""
MarkdownHandler - emit OCR results as a Markdown document.

Markdown is the simplest of the three output formats: no rasterization,
no positioning, just text in reading order. We trust the aligner /
grounded-VLM to have already ordered boxes correctly, and emit one
paragraph per non-empty box separated by blank lines.

Why no paragraph-break heuristic?
---------------------------------
A naive vertical-gap heuristic ("gap > 1.5× line height = new
paragraph") misfires on multi-column layouts because the y-coordinate
jumps backward at each column break, and the gap math becomes
nonsense. Single-column layouts work fine, but the broken cases are
worse than just emitting one block per box. So we keep it boring and
predictable: every box is its own block. Users who want flowed text
can post-process trivially.
"""

from __future__ import annotations

from pathlib import Path


class MarkdownHandler:
    """Output writer mirroring ``PDFHandler.embed_structured_text``.

    The ``dpi`` parameter is accepted for signature compatibility with
    the other writers but is unused — Markdown has no rasterization.
    """

    def embed_structured_text(
        self,
        input_path: str,
        output_path: str,
        pages_data: dict,
        dpi: int = 200,  # noqa: ARG002  (signature parity with other writers)
    ) -> None:
        title = Path(input_path).name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR output: {title}\n\n")
            for page_num in sorted(pages_data.keys()):
                f.write(f"## Page {page_num + 1}\n\n")
                items = pages_data.get(page_num) or []
                for _, text in items:
                    text = (text or "").strip()
                    if not text:
                        continue
                    f.write(text)
                    f.write("\n\n")
