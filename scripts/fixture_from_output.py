#!/usr/bin/env python3
"""
Build a ground-truth JSON fixture by extracting text + bboxes from an
already-produced sandwich PDF (one of examples/output_*.pdf).

Useful for examples that are too dense for the grounded VLM to handle
in one call (notes.pdf, dense.pdf) — the hybrid pipeline already
produced per-box (bbox, text) pairs, so we can read them straight back
out of the embedded text layer.

Usage:
    uv run scripts/fixture_from_output.py examples/output_notes.pdf \\
        tests/fixtures/ground_truth_notes.json
    uv run scripts/fixture_from_output.py examples/output_dense.pdf \\
        tests/fixtures/ground_truth_dense.json --source-name dense.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_pdf", help="Path to an existing output_*.pdf")
    parser.add_argument("fixture", help="Path to write the new fixture JSON")
    parser.add_argument(
        "--source-name", default=None,
        help="file_name field for the fixture (defaults to the output PDF stem "
             "with the 'output_' prefix stripped — e.g. output_notes.pdf -> notes.pdf)",
    )
    args = parser.parse_args()

    out_pdf = Path(args.output_pdf)
    fixture_path = Path(args.fixture)
    source_name = args.source_name or out_pdf.name.replace("output_", "")

    doc = fitz.open(str(out_pdf))
    page_sizes: list[tuple[int, int]] = []
    layout: list[dict] = []
    block_id = 0

    for page_idx, page in enumerate(doc):
        pw, ph = int(round(page.rect.width)), int(round(page.rect.height))
        page_sizes.append((pw, ph))
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    layout.append({
                        "block_content": text,
                        "bbox": [
                            int(round(y0)),
                            int(round(x0)),
                            int(round(y1)),
                            int(round(x1)),
                        ],
                        "block_id": block_id,
                        "page_index": page_idx,
                        "block_label": "text",
                        "score": 1.0,
                    })
                    block_id += 1
    doc.close()

    if not layout:
        print(f"ERROR: extracted 0 blocks from {out_pdf}; refusing to write empty fixture")
        sys.exit(1)

    fixture = {
        "data": {
            "file_name": source_name,
            "file_type": "pdf" if source_name.lower().endswith(".pdf") else "image",
            "layout": layout,
            "data_info": {
                "pages": [{"width": w, "height": h} for (w, h) in page_sizes],
                "num_pages": len(page_sizes),
            },
        }
    }

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)

    print(
        f"wrote {len(layout)} blocks across {len(page_sizes)} pages "
        f"(dims={'x'.join(f'{w}x{h}' for w, h in page_sizes[:3])}{'...' if len(page_sizes)>3 else ''})"
    )
    print(f"  -> {fixture_path}")


if __name__ == "__main__":
    main()
