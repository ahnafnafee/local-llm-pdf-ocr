"""Render an audited fixture correction against its original PDF pixels.

Example: uv run scripts/review_reference_corrections.py dense 1 --output review.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import fitz
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", choices=["dense", "notes"])
    parser.add_argument("index", type=int, help="Original layout_index in the audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "evals/reference-corrections.json").read_text(encoding="utf-8"))
    document = audit["documents"][args.document]
    correction = next((c for c in document["changes"] if c["layout_index"] == args.index), None)
    if correction is None:
        parser.error("This layout index has no audited correction")
    source = root / document["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != document["source_sha256"]:
        parser.error("Source PDF hash differs from the audited source")
    y0, x0, y1, x1 = correction["bbox_yxyx"]
    with fitz.open(source) as pdf:
        page = pdf[correction["source_page"] - 1]
        target = fitz.Rect(x0, y0, x1, y1)
        region = fitz.Rect(x0 - 4, y0 - 4, x1 + 4, y1 + 4) & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=region, alpha=False)
        crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(crop)
    draw.rectangle(((target.x0-region.x0)*4, (target.y0-region.y0)*4,
                    (target.x1-region.x0)*4, (target.y1-region.y0)*4), outline="red", width=2)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    lines = [f"{args.document}, page {correction['source_page']}, layout {args.index}"]
    for label in ("before", "after"):
        lines.extend(textwrap.wrap(f"{label}: {correction[label]}", width=90))
    width = max(1100, crop.width)
    result = Image.new("RGB", (width, crop.height + 32*len(lines) + 24), "white")
    result.paste(crop, (0, 0))
    ImageDraw.Draw(result).multiline_text((8, crop.height+10), "\n".join(lines), font=font,
                                        fill="black", spacing=8)
    result.save(args.output)


if __name__ == "__main__":
    main()
