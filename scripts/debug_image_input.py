#!/usr/bin/env python3
"""Diagnostic: render Surya boxes + final output text positions for an image input."""

import asyncio
import os
import sys

import fitz
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_ocr import HybridAligner, OCRPipeline, OCRProcessor, PDFHandler


async def main(image_path: str, output_pdf: str) -> None:
    pdf_handler = PDFHandler()
    aligner = HybridAligner()
    ocr_processor = OCRProcessor()

    pipeline = OCRPipeline(
        aligner=aligner,
        ocr_processor=ocr_processor,
        pdf_handler=pdf_handler,
    )

    # Disable refine so we see raw DP output
    pages_text = await pipeline.run(image_path, output_pdf, dpi=200, refine=False)
    print("\n=== LLM lines per page ===")
    for p, lines in pages_text.items():
        print(f"page {p}: {len(lines)} lines")
        # Detect repetition: count consecutive duplicates
        from collections import Counter
        counts = Counter(lines)
        repeats = [(k, v) for k, v in counts.items() if v > 3]
        if repeats:
            print(f"  REPETITION DETECTED: {repeats[:5]}")
        for i, l in enumerate(lines[:30]):
            print(f"  [{i}] {l!r}")
        if len(lines) > 30:
            print(f"  ... [{len(lines) - 30} more lines]")
            for i, l in enumerate(lines[-5:]):
                print(f"  [{len(lines) - 5 + i}] {l!r}")

    # Visualize Surya boxes on the source image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    boxes = aligner.get_detected_boxes_batch([buf.getvalue()])[0]
    print(f"\n=== Surya boxes: {len(boxes)} ===")
    for i, b in enumerate(boxes):
        print(f"  [{i}] {b}")

    # Draw boxes on the image
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = draw_img.size
    for b in boxes:
        x0, y0, x1, y1 = b
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline="red", width=3)
    boxes_png = os.path.splitext(output_pdf)[0] + "_boxes.png"
    draw_img.save(boxes_png)
    print(f"\nSaved Surya bbox visualization -> {boxes_png}")

    # Inspect text positions in the output PDF — word level (not block grouped)
    out = fitz.open(output_pdf)
    print(f"\n=== Output PDF words ===")
    for pn, page in enumerate(out):
        print(f"page {pn} size={page.rect}")
        words = page.get_text("words")
        for x0, y0, x1, y1, w, *_ in words:
            print(f"  bbox=({x0:6.1f},{y0:6.1f},{x1:6.1f},{y1:6.1f}) word={w!r}")
    out.close()

    # Run align_text directly to inspect post-DP mapping
    structured = [(b, "") for b in boxes]
    aligned = aligner.align_text(structured, pages_text[0])
    print(f"\n=== Aligned (box, text) pairs (raw DP, no refine) ===")
    for i, (bbox, text) in enumerate(aligned):
        print(f"  [{i}] {bbox} -> {text!r}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
