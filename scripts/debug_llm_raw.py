#!/usr/bin/env python3
"""Dump the raw LLM lines per page for a PDF/image input — no DP, no embed."""

import asyncio
import os
import sys
from collections import Counter

# Force UTF-8 stdout on Windows so unicode in OCR'd text doesn't blow up.
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_ocr import OCRProcessor, PDFHandler


async def main(input_path: str) -> None:
    handler = PDFHandler()
    ocr = OCRProcessor()
    images = handler.convert_to_images(input_path)

    for page_num in sorted(images):
        print(f"\n=== page {page_num} ===")
        lines = await ocr.perform_ocr(images[page_num])
        print(f"  {len(lines)} lines")
        counts = Counter(lines)
        repeats = sorted(
            ((k, v) for k, v in counts.items() if v > 2),
            key=lambda kv: -kv[1],
        )
        if repeats:
            print(f"  REPETITION: {repeats[:5]}")
        for i, line in enumerate(lines[:20]):
            print(f"  [{i}] {line!r}")
        if len(lines) > 20:
            print(f"  ... [{len(lines) - 20} more]")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
