#!/usr/bin/env python3
"""
Generate side-by-side comparison of raw Surya boxes vs aligned output.
"""

import argparse
import asyncio
import logging
import fitz
from PIL import Image, ImageDraw, ImageFont
import io
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_ocr.core.ocr import OCRProcessor
from pdf_ocr.core.aligner import HybridAligner

# Configure logging
logging.basicConfig(level=logging.INFO)


async def generate_comparison(pdf_path, output_path):
    print(f"Processing {pdf_path}...")
    
    # 1. Convert first page to image
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    img_data = pix.tobytes("png")
    
    # Create PIL Image
    original_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    width, height = original_img.size
    
    # Compress to JPEG for LLM (Robustness + Size)
    # Resize if too large to prevent 400 Bad Request
    max_dim = 1500
    if width > max_dim or height > max_dim:
        original_img.thumbnail((max_dim, max_dim))
        
    buffer = io.BytesIO()
    original_img.save(buffer, format="JPEG", quality=50)
    jpeg_bytes = buffer.getvalue()
    print(f"Compressed Image Size: {len(jpeg_bytes)/1024:.2f} KB")
    
    # 2. Run Surya (Layout) via batch API with a single-element list
    aligner = HybridAligner()
    print("Running Surya Detection...")
    batch = await asyncio.to_thread(aligner.get_detected_boxes_batch, [img_data])
    structured_data = [(box, "") for box in batch[0]]
    
    # 3. Run LLM OCR
    processor = OCRProcessor()
    
    print("Running LLM OCR...")
    try:
        import base64
        # Processor expects base64 string
        b64_img = base64.b64encode(jpeg_bytes).decode('utf-8')
        
        llm_text_lines = await processor.perform_ocr(b64_img)
        print(f"LLM Response: {len(llm_text_lines)} lines found.")
    except Exception as e:
        print(f"LLM Failed: {e}. Using dummy text for visualization if needed, but alignment will fail.")
        llm_text_lines = []

    # 4. Run Hybrid Alignment
    print("Running Hybrid Alignment...")
    final_output = aligner.align_text(structured_data, llm_text_lines)
    
    # --- DRAWING ---
    
    # Image 1: Raw Surya Boxes
    img_raw = original_img.copy()
    draw_raw = ImageDraw.Draw(img_raw)
    
    # Draw Raw Boxes (Red)
    for (rect, text) in structured_data:
        # rect is normalized [x0, y0, x1, y1]
        x0 = rect[0] * width
        y0 = rect[1] * height
        x1 = rect[2] * width
        y1 = rect[3] * height
        
        draw_raw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    
    # Image 2: Hybrid Aligned (Green)
    img_hybrid = original_img.copy()
    draw_hybrid = ImageDraw.Draw(img_hybrid)
    
    # Draw Aligned Boxes (Green)
    for (rect, text) in final_output:
        x0 = rect[0] * width
        y0 = rect[1] * height
        x1 = rect[2] * width
        y1 = rect[3] * height
        
        draw_hybrid.rectangle([x0, y0, x1, y1], outline="#00ff00", width=3)
    
    # Create Side-by-Side Comparison
    total_width = width * 2 + 50 # 50px gap
    comparison_img = Image.new('RGB', (total_width, height), color='white')
    
    comparison_img.paste(img_raw, (0, 0))
    comparison_img.paste(img_hybrid, (width + 50, 0))
    
    # Add Labels
    draw = ImageDraw.Draw(comparison_img)
    
    draw.text((10, 10), "Before: Raw Surya Boxes", fill="red")
    draw.text((width + 60, 10), "After: Aligned (Gap Filling)", fill="#00aa00")
    
    comparison_img.save(output_path)
    print(f"Comparison saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_comparison.py <input_pdf> <output_image_path>")
    else:
        asyncio.run(generate_comparison(sys.argv[1], sys.argv[2]))
