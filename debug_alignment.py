
import fitz
import base64
import argparse
import sys
import io
import os
from PIL import Image, ImageDraw, ImageFont
from pdf_utils import PDFHandler
from ocr import OCRProcessor
from hybrid_aligner import HybridAligner

def debug_alignment(pdf_path):
    print(f"Debug Alignment for: {pdf_path}")
    
    # 1. Init
    pdf_handler = PDFHandler()
    ocr_processor = OCRProcessor()
    hybrid_aligner = HybridAligner()
    
    # 2. Get images
    print("Converting PDF to images...")
    images_dict = pdf_handler.convert_to_images(pdf_path)
    
    # Process only first page for debug
    page_num = 0
    if page_num not in images_dict:
        print("No page 0 found.")
        return

    print("Processing Page 1...")
    image_base64 = images_dict[page_num]
    image_bytes = base64.b64decode(image_base64)
    
    # Restore image object for drawing
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    print(f"Image Size: {width}x{height}")
    
    # 3. Surya Layout
    print("Running Surya Layout...")
    structured_data = hybrid_aligner.get_structured_text(image_bytes)
    print(f"Surya found {len(structured_data)} boxes.")
    
    # 4. LLM OCR
    print("Running LLM OCR...")
    llm_lines = ocr_processor.perform_ocr(image_base64)
    print(f"LLM found {len(llm_lines)} lines.")
    
    # 5. Align
    print("Aligning...")
    aligned_data = hybrid_aligner.align_text(structured_data, llm_lines)
    print(f"Aligned into {len(aligned_data)} blocks.")
    
    # 6. Visualize
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, else default
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for i, (rect, text) in enumerate(aligned_data):
        # rect is [nx0, ny0, nx1, ny1]
        nx0, ny0, nx1, ny1 = rect
        
        x0 = nx0 * width
        y0 = ny0 * height
        x1 = nx1 * width
        y1 = ny1 * height
        
        # Color code: Green = Anchor, Blue = Gap-Filled/Other
        # We don't have type info here easily unless we modify aligner return, 
        # but usually tight boxes are anchors.
        
        color = "green"
        # draw box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # draw text index/snippet
        if text:
            label = f"{i}: {text}"
            draw.text((x0, y0 - 15), label, fill="red", font=font)
            
    output_filename = f"debug_align_{os.path.basename(pdf_path)}.png"
    img.save(output_filename)
    print(f"Saved debug image to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_alignment(sys.argv[1])
    else:
        print("Please provide a PDF path.")
        # Default debug
        debug_alignment("examples/hybrid.pdf")
