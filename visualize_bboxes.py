
import fitz
from PIL import Image, ImageDraw
import io
import os
from hybrid_aligner import HybridAligner

def visualize_boxes(pdf_filename):
    input_path = os.path.join("examples", pdf_filename)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Processing {pdf_filename}...")
    
    # Initialize aligner
    aligner = HybridAligner()
    
    # Load PDF
    doc = fitz.open(input_path)
    page = doc[0] # Just checking first page
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    
    # Create PIL Image for drawing
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Get boxes
    structured_data = aligner.get_structured_text(img_bytes)
    print(f"  Found {len(structured_data)} text blocks.")
    
    for (rect, text) in structured_data:
        # rect is normalized [nx0, ny0, nx1, ny1]
        nx0, ny0, nx1, ny1 = rect
        
        # Scale to image dimensions
        x0 = nx0 * width
        y0 = ny0 * height
        x1 = nx1 * width
        y1 = ny1 * height
        
        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        
    output_filename = f"bbox_{os.path.splitext(pdf_filename)[0]}.png"
    img.save(output_filename)
    print(f"  Saved visualization to {output_filename}")

if __name__ == "__main__":
    visualize_boxes("digital.pdf")
    visualize_boxes("hybrid.pdf")
    visualize_boxes("handwritten.pdf")
