"""
Debug script to test Surya's DetectionPredictor ONLY (no recognition).
This approach gives us bounding boxes without the expensive recognition model.

The idea: Since LLM provides high-quality text, we only need boxes from Surya.
Alignment becomes position-based rather than anchor-based.
"""
import fitz
from PIL import Image, ImageDraw, ImageFont
import io
import os
import time

def get_detection_only_boxes(image_bytes):
    """Get bounding boxes using ONLY Surya's DetectionPredictor (no recognition)."""
    from PIL import Image
    from surya.detection import DetectionPredictor
    
    start = time.time()
    
    # Initialize ONLY detection predictor
    detection = DetectionPredictor()
    
    init_time = time.time() - start
    print(f"  [Detection-Only] Model init: {init_time:.2f}s")
    
    # Run detection
    start = time.time()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size
    
    predictions = detection([image])
    
    infer_time = time.time() - start
    print(f"  [Detection-Only] Inference: {infer_time:.2f}s")
    
    results = []
    if predictions and predictions[0].bboxes:
        for bbox in predictions[0].bboxes:
            # bbox is [x0, y0, x1, y1]
            x0, y0, x1, y1 = bbox.bbox
            
            # Normalize coords
            nx0, ny0 = x0/img_w, y0/img_h
            nx1, ny1 = x1/img_w, y1/img_h
            
            # Clamp to bounds
            nx0 = max(0.0, min(1.0, nx0))
            ny0 = max(0.0, min(1.0, ny0))
            nx1 = max(0.0, min(1.0, nx1))
            ny1 = max(0.0, min(1.0, ny1))
            
            results.append([nx0, ny0, nx1, ny1])
    
    # Sort top-to-bottom, then left-to-right
    results.sort(key=lambda r: (r[1], r[0]))
    
    return results, init_time, infer_time


def visualize_detection_boxes(img, boxes, color, label):
    """Draw detection-only boxes on image."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, rect in enumerate(boxes):
        nx0, ny0, nx1, ny1 = rect
        
        x0 = nx0 * width
        y0 = ny0 * height
        x1 = nx1 * width
        y1 = ny1 * height
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # Draw box number
        draw.text((x0 + 2, y0 + 2), str(i+1), fill=color, font=font)
    
    draw.text((10, 10), f"{label}: {len(boxes)} boxes", fill=color, font=font)
    return img


def test_detection(pdf_filename):
    """Test detection-only on a PDF file."""
    input_path = os.path.join("examples", pdf_filename)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Detection-Only Test: {pdf_filename}")
    print(f"{'='*60}")
    
    # Load PDF
    doc = fitz.open(input_path)
    page = doc[0]
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    
    # Detection only
    print("\nRunning Detection-Only...")
    det_boxes, det_init, det_infer = get_detection_only_boxes(img_bytes)
    
    print(f"\n  Total time: {det_init + det_infer:.2f}s")
    print(f"  Boxes detected: {len(det_boxes)}")
    
    # Visualize
    base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    det_img = visualize_detection_boxes(base_img, det_boxes, "blue", "DETECTION-ONLY")
    
    output_base = os.path.splitext(pdf_filename)[0]
    det_img.save(f"debug_det_{output_base}.png")
    print(f"\nSaved: debug_det_{output_base}.png")
    
    doc.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_detection(sys.argv[1])
    else:
        for pdf_file in ["digital.pdf", "hybrid.pdf", "handwritten.pdf"]:
            test_detection(pdf_file)
