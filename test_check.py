
import fitz
from PIL import Image
import sys
from hybrid_aligner import HybridAligner
import os

def check_file(filename):
    path = os.path.join("examples", filename)
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return

    print(f"\nEvaluating {filename}...")
    try:
        aligner = HybridAligner()
        doc = fitz.open(path)
        page = doc[0]
        pix = page.get_pixmap()
        image_bytes = pix.tobytes("png") 
        
        structured_data = aligner.get_structured_text(image_bytes)
        print(f"  Items found: {len(structured_data)}")
        if structured_data:
            print(f"  First item: {structured_data[0]}")
            # Check range
            r = structured_data[0][0]
            if all(0.0 <= x <= 1.0 for x in r):
                print("  Status: OK (Normalized)")
            else:
                print(f"  Status: FAIL (Not Normalized: {r})")
            
            # Check sorting
            y_iter = [item[0][1] for item in structured_data] # y0
            is_sorted = all(y_iter[i] <= y_iter[i+1] for i in range(len(y_iter)-1))
            print(f"  Sorted by Y: {is_sorted}")
            if not is_sorted:
                print(f"  Y-coords first 10: {y_iter[:10]}")
        else:
             print("  Status: WARNING (No text found)")
             
    except Exception as e:
        print(f"  Status: ERROR ({e})")

if __name__ == "__main__":
    check_file("digital.pdf")
    check_file("hybrid.pdf")
    check_file("handwritten.pdf")
