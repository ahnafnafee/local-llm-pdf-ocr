
import fitz
import os

def inspect_pdf(filename):
    path = os.path.join("examples", filename)
    print(f"\nInspecting {filename}...")
    doc = fitz.open(path)
    page = doc[0]
    print(f"  Rotation: {page.rotation}")
    print(f"  MediaBox: {page.mediabox}")
    print(f"  CropBox:  {page.cropbox}")
    print(f"  Rect:     {page.rect}")
    
    # Check image size
    pix = page.get_pixmap()
    print(f"  Pixmap:   {pix.width} x {pix.height}")

if __name__ == "__main__":
    inspect_pdf("hybrid.pdf")
    inspect_pdf("digital.pdf")
    inspect_pdf("handwritten.pdf")
