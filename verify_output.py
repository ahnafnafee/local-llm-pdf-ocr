import fitz
import sys

def verify(pdf_path):
    print(f"Verifying '{pdf_path}'...")
    try:
        doc = fitz.open(pdf_path)
        text_found = False
        full_text = ""
        positions = []
        
        for i, page in enumerate(doc):
            # get_text("words") returns (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words = page.get_text("words")
            if words:
                text_found = True
                for w in words:
                    full_text += w[4] + " "
                    positions.append((w[0], w[1]))
            
            print(f"--- Page {i+1} Stats ---")
            print(f"Word count: {len(words)}")
            if words:
                 y_vals = sorted([w[1] for w in words])
                 print(f"Y-coordinate samples: {y_vals[::max(1, len(y_vals)//5)]}")
            print("-----------------------------")


        doc.close()
        
        if not text_found:
            print("FAILURE: No text found in PDF.")
            sys.exit(1)
        
        # Check if positions are distributed (not all near 0,0)
        # We expect y to vary.
        y_coords = [p[1] for p in positions]
        if not y_coords:
             print("FAILURE: No words.")
             sys.exit(1)
             
        min_y, max_y = min(y_coords), max(y_coords)
        print(f"Y-coordinate range: {min_y} - {max_y}")
        
        if max_y - min_y < 10:
            print("WARNING: Text seems clustered vertically (bad distribution).")
            # This might happen if 'y' prompt failed and returned 0.
            # But let's see.
        
        keywords = ["Algorithms", "computational", "finite", "mapped"]
        found_keywords = [k for k in keywords if k.lower() in full_text.lower()]
        print(f"Keywords found: {found_keywords}")
        
        if len(found_keywords) > 0:
            print("SUCCESS: Text content verified.")
        else:
             print("FAILURE: Text content missing keywords.")
             sys.exit(1)

    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify("output_ocr.pdf")

