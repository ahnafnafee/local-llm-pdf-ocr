import easyocr
import io
import sys
import logging
import string
from rapidfuzz import fuzz, process

from PIL import Image

class HybridAligner:
    def __init__(self):
        # Initialize EasyOCR reader (loads model once)
        # gpu=False for compatibility, True if available. Let's auto-detect or default to False to be safe first.
        self.reader = easyocr.Reader(['en'], gpu=False) 

    def get_structured_text(self, image_bytes):
        """
        Run EasyOCR on image -> list of (box, text, conf)
        Returns: list of (rect, text) tuples where rect is [nx0, ny0, nx1, ny1] (normalized 0..1)
        """
        # We need image dimensions to normalize coordinates
        with Image.open(io.BytesIO(image_bytes)) as img:
             img_w, img_h = img.size
        
        # EasyOCR expects bytes or path
        results = self.reader.readtext(image_bytes) 
        # results format: [ ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text", 0.99), ... ]

        structured_data = []
        
        for (bbox, text, conf) in results:
            # bbox is list of 4 points. fitz wants clean Rect [x0, y0, x1, y1]
            # EasyOCR returns detection poly, usually 4 points.
            # We take min/max to get rect.
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            
            # Normalize to 0..1
            nx0, ny0 = x0 / img_w, y0 / img_h
            nx1, ny1 = x1 / img_w, y1 / img_h
            
            structured_data.append(([nx0, ny0, nx1, ny1], text))
            
        return structured_data

    def _clean_text(self, text):
        """Standardizes text for matching: lower, strip punctuation."""
        text = text.lower().strip()
        return text.translate(str.maketrans('', '', string.punctuation))

    def align_text(self, structured_data, llm_text):
        """
        Aligns high-quality LLM text with EasyOCR bounding boxes using Anchor-Based Gap Filling.
        """
        if not llm_text:
            return structured_data
            
        if isinstance(llm_text, list):
            full_llm_text = " ".join(llm_text)
        else:
            full_llm_text = llm_text
            
        llm_tokens = full_llm_text.split()
        if not llm_tokens:
            return structured_data

        matches = [] 
        llm_search_idx = 0
        llm_search_idx = 0
        window_size = 100 # Increased to handle dense sections like "Notes"
        
        # 1. Identify Anchors with Normalized Matching
        for box_idx, (rect, box_text) in enumerate(structured_data):
            if not box_text.strip():
                continue
                
            clean_box = self._clean_text(box_text)
            if not clean_box: continue

            end_idx = min(len(llm_tokens), llm_search_idx + window_size)
            candidates = llm_tokens[llm_search_idx:end_idx]
            
            # Create cleaned candidates map to preserve original indices
            clean_candidates = [self._clean_text(c) for c in candidates]
            
            match = process.extractOne(clean_box, clean_candidates, scorer=fuzz.ratio)
            
            if match:
                best_clean_token, score, candidate_offset = match
                
                # Check for "Exact-ish" match on content
                if score > 80:
                    real_llm_idx = llm_search_idx + candidate_offset
                    # Retrieve the ORIGINAL token (with punctuation etc)
                    best_token = candidates[candidate_offset]
                    
                    matches.append({
                        "box_idx": box_idx,
                        "llm_idx": real_llm_idx,
                        "text": best_token,
                        "rect": rect,
                        "type": "anchor"
                    })
                    llm_search_idx = real_llm_idx + 1

        # 2. Reconstruct Page with Line-Aware Gap Filling
        final_output = []
        
        prev_box_idx = -1
        prev_llm_idx = -1
        
        matches.append({
            "box_idx": len(structured_data),
            "llm_idx": len(llm_tokens),
            "text": None, 
            "rect": None
        })
        
        for anchor in matches:
            curr_box_idx = anchor["box_idx"]
            curr_llm_idx = anchor["llm_idx"]
            
            gap_tokens = llm_tokens[prev_llm_idx+1 : curr_llm_idx]
            gap_boxes = structured_data[prev_box_idx+1 : curr_box_idx]
            
            if gap_tokens:
                 if gap_boxes:
                     # Strategy: Group gap boxes by Line (Y-coordinate)
                     # Valid lines usually have overlapping Y-ranges.
                     rows = []
                     if gap_boxes:
                         current_row = [gap_boxes[0]]
                         rows.append(current_row)
                         
                         for i in range(1, len(gap_boxes)):
                             prev_b = current_row[-1][0] # rect of last box in current row
                             curr_b = gap_boxes[i][0]
                             
                             # Check vertical overlap to determine if same line
                             # rect is [nx0, ny0, nx1, ny1]
                             
                             y0_a, y1_a = prev_b[1], prev_b[3]
                             y0_b, y1_b = curr_b[1], curr_b[3]
                             
                             # Intersection
                             inter_y0 = max(y0_a, y0_b)
                             inter_y1 = min(y1_a, y1_b)
                             intersection = max(0, inter_y1 - inter_y0)
                             
                             min_height = min(y1_a - y0_a, y1_b - y0_b)
                             
                             # Robust Line Check: Significant overlap (> 50% of the shorter box)
                             if intersection > (min_height * 0.5):
                                 current_row.append(gap_boxes[i])
                             else:
                                 # New line
                                 current_row = [gap_boxes[i]]
                                 rows.append(current_row)
                     
                     # Distribute tokens to rows
                     # Simple heuristic: Split tokens equally-ish among rows?
                     # Or just assign all tokens to the "Union of all rows"? 
                     # Merging all rows is what caused the "Diagonal Box" issue.
                     # Let's try to distribute tokens by "Row Width" weight.
                     
                     total_width = 0
                     row_widths = []
                     for row in rows:
                         # Union width of row
                         xs = [b[0][0] for b in row] # x0
                         xe = [b[0][2] for b in row] # x1
                         w = max(xe) - min(xs)
                         row_widths.append(w)
                         total_width += w
                         
                     # Assign tokens
                     token_start = 0
                     for i, row in enumerate(rows):
                         if total_width == 0:
                             chunk = gap_tokens
                         else:
                             # Calculate roughly how many tokens fit in this row
                             ratio = row_widths[i] / total_width
                             count = int(round(len(gap_tokens) * ratio))
                             if count == 0 and len(gap_tokens) > 0 and token_start < len(gap_tokens):
                                 count = 1 # Ensure at least 1 if possible
                             
                             # Adjust last row to take everything remaining
                             if i == len(rows) - 1:
                                 chunk = gap_tokens[token_start:]
                             else:
                                 chunk = gap_tokens[token_start : token_start + count]
                                 token_start += count
                         
                         if not chunk: continue
                         
                         gap_text = " ".join(chunk)
                         
                         # Union rect for this ROW only
                         xs0 = [b[0][0] for b in row]
                         ys0 = [b[0][1] for b in row]
                         xs1 = [b[0][2] for b in row]
                         ys1 = [b[0][3] for b in row]
                         union_rect = [min(xs0), min(ys0), max(xs1), max(ys1)]
                         
                         if len(gap_text) > 50: gap_text += "\n"
                         
                         final_output.append((union_rect, gap_text))
                 else:
                     # We have text but no boxes (EasyOCR missed it).
                     # Strategy: Append it after the last known element.
                     gap_text = " ".join(gap_tokens)
                     logging.debug(f"DEBUG: Orphaned LLM text (no box): '{gap_text[:20]}...' - Creating Synthetic Box")
                     
                     if final_output:
                         # Get last rect to calculate Y
                         last_rect, _ = final_output[-1]
                         
                         h = last_rect[3] - last_rect[1]
                         if h < 0.02: h = 0.02 # Force min height
                         
                         new_y0 = last_rect[3] + (h * 0.2) 
                         new_y1 = new_y0 + h 
                         
                         if new_y1 > 0.98:
                             new_y1 = 0.98
                             new_y0 = max(0.9, new_y1 - h)
                             
                         # Use FULL WIDTH (with margins) for synthetic box to prevent truncation
                         synthetic_rect = [0.05, new_y0, 0.95, new_y1]
                         final_output.append((synthetic_rect, gap_text))
                     else:
                         # No previous output? Just put it at top or full page.
                         final_output.append(([0.0, 0.0, 1.0, 0.1], gap_text))
            
            if anchor["text"] is not None:
                final_output.append((anchor["rect"], anchor["text"]))
                
            prev_box_idx = curr_box_idx
            prev_llm_idx = curr_llm_idx

        # DEBUG: Print first few output items to verify text source
        if final_output:
            logging.debug(f"DEBUG: Final Output Sample: {final_output[:3]}")

        if not final_output and structured_data:
             logging.debug("DEBUG: No anchors found. Fallback to Full Page.")
             return [([0.0, 0.0, 1.0, 1.0], full_llm_text + "\n")] 
             
        return final_output

