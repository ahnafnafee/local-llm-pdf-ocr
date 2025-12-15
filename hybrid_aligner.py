import io
import sys
import logging

import tqdm_patch
# Silence Surya's progress bars to prevent collision with Rich
tqdm_patch.apply()

from PIL import Image
from surya.detection import DetectionPredictor

class HybridAligner:
    """
    Detection-Only OCR Aligner.
    
    Uses Surya's DetectionPredictor (fast) for bounding boxes,
    then distributes LLM text across boxes using position-based alignment.
    
    This is ~10-21x faster than using RecognitionPredictor since
    we skip the expensive text recognition step.
    """
    
    def __init__(self):
        # Initialize ONLY the detection predictor (no recognition = fast!)
        self.detection_predictor = DetectionPredictor()

    def get_detected_boxes(self, image_bytes):
        """
        Run Surya detection on image -> list of bounding boxes (no text).
        Returns: list of [nx0, ny0, nx1, ny1] (normalized 0..1)
        
        This is the fast path - detection only, no recognition.
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size
        
        # Run Surya detection only (FAST: ~1s vs ~20s for recognition)
        predictions = self.detection_predictor([image])
        
        boxes = []
        
        if predictions and predictions[0].bboxes:
            for bbox in predictions[0].bboxes:
                # bbox.bbox is [x0, y0, x1, y1]
                x0, y0, x1, y1 = bbox.bbox
                
                # Normalize to 0..1
                nx0 = x0 / img_w
                ny0 = y0 / img_h
                nx1 = x1 / img_w
                ny1 = y1 / img_h
                
                # Ensure coordinates are within bounds
                nx0 = max(0.0, min(1.0, nx0))
                ny0 = max(0.0, min(1.0, ny0))
                nx1 = max(0.0, min(1.0, nx1))
                ny1 = max(0.0, min(1.0, ny1))
                
                boxes.append([nx0, ny0, nx1, ny1])
            
            # Sort by Y (top-to-bottom), then X (left-to-right)
            boxes.sort(key=lambda b: (b[1], b[0]))
            
        return boxes

    def get_detected_boxes_batch(self, images_bytes_list):
        """
        BATCH detection: Process multiple images in one Surya call.
        Much faster than calling get_detected_boxes() for each page individually.
        
        Args:
            images_bytes_list: list of image bytes (one per page)
            
        Returns:
            list of boxes for each page, where each is list of [nx0, ny0, nx1, ny1]
        """
        # Load all images
        images = []
        sizes = []
        for img_bytes in images_bytes_list:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(image)
            sizes.append(image.size)  # (width, height)
        
        if not images:
            return []
        
        # Run Surya detection on ALL images at once (batch processing)
        predictions = self.detection_predictor(images)
        
        # Process results for each page
        all_boxes = []
        for page_idx, pred in enumerate(predictions):
            img_w, img_h = sizes[page_idx]
            boxes = []
            
            if pred.bboxes:
                for bbox in pred.bboxes:
                    x0, y0, x1, y1 = bbox.bbox
                    
                    # Normalize to 0..1
                    nx0 = x0 / img_w
                    ny0 = y0 / img_h
                    nx1 = x1 / img_w
                    ny1 = y1 / img_h
                    
                    # Ensure coordinates are within bounds
                    nx0 = max(0.0, min(1.0, nx0))
                    ny0 = max(0.0, min(1.0, ny0))
                    nx1 = max(0.0, min(1.0, nx1))
                    ny1 = max(0.0, min(1.0, ny1))
                    
                    boxes.append([nx0, ny0, nx1, ny1])
                
                # Sort by Y (top-to-bottom), then X (left-to-right)
                boxes.sort(key=lambda b: (b[1], b[0]))
            
            all_boxes.append(boxes)
        
        return all_boxes

    def get_structured_text(self, image_bytes):
        """
        Legacy compatibility wrapper.
        Returns boxes with empty text placeholders for backward compatibility.
        The actual text will be filled in by align_text().
        """
        boxes = self.get_detected_boxes(image_bytes)
        # Return as (rect, "") tuples for compatibility with existing code
        return [(box, "") for box in boxes]

    def align_text(self, structured_data, llm_text):
        """
        Position-based alignment: distributes LLM text across detected boxes.
        
        Since we use detection-only (no OCR text), we can't do anchor matching.
        Instead, we distribute tokens proportionally based on box widths
        in reading order (top-to-bottom, left-to-right).
        
        Args:
            structured_data: list of ([nx0, ny0, nx1, ny1], placeholder_text) tuples
            llm_text: string or list of lines from LLM OCR
            
        Returns:
            list of ([nx0, ny0, nx1, ny1], text) tuples with LLM text distributed
        """
        if not llm_text:
            return structured_data
            
        # Parse LLM text into tokens
        if isinstance(llm_text, list):
            full_llm_text = " ".join(llm_text)
        else:
            full_llm_text = llm_text
            
        llm_tokens = full_llm_text.split()
        if not llm_tokens:
            return structured_data
        
        # Extract just the boxes (ignore placeholder text)
        boxes = [item[0] for item in structured_data]
        
        if not boxes:
            # No boxes detected - return full text in a single full-page box
            logging.debug("DEBUG: No boxes detected. Using full-page fallback.")
            return [([0.0, 0.0, 1.0, 1.0], full_llm_text)]
        
        # Calculate total width of all boxes for proportional distribution
        total_width = 0
        box_widths = []
        for box in boxes:
            w = box[2] - box[0]  # x1 - x0
            box_widths.append(w)
            total_width += w
        
        # Distribute tokens across boxes proportionally by width
        final_output = []
        token_idx = 0
        num_tokens = len(llm_tokens)
        
        for i, box in enumerate(boxes):
            if total_width == 0:
                # All boxes have zero width - distribute evenly
                count = num_tokens // len(boxes)
                if i == len(boxes) - 1:
                    count = num_tokens - token_idx  # Last box gets remainder
            else:
                # Proportional by width
                ratio = box_widths[i] / total_width
                count = int(round(num_tokens * ratio))
                
                # Ensure last box gets all remaining tokens
                if i == len(boxes) - 1:
                    count = num_tokens - token_idx
            
            # Extract tokens for this box
            chunk = llm_tokens[token_idx : token_idx + count]
            token_idx += count
            
            if chunk:
                text = " ".join(chunk)
                final_output.append((box, text))
            else:
                # No tokens for this box, but we still might want to keep it
                # for layout fidelity. Skip for now to avoid empty boxes.
                pass
        
        # Handle any remaining tokens (shouldn't happen, but safety net)
        if token_idx < num_tokens:
            remaining = " ".join(llm_tokens[token_idx:])
            if final_output:
                # Append to last box
                last_box, last_text = final_output[-1]
                final_output[-1] = (last_box, last_text + " " + remaining)
            else:
                # Create a full-page box
                final_output.append(([0.0, 0.0, 1.0, 1.0], remaining))
        
        logging.debug(f"DEBUG: Distributed {num_tokens} tokens across {len(final_output)} boxes")
        
        return final_output
