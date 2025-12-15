import fitz  # PyMuPDF
from PIL import Image
import io
import base64

class PDFHandler:
    def __init__(self):
        pass

    def pdf_to_base64_images(self, pdf_path):
        """
        Yields (page_number, base64_image_string, page_width, page_height) for each page in the PDF.
        """
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("jpg", jpg_quality=50)
            # Resize to max 1024x1024 to ensure it fits in context window
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail((1024, 1024))
            
            # Save back to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=50)
            base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            yield page_num, base64_img, page.rect.width, page.rect.height
        doc.close()

    def convert_to_images(self, pdf_path):
        """
        Returns a dict of {page_num: base64_image} for all pages.
        """
        images = {}
        for page_num, img, _, _ in self.pdf_to_base64_images(pdf_path):
            images[page_num] = img
        return images


    def embed_text_into_pdf(self, input_pdf_path, output_pdf_path, pages_text):
        """
        Embeds text into the PDF linearly.
        Replaces the original page content with the rasterized image to remove any existing text
        and ensure a clean 'sandwich' PDF.
        """
        doc = fitz.open(input_pdf_path)
        
        for page_num, lines in pages_text.items():
            if page_num < len(doc):
                page = doc[page_num]
                
                # 1. capture page geometry
                rect = page.rect
                width = rect.width
                height = rect.height
                
                # 2. Render page to image (high res for quality)
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                
                # Strategy: Render Image -> Insert Image -> Insert Text (on new page)
                # We build a NEW document page by page to ensure clean "sandwich" PDF.
        new_doc = fitz.open()
        
        for page_num in range(len(doc)):
            old_page = doc[page_num]
            width = old_page.rect.width
            height = old_page.rect.height
            
            # Render image of original page
            pix = old_page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            
            # Create new page in new doc
            new_page = new_doc.new_page(width=width, height=height)
            
            # Draw the image (background)
            new_page.insert_image(new_page.rect, stream=img_data)
            
            # Add text if we have it
            lines = pages_text.get(page_num, [])
            if lines:
                full_text = "\n".join(lines)
                # Define a rectangle with margins (tbh just inside safe area)
                # Using 8pt font to ensure it fits better while still being selectable
                text_rect = fitz.Rect(30, 30, width - 30, height - 30)
                
                # insert_textbox automatically wraps text that hits the right margin
                new_page.insert_textbox(
                    text_rect, 
                    full_text, 
                    fontsize=8, 
                    fontname="helv",
                    render_mode=3,
                    color=(0,0,0) # Invisible but conceptually black
                )
                    
        new_doc.save(output_pdf_path)
        new_doc.close()
        doc.close()


    def embed_structured_text(self, input_pdf_path, output_pdf_path, pages_data):
        """
        Embeds structured text (box, text) into the PDF.
        Replaces page content with image, then overlays boxes.
        pages_data: dict {page_num: [([x0,y0,x1,y1], text), ...]}
        """
        doc = fitz.open(input_pdf_path)
        new_doc = fitz.open()

        for page_num in range(len(doc)):
            old_page = doc[page_num]
            width = old_page.rect.width
            height = old_page.rect.height
            
            # Render image of original page
            pix = old_page.get_pixmap(dpi=200)
            img_data = pix.tobytes("jpg", jpg_quality=80)
            
            # Create new page in new doc
            new_page = new_doc.new_page(width=width, height=height)
            
            # Draw the image (background)
            new_page.insert_image(new_page.rect, stream=img_data)
            
            # Add structured text
            boxes = pages_data.get(page_num, [])
            
            if boxes:
                for (rect_coords, text) in boxes:
                    # rect_coords are assumed to be NORMALIZED (0..1)
                    # We scale them to the PDF page dimensions (points)
                    
                    nx0, ny0, nx1, ny1 = rect_coords
                    
                    pdf_rect = fitz.Rect(
                        nx0 * width, 
                        ny0 * height, 
                        nx1 * width, 
                        ny1 * height
                    )
                    
                    # Calculate dynamic font size based on box height
                    box_height = pdf_rect.height
                    
                    # If text is multi-line (fallback block)
                    if '\n' in text:
                         # Fallback Mode: Full page text.
                         dynamic_fontsize = 6
                         pdf_rect = fitz.Rect(10, 10, width - 10, height - 10)
                         
                         new_page.insert_textbox(
                            pdf_rect, 
                            text, 
                            fontsize=dynamic_fontsize, 
                            fontname="helv",
                            render_mode=3,
                            color=(0, 0, 0),
                            align=0
                         )
                    else:
                        # Single line: Size to fill box width, constrained by height
                        font = fitz.Font("helv")
                        box_width = pdf_rect.width
                        
                        # Strategy: Start with a reference size and calculate what we need
                        # to fill the box width, then constrain by height
                        
                        # Calculate font size needed to fill box width
                        # Use reference size of 12pt to measure text width ratio
                        ref_size = 12.0
                        ref_width = font.text_length(text, fontsize=ref_size)
                        
                        if ref_width > 0:
                            # Scale to fill box width (with small margin)
                            width_based_size = (box_width * 0.98) / ref_width * ref_size
                        else:
                            width_based_size = box_height * 0.8
                        
                        # Constrain by box height (font shouldn't exceed box height)
                        # Typical font ascender is ~0.8 of font size
                        height_based_size = box_height * 0.85
                        
                        # Use the smaller of the two to ensure it fits both dimensions
                        target_fontsize = min(width_based_size, height_based_size)
                        
                        # Enforce min/max constraints
                        dynamic_fontsize = max(3, min(target_fontsize, 72))
                        
                        # Use insert_textbox for proper text placement
                        res = new_page.insert_textbox(
                            pdf_rect, 
                            text, 
                            fontsize=dynamic_fontsize, 
                            fontname="helv",
                            render_mode=3,  # Invisible
                            color=(0, 0, 0),
                            align=0  # Left align
                        )
                        
                        if res < 0:
                            # If insert_textbox fails, use insert_text directly
                            # Position at baseline (bottom of box with slight offset)
                            text_pos = fitz.Point(pdf_rect.x0, pdf_rect.y1 - (box_height * 0.15))
                            new_page.insert_text(
                                text_pos,
                                text,
                                fontsize=dynamic_fontsize, 
                                fontname="helv",
                                render_mode=3, 
                                color=(0, 0, 0)
                            )

        new_doc.save(output_pdf_path)
        new_doc.close()
        doc.close()


