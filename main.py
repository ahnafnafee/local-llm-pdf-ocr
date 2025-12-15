import os
# Disable TQDM for Surya to prevent collision with Rich
os.environ["TQDM_DISABLE"] = "1"

import sys
import base64
from pathlib import Path
from pdf_utils import PDFHandler
from ocr import OCRProcessor
from hybrid_aligner import HybridAligner

# Rich imports
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.live import Live

import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="PDF OCR with Local LLM")
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("output_pdf", help="Path to output PDF")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stderr)

    input_pdf_path = args.input_pdf
    output_pdf_path = args.output_pdf
    
    console = Console()
    
    # 1. Initialize Components
    pdf_handler = PDFHandler()
    ocr_processor = OCRProcessor()

    # 2. Convert PDF to Images
    console.print(f"[bold cyan]Processing '{input_pdf_path}'...[/bold cyan]")
    
    try:
        images_dict = pdf_handler.convert_to_images(input_pdf_path)
    except Exception as e:
        console.print(f"[bold red]Error converting PDF:[/bold red] {e}")
        sys.exit(1)
        
    page_nums = sorted(images_dict.keys())
    total_pages = len(page_nums)
    
    console.print(f"[green]✓[/green] Converted [bold]{total_pages}[/bold] pages.")

    pages_text = {}
    
    # 3. Hybrid OCR Loop with Rich Progress
    hybrid_aligner = HybridAligner()

    pages_structured_data = {}
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        # ===== PHASE 1: Batch Layout Detection (Surya) =====
        task_layout = progress.add_task("[cyan]Detecting layouts (batch)...", total=1)
        
        # Collect all image bytes for batch processing
        all_image_bytes = []
        for page_num in page_nums:
            image_base64 = images_dict[page_num]
            image_bytes = base64.b64decode(image_base64)
            all_image_bytes.append(image_bytes)
        
        # Run batch detection (one Surya call for all pages)
        all_boxes = hybrid_aligner.get_detected_boxes_batch(all_image_bytes)
        progress.update(task_layout, completed=1)
        
        # ===== PHASE 2: LLM OCR + Alignment (per page) =====
        task_ocr = progress.add_task(f"[cyan]LLM OCR Processing ({total_pages} pages)...", total=total_pages)
        
        for idx, page_num in enumerate(page_nums):
            progress.update(task_ocr, description=f"[cyan]LLM OCR Page {page_num + 1}/{total_pages}...")
            
            image_base64 = images_dict[page_num]
            boxes = all_boxes[idx]
            
            # Convert boxes to structured_data format
            structured_data = [(box, "") for box in boxes]
            
            # LLM OCR (Content)
            llm_lines = ocr_processor.perform_ocr(image_base64)
            logging.debug(f"DEBUG: LLM OCR retrieved {len(llm_lines)} lines for page {page_num}")
            
            # Align
            if llm_lines:
                aligned_data = hybrid_aligner.align_text(structured_data, llm_lines)
                pages_structured_data[page_num] = aligned_data
            else:
                pages_structured_data[page_num] = structured_data
            
            progress.advance(task_ocr)

    console.print("[green]✓[/green] OCR & Layout Analysis Complete.")

    # 4. Embed Text
    console.print("[bold blue]Embedding text into PDF...[/bold blue]")
    pdf_handler.embed_structured_text(input_pdf_path, output_pdf_path, pages_structured_data)

    console.print(f"[bold green]Done! Saved to '{output_pdf_path}'[/bold green]")

if __name__ == "__main__":
    main()
