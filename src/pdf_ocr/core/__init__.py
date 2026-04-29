"""Core OCR processing modules."""

from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.core.ocr import OCRProcessor
from pdf_ocr.core.aligner import HybridAligner

__all__ = ["PDFHandler", "OCRProcessor", "HybridAligner"]
