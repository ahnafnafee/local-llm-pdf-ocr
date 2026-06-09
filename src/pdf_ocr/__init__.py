"""
Local LLM PDF OCR - Package for OCR processing using local vision models.

Converts scanned PDFs into searchable documents using local vision LLMs
for text extraction and Surya for layout detection.
"""

__version__ = "1.0.0"

from pdf_ocr.core.aligner import HybridAligner
from pdf_ocr.core.grounded import (
    DEFAULT_GROUNDING_PROMPT,
    GroundedBlock,
    GroundedOCRBackend,
    GroundedResponse,
    PromptedGroundedOCR,
    ZAIHostedOCR,
    parse_glm_layout_details,
    parse_zai_response,
)
from pdf_ocr.core.html import HTMLHandler
from pdf_ocr.core.markdown import MarkdownHandler
from pdf_ocr.core.ocr import OCRProcessor
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.output import (
    SUPPORTED_FORMATS,
    format_from_path,
    media_type_for,
    resolve_output_writer,
    suffix_for_format,
)
from pdf_ocr.pipeline import OCRPipeline, parse_page_range

__all__ = [
    "PDFHandler",
    "HTMLHandler",
    "MarkdownHandler",
    "OCRProcessor",
    "HybridAligner",
    "OCRPipeline",
    "GroundedBlock",
    "GroundedResponse",
    "GroundedOCRBackend",
    "PromptedGroundedOCR",
    "ZAIHostedOCR",
    "DEFAULT_GROUNDING_PROMPT",
    "parse_zai_response",
    "parse_glm_layout_details",
    "parse_page_range",
    "resolve_output_writer",
    "format_from_path",
    "suffix_for_format",
    "media_type_for",
    "SUPPORTED_FORMATS",
    "__version__",
]
