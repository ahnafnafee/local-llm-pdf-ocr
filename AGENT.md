# Local LLM PDF OCR - Agent Reference

Quick reference for AI assistants working on this codebase.

## Project Overview

A local OCR tool that transforms scanned PDFs into searchable documents using:

-   **Surya** for fast bounding box detection (no text recognition)
-   **Local LLM** (OlmOCR via LM Studio) for text extraction
-   **PyMuPDF** for creating "sandwich" PDFs with invisible text layers

## Architecture

```
Input PDF → Images → Surya Detection → LLM OCR → Align Text → Searchable PDF
                     (boxes only)      (content)   (combine)
```

## Project Structure

```
local-llm-pdf-ocr/
├── src/pdf_ocr/              # Core package
│   ├── core/
│   │   ├── aligner.py        # HybridAligner - detection + text alignment
│   │   ├── ocr.py            # OCRProcessor - async LLM text extraction
│   │   └── pdf.py            # PDFHandler - PDF↔image, text embedding
│   └── utils/
│       └── tqdm_patch.py     # Silences Surya progress bars
├── scripts/                  # Debug/visualization tools
├── static/                   # Web UI (HTML/CSS/JS)
├── examples/                 # Sample PDFs for testing
├── main.py                   # CLI entry point
└── server.py                 # FastAPI web server
```

## Key Classes

| Class           | Location                      | Purpose                                    |
| --------------- | ----------------------------- | ------------------------------------------ |
| `PDFHandler`    | `src/pdf_ocr/core/pdf.py`     | Convert PDF↔images, embed invisible text   |
| `OCRProcessor`  | `src/pdf_ocr/core/ocr.py`     | Async LLM API calls for text extraction    |
| `HybridAligner` | `src/pdf_ocr/core/aligner.py` | Surya detection + position-based alignment |

## Import Pattern

```python
from src.pdf_ocr.core.pdf import PDFHandler
from src.pdf_ocr.core.ocr import OCRProcessor
from src.pdf_ocr.core.aligner import HybridAligner
```

**Important**: Use lazy imports in CLI scripts to keep `--help` fast (avoid loading Surya at module level).

## Entry Points

-   **CLI**: `uv run main.py input.pdf [output.pdf] [--dpi N] [--pages 1-3,5]`
-   **Web**: `uv run uvicorn server:app --port 8000`

## Configuration

Via `.env` or CLI args:

```
LLM_API_BASE=http://localhost:1234/v1
LLM_MODEL=allenai/olmocr-2-7b
```

## Development Notes

-   Coordinates are normalized (0..1), scaled to PDF points when embedding
-   OCRProcessor uses async OpenAI client for LLM calls
-   HybridAligner sorts boxes top→bottom, left→right, then distributes text by box width
-   Server uses WebSocket for real-time progress updates
