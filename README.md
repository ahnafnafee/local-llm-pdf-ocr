# 📄 Local LLM PDF OCR

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![Local AI](https://img.shields.io/badge/AI-100%25_Local-orange?style=for-the-badge)](https://lmstudio.ai)

> **Transform scanned and written documents into fully searchable, selectable PDFs using the power of Local LLM Vision.**

**PDF LLM OCR** is a next-generation OCR tool that moves beyond traditional Tesseract-based scanning. By leveraging OCR Vision Language Models (VLMs) like `olmOCR` running locally on your machine, it "reads" documents with human-like understanding while keeping 100% of your data private.

---

## ✨ Features

-   **🧠 AI-Powered Vision**: Uses advanced VLMs to transcribe text with high accuracy, even on complex layouts or noisy scans.
-   **🤝 Hybrid Alignment Strategy**: Combines **Surya OCR Detection** for precise bounding boxes with **Local LLM** for perfect text content via position-based alignment.
-   **⚡ 10-21x Faster Detection**: Uses detection-only mode (skips slow recognition) and batch processing for maximum speed.
-   **🔒 100% Local & Private**: No cloud APIs, no subscription fees. Run it entirely offline using [LM Studio](https://lmstudio.ai).
-   **🔍 Searchable Outputs**: Embeds an invisible text layer directly into your PDF, making it compatible with valid PDF readers for searching (Ctrl+F) and selecting.
-   **🖥️ Dual Interfaces**:
    -   **Web UI**: An interface with Drag & Drop, Dark Mode, and Real-time progress tracking.
    -   **CLI**: A robust command-line tool for power users and batch automation, featuring a "lively" terminal UI.
-   **⚡ Real-time Feedback**: Watch your document process page-by-page with live web sockets or animated terminal bars.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Input PDF] --> B[PDF to Image Conversion]
    B --> C[Batch Processing]

    subgraph "Phase 1: Layout Detection (Surya)"
        C --> D[Surya DetectionPredictor]
        D --> E[Bounding Boxes]
        E --> F[Sorted by Reading Order]
    end

    subgraph "Phase 2: Text Extraction (Local LLM)"
        C --> G[OlmOCR Vision Model]
        G --> H[Pure Text Content]
    end

    F --> I[Position-Based Aligner]
    H --> I

    I -->|Distribute by Box Width| J[Aligned Text Blocks]
    J --> K[Sandwich PDF Generator]
    K --> L[Searchable PDF Output]
```

### How It Works

1. **Batch Layout Detection**: Surya's `DetectionPredictor` processes all pages at once, extracting bounding boxes without slow text recognition (~1s total vs ~20s per page with recognition).

2. **LLM Text Extraction**: A local vision model (OlmOCR) reads each page with human-like understanding, handling handwriting and complex layouts perfectly.

3. **Position-Based Alignment**: The aligner distributes LLM text across detected boxes proportionally by box width in reading order—no fuzzy matching needed.

4. **Sandwich PDF**: The original page is rendered as an image with invisible, searchable text overlaid using PyMuPDF.

---

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.10+**
2.  **LM Studio**: Download and install [LM Studio](https://lmstudio.ai).
    -   Load a Vision Model (highly recommended: `allenai/olmocr-2-7b`).
    -   Start the Local Server at default port `1234`.

### Configuration

Create a `.env` file in the root directory to configure your Local LLM:

```env
LLM_API_BASE=http://localhost:1234/v1
LLM_MODEL=allenai/olmocr-2-7b
```

### Installation

This project is managed with [`uv`](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1.  **Install `uv`** (if not installed):

    ```bash
    pip install uv
    ```

2.  **Clone the repository**:

    ```bash
    git clone https://github.com/ahnafnafee/pdf-ocr-llm.git
    cd pdf-ocr-llm
    ```

3.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

---

## Usage

### 1. 🌐 Web Interface (Recommended)

The easiest way to use the tool. Features a modern dashboard with Dark Mode and Text Preview.

1.  **Start the Server**:
    ```bash
    uv run uvicorn server:app --reload --port 8000
    ```
2.  Open your browser to `http://localhost:8000`.
3.  **Drag & Drop** your PDF.
4.  Watch the magic happen! ✨
    -   **Real-time Progress**: Track per-page OCR status.
    -   **Preview**: Click "View Text" to inspect the raw AI extraction.
    -   **Dark Mode**: Toggle the moon icon for a sleek dark theme.

### 2. 💻 Command Line Interface (CLI)

Perfect for developers or integrating into scripts.

Run the OCR tool on any PDF:

```bash
uv run main.py input.pdf output_ocr.pdf
```

**Options**:

| Option                    | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| `input_pdf`               | Path to input PDF (required)                                          |
| `output_pdf`              | Path to output PDF (optional, defaults to `<input>_ocr.pdf`)          |
| `-v`, `--verbose`         | Enable debug logging (alignment details, box counts)                  |
| `-q`, `--quiet`           | Suppress all output except errors                                     |
| `--dpi <int>`             | DPI for image rendering (default: 200)                                |
| `--pages <range>`         | Page range to process, e.g., `1-3,5` (default: all)                   |
| `--concurrency <int>`     | Parallel LLM requests (default: 1)                                    |
| `--no-refine`             | Skip per-box crop re-OCR (faster, less robust on tables/multi-column) |
| `--max-image-dim <int>`   | Longest-edge px cap for page images (default: 1024; see note below)   |
| `--api-base <url>`        | Override LLM API base URL                                             |
| `--model <name>`          | Override LLM model name                                               |

**Examples**:

```bash
# Basic usage (auto-generates input_ocr.pdf, uses LM Studio + OlmOCR)
uv run main.py scan.pdf

# Specific pages with higher rendering DPI
uv run main.py document.pdf output.pdf --pages 1-5 --dpi 300

# Parallel LLM calls on a multi-page doc
uv run main.py long.pdf --concurrency 3

# Use Ollama + GLM-OCR instead of LM Studio
uv run main.py scan.pdf \
    --api-base http://localhost:11434/v1 \
    --model glm-ocr:latest \
    --max-image-dim 640

# Grounded path: bbox-native VLM (Qwen2.5-VL / Qwen3-VL) — skips Surya, DP, refine
uv run main.py scan.pdf --grounded \
    --api-base http://localhost:1234/v1 \
    --model qwen/qwen3-vl-8b

# Raw image input — no PDF required. Accepts JPEG/PNG/BMP/WebP, and
# multi-page TIFFs (each frame becomes one page in the output PDF).
uv run main.py scan.png scan_ocr.pdf
uv run main.py archive.tiff archive_ocr.pdf
```

### Two pipeline paths

| Path | Flag | Detection | Text | Alignment | Refine | When to use |
|------|------|-----------|------|-----------|--------|-------------|
| **Hybrid** (default) | _none_ | Surya | LLM full-page | DP | Per-box crop | Text-only VLMs (OlmOCR, GLM-OCR); max coverage |
| **Grounded** | `--grounded` | — | Bbox-native VLM returns both | — | — | Qwen2.5/3-VL, MinerU, etc.; simpler, fewer moving parts |

The hybrid path is the safe default: it works with *any* OCR-capable VLM, including models that can only return plain text. The grounded path is faster and eliminates the DP-alignment class of bugs entirely, but requires a VLM that emits `{"bbox_2d": [...], "content": "..."}` JSON when asked (Qwen2.5-VL / Qwen3-VL confirmed working; others untested).

> **Note on `--max-image-dim`**: small local VLMs have tight context windows.
> OlmOCR-2-7B (Qwen2.5-VL base) is happy with the 1024 default.
> **GLM-OCR:1.1B via Ollama crashes its runner above ~640 px**, so drop the
> cap when you use it. If Ollama dies mid-run, restart it with `ollama serve`
> and lower `--max-image-dim`.

_You'll see animated progress bars showing detection, LLM OCR, refinement, and embedding._

---

## 📁 Project Structure

```
local-llm-pdf-ocr/
├── src/pdf_ocr/           # Core package
│   ├── core/              # OCR processing modules
│   │   ├── aligner.py     # Hybrid text alignment
│   │   ├── ocr.py         # LLM OCR processor
│   │   └── pdf.py         # PDF handling utilities
│   └── utils/             # Utility modules
│       └── tqdm_patch.py  # Progress bar silencer
├── scripts/               # Debug and visualization tools
├── static/                # Web UI assets
├── examples/              # Sample PDFs
├── main.py                # CLI entry point
└── server.py              # Web server
```

---

## 🛠️ Tech Stack

-   **Backend**: FastAPI (Async Web Framework)
-   **Frontend**: Vanilla JS + CSS Variables
-   **PDF Processing**: PyMuPDF (Fitz)
-   **Layout Detection**: Surya OCR (Detection-only mode)
-   **AI Integration**: OpenAI Client (compatible with Local LLM servers)
-   **CLI UI**: Rich (Terminal formatting)

---

## ⚡ Performance

| Document Type | Detection Time | Speedup vs Recognition |
| ------------- | -------------- | ---------------------- |
| Digital PDF   | ~1s            | **21x faster**         |
| Handwritten   | ~1s            | **10x faster**         |
| Hybrid Form   | ~1s            | **11x faster**         |

_Detection uses batch processing—all pages in one call._

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**License**: MIT
