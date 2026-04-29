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
-   **🤝 DP-Based Text↔Box Alignment**: **Surya OCR** detects layout boxes; a **Local LLM** transcribes the whole page; a Needleman-Wunsch dynamic-programming aligner binds LLM lines to the correct boxes in reading order, with a per-box crop re-OCR fallback for boxes the DP cannot confidently populate.
-   **🛰️ Grounded Path (opt-in)**: Point the tool at a bbox-native VLM (Qwen2.5-VL, Qwen3-VL, MinerU, Florence-2, …) with `--grounded` and it skips Surya/DP/refine entirely — the model returns text + coordinates in a single call.
-   **🖼️ PDF or Raw Image Input**: Accepts **`.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`/`.tiff`, `.avif`**. Multi-frame TIFFs become multi-page output PDFs — no manual PDF-wrap step.
-   **⚡ Fast Detection**: Surya runs in detection-only mode (no recognition) and batches across pages.
-   **🔒 100% Local & Private**: No cloud APIs, no subscription fees. Run it entirely offline using [LM Studio](https://lmstudio.ai) or [Ollama](https://ollama.com).
-   **🔍 Searchable Outputs**: Embeds an invisible text layer into a sandwich PDF. Glyph bboxes are horizontally scaled so selection in a PDF viewer covers the full width of each text region.
-   **🖥️ Dual Interfaces**:
    -   **Web UI**: Drag & drop, Dark Mode, real-time per-page progress.
    -   **CLI**: Documented flags for power users and batch automation, Rich progress bars.
-   **📚 Dense-Page Mode**: Auto-detects densely-laid-out pages (default >60 detected boxes) and switches to per-box OCR — bypasses the failure modes (loops, hallucination, pangram fallback) that full-page OCR exhibits on dense handwritten content. Configurable via `--dense-mode` and `--dense-threshold`.
-   **🧪 Tested**: 167-test suite covering DP invariants, reading-order auto-detection, blank-crop / pangram filters, embedding geometry, grounded JSON parsing, and end-to-end runs against the example PDFs.

---

## 🏗️ Architecture

The tool has two execution paths behind a single `OCRPipeline` seam (`src/pdf_ocr/pipeline.py`). The default **hybrid path** works with any OCR-capable VLM; the opt-in **grounded path** collapses the whole flow into one call for VLMs that emit text+bbox natively.

```mermaid
graph TD
    A[Input: PDF / JPEG / PNG / TIFF / AVIF] --> B[Rasterize to images]
    B -->|--grounded| Z[Grounded VLM: text+bbox in one call]
    Z --> EMB

    B -->|default| C[Surya DetectionPredictor<br/>batch, detection-only]
    C --> DM{Dense?<br/>boxes/page > threshold}
    DM -->|sparse| D[LLM full-page OCR<br/>OlmOCR / GLM-OCR / etc.]
    DM -->|dense| P[Per-box OCR<br/>each Surya box → LLM crop]
    D --> F[Plain text with line breaks]
    C --> E[Layout boxes in reading order]
    E --> G[Needleman-Wunsch DP aligner<br/>line ↔ box, auto row/column-major]
    F --> G
    G --> H{Boxes the DP<br/>left empty?}
    H -->|yes| R[Per-box crop re-OCR<br/>refine stage]
    H -->|no| EMB[Sandwich PDF writer]
    R --> EMB
    P --> EMB
    EMB --> L[Searchable PDF output]
```

### How It Works

1. **Input**: PDFs *or* raw images. Multi-frame TIFFs expand to one page per frame. Images skip the PDF round-trip and feed straight into the pipeline.

2. **Batch Layout Detection** *(hybrid path)*: Surya's `DetectionPredictor` processes all pages in one call, ~10-21× faster than running full recognition.

3. **LLM Text Extraction** *(hybrid path)*: A local vision model (OlmOCR by default via LM Studio) transcribes each page's full content with human-like understanding. **Dense pages (>60 detected boxes by default) automatically switch to per-box OCR** instead — the model sees one Surya box at a time, which avoids the loop / hallucination failure modes that full-page OCR exhibits on dense handwritten content.

4. **Needleman-Wunsch Alignment** *(hybrid path, full-page mode)*: The DP aligner binds each LLM line to its Surya box using character-count fit + reading-order monotonicity. **Model-agnostic**: it tries both row-major and column-major box orderings and picks the lower-cost result, so it works whether the LLM emits text column-by-column (OlmOCR-2) or row-by-row (Qwen-VL family). Cheap `skip_box` ops (many detected boxes are rules/decorations), expensive `skip_line` ops — but unmatched lines are attached to the nearest matched box so no LLM text is lost.

5. **Refine Fallback** *(hybrid path, optional)*: Any sizeable box the DP couldn't populate gets its image crop re-OCR'd individually. A pre-OCR blank-crop check (pixel stddev) skips dotted notebook backgrounds and other near-uniform regions to avoid the model's "The quick brown fox..." pangram fallback. Disable refine entirely with `--no-refine`.

6. **Grounded Path** *(opt-in alternative)*: With `--grounded` pointed at a bbox-native VLM (Qwen2.5-VL, Qwen3-VL, MinerU, …), the model returns `{bbox, text}` tuples in a single call — Surya, DP, and refine are all skipped. The grounding prompt explicitly demands one element per visual line so wrapped phrases stay separated.

7. **Sandwich PDF**: The page is rasterized as a background image and invisible text is overlaid with horizontal-scale matrices so glyph bboxes span the full width of each source box — selection in a PDF viewer correctly covers the whole region.

---

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.10+**
2.  **A local OpenAI-compatible LLM server**. Any of:
    -   **[LM Studio](https://lmstudio.ai)** — recommended default. Load `allenai/olmocr-2-7b` (hybrid path) or `qwen/qwen3-vl-8b` / `qwen/qwen2.5-vl-7b` (grounded path). Start the local server (default port `1234`).
    -   **[Ollama](https://ollama.com)** — pull `glm-ocr:latest` (requires `--max-image-dim 640`) or any vision model. Served at `http://localhost:11434/v1`.
    -   **vLLM / SGLang / any OpenAI-compatible endpoint**.

### Configuration

Create a `.env` file in the root directory to configure your Local LLM:

```env
LLM_API_BASE=http://localhost:1234/v1
LLM_MODEL=allenai/olmocr-2-7b
```

### Installation

#### Option A — Install from PyPI (recommended for end users / coding agents)

CLI only:

```bash
pipx install local-llm-pdf-ocr                  # or: uv tool install local-llm-pdf-ocr
```

CLI + Web server (FastAPI + WebSocket UI):

```bash
pipx install "local-llm-pdf-ocr[web]"           # or: uv tool install "local-llm-pdf-ocr[web]"
```

After install you'll have two console scripts on `PATH`:

- `local-llm-pdf-ocr` — the CLI (always available)
- `local-llm-pdf-ocr-server` — the web UI (only with the `[web]` extra)

> **Heads up:** Surya downloads its detection model from Hugging Face Hub on first run (~500 MB, cached afterwards). The hybrid/grounded LLM is *your* responsibility — bring up LM Studio, Ollama, vLLM, or any other OpenAI-compatible vision endpoint before running OCR.

#### Option B — Develop from source with `uv`

1.  **Install `uv`** (if not installed):

    ```bash
    pip install uv
    ```

2.  **Clone the repository**:

    ```bash
    git clone https://github.com/ahnafnafee/local-llm-pdf-ocr.git
    cd local-llm-pdf-ocr
    ```

3.  **Sync dependencies** (installs the project + dev tools in editable mode):

    ```bash
    uv sync                       # CLI only
    uv sync --extra web           # CLI + FastAPI server
    ```

---

## Usage

### 1. 🌐 Web Interface (Recommended)

The easiest way to use the tool. Features a modern dashboard with Dark Mode and Text Preview.

1.  **Start the Server**:
    ```bash
    local-llm-pdf-ocr-server --port 8000          # installed via `pipx install local-llm-pdf-ocr[web]`
    # — or, in a source checkout —
    uv run local-llm-pdf-ocr-server --port 8000
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
local-llm-pdf-ocr input.pdf output_ocr.pdf      # installed via pipx / uv tool install
# — or, in a source checkout —
uv run local-llm-pdf-ocr input.pdf output_ocr.pdf
```

**Options**:

| Option                    | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| `input`                   | Path to a PDF **or** image file (`.jpg`/`.jpeg`/`.png`/`.bmp`/`.webp`/`.tif`/`.tiff`/`.avif`). Required. Multi-frame TIFFs expand to multiple output pages. |
| `output`                  | Path to output PDF (optional, defaults to `<input_stem>_ocr.pdf`; always a PDF, even for image inputs). |
| `-v`, `--verbose`         | Enable debug logging (alignment details, box counts)                  |
| `-q`, `--quiet`           | Suppress all output except errors                                     |
| `--dpi <int>`             | DPI for image rendering (default: 200)                                |
| `--pages <range>`         | Page range to process, e.g., `1-3,5` (default: all)                   |
| `--concurrency <int>`     | Parallel LLM requests (default: 1; bump to 3-5 for `--dense-mode always`) |
| `--no-refine`             | Skip per-box crop re-OCR (faster, less robust on tables/multi-column) |
| `--max-image-dim <int>`   | Longest-edge px cap for page images (default: 1024; see note below)   |
| `--dense-mode {auto,always,never}` | `auto` (default) switches to per-box OCR for pages above `--dense-threshold`; `always` forces per-box for every page (most accurate on handwriting); `never` keeps the original full-page path. |
| `--dense-threshold <int>` | In `auto` dense-mode, pages with more than this many detected boxes use per-box OCR (default: 60). |
| `--grounded`              | Use a bbox-native VLM that returns text + coordinates in one call (skips Surya, DP, refine). Requires a grounding-capable model via `--model`. |
| `--api-base <url>`        | Override LLM API base URL                                             |
| `--model <name>`          | Override LLM model name                                               |

**Examples** (substitute `uv run local-llm-pdf-ocr` for `local-llm-pdf-ocr` if developing from source):

```bash
# Basic usage (auto-generates input_ocr.pdf, uses LM Studio + OlmOCR)
local-llm-pdf-ocr scan.pdf

# Specific pages with higher rendering DPI
local-llm-pdf-ocr document.pdf output.pdf --pages 1-5 --dpi 300

# Parallel LLM calls on a multi-page doc
local-llm-pdf-ocr long.pdf --concurrency 3

# Use Ollama + GLM-OCR instead of LM Studio
local-llm-pdf-ocr scan.pdf \
    --api-base http://localhost:11434/v1 \
    --model glm-ocr:latest \
    --max-image-dim 640

# Grounded path: bbox-native VLM (Qwen2.5-VL / Qwen3-VL) — skips Surya, DP, refine
local-llm-pdf-ocr scan.pdf --grounded \
    --api-base http://localhost:1234/v1 \
    --model qwen/qwen3-vl-8b

# Raw image input — no PDF required. Accepts JPEG/PNG/BMP/WebP/AVIF, and
# multi-page TIFFs (each frame becomes one page in the output PDF).
local-llm-pdf-ocr scan.png scan_ocr.pdf
local-llm-pdf-ocr archive.tiff archive_ocr.pdf
local-llm-pdf-ocr photo.avif photo_ocr.pdf

# Dense handwritten content: force per-box OCR everywhere with extra concurrency
local-llm-pdf-ocr notes.pdf --dense-mode always --concurrency 5

# Custom dense-mode threshold (auto-detect kicks in earlier)
local-llm-pdf-ocr mixed.pdf --dense-threshold 40
```

### Two pipeline paths

| Path | Flag | Detection | Text | Alignment | Refine | When to use |
|------|------|-----------|------|-----------|--------|-------------|
| **Hybrid** (default) | _none_ | Surya | LLM full-page | DP (auto row/column-major) | Per-box crop (with blank-skip) | Text-only VLMs (OlmOCR, GLM-OCR); max coverage |
| **Hybrid + dense** (auto) | `--dense-mode` | Surya | LLM per-box (each Surya box → one crop call) | — (boxes already individually transcribed) | — | Dense handwriting / multi-column where full-page OCR loops or hallucinates |
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
├── src/pdf_ocr/
│   ├── cli.py                 # CLI entry point (`local-llm-pdf-ocr`)
│   ├── server.py              # FastAPI web server (`local-llm-pdf-ocr-server`, requires [web] extra)
│   ├── pipeline.py            # OCRPipeline orchestration seam (hybrid + grounded)
│   ├── core/
│   │   ├── aligner.py         # HybridAligner: Surya detect + Needleman-Wunsch DP
│   │   ├── ocr.py             # OCRProcessor: OpenAI-compat LLM client + crop OCR
│   │   ├── pdf.py             # PDFHandler: PDF/image I/O + sandwich-PDF embedding
│   │   └── grounded.py        # Grounded backends (PromptedGroundedOCR, ZAIHostedOCR) + parsers
│   ├── evaluation.py          # Confidence comparator (IoU + text similarity)
│   ├── static/                # Web UI assets bundled into the wheel
│   └── utils/
│       ├── image.py           # Crop utility for the refine stage
│       └── tqdm_patch.py      # Silences Surya's internal progress bars
├── tests/                     # 167-test suite (fast tier + Surya-integration tier)
│   └── fixtures/              # Ground-truth JSON for confidence evaluation
├── scripts/
│   ├── confidence_eval.py     # Score either path against ground-truth fixtures
│   ├── debug_alignment.py     # Visualize alignment for a single PDF
│   ├── visualize_bboxes.py    # Render Surya's detected boxes
│   └── ...                    # Other debug tools
├── examples/                  # Sample PDFs (digital, hybrid, handwritten)
└── pyproject.toml             # PEP 621 metadata, build backend, console scripts
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

Detection is no longer the bottleneck — full-page LLM OCR is. Rough per-page timings on a warm run (Surya loaded, LM Studio serving OlmOCR-2-7B on a single GPU):

| Phase | Time / page | Notes |
|---|---|---|
| Rasterize PDF → image | ~0.3 s | Linear in pages |
| Surya batch detection | ~0.5 s | Amortized across all pages in one call |
| **LLM full-page OCR** *(sparse pages)* | **~2–4 s** | **Dominant cost on sparse pages.** Set `--concurrency 3` to parallelize on multi-page docs |
| **Per-box OCR** *(dense pages, auto-mode)* | **~0.2–0.4 s × box count** | ~30 s for a 150-box page at `--concurrency 5`. Trades latency for accuracy on dense handwriting where full-page OCR loops or hallucinates |
| Per-box refine (sparse pages, if needed) | ~0.5–1 s × empty boxes | Typically 0–2 s; blank-crop check skips most empties; `--no-refine` to disable |
| PDF assembly | ~0.2 s | Linear in pages |
| Cold-start Surya load | +5–10 s (once) | Paid even on `--grounded` runs |

On our three example PDFs (hybrid path, `allenai/olmocr-2-7b`, warm): digital ≈ 14 s, hybrid ≈ 5 s, handwritten ≈ 4 s. On the dense-handwriting `examples/dense.pdf` (3 pages, ~150 boxes/page), auto-mode picks per-box OCR for all pages and finishes in ~57 s with `--concurrency 5`.

---

## 🧪 Testing

```bash
uv run pytest                      # full suite (~75s, loads Surya once)
uv run pytest -m "not slow"        # fast tier (~15s, no model loads)
uv run pytest tests/test_aligner.py -v
```

Confidence evaluation (needs a live LLM endpoint):

```bash
uv run scripts/confidence_eval.py --path both \
    --grounded-model qwen/qwen3-vl-8b \
    --hybrid-model allenai/olmocr-2-7b
```

Scores either path against the fixtures in `tests/fixtures/ground_truth_*.json` — block recall at IoU≥0.3, average IoU of matched pairs, average text similarity.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**License**: MIT
