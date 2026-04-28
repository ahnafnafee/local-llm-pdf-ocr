# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync                                                       # install/sync deps
uv run main.py input.pdf [output.pdf]                         # CLI OCR (DP align + crop-refine, auto dense-mode)
uv run main.py input.pdf --pages 1-3,5 --dpi 300              # page range + DPI
uv run main.py input.pdf --concurrency 3                      # parallel LLM calls
uv run main.py input.pdf --no-refine                          # skip crop re-OCR (faster, less robust)
uv run main.py input.pdf --max-image-dim 640                  # smaller VLM context (needed for GLM-OCR:1.1B)
uv run main.py input.pdf --dense-mode always --concurrency 5  # force per-box OCR every page (handwriting / dense)
uv run main.py input.pdf --dense-threshold 40                 # auto-mode kicks in earlier (default 60 boxes/page)
uv run main.py input.pdf --api-base http://localhost:11434/v1 --model glm-ocr:latest --max-image-dim 640   # Ollama + GLM-OCR
uv run main.py input.pdf --grounded --model qwen/qwen3-vl-8b  # grounded path: bbox-native VLM, no Surya
uv run main.py photo.avif                                     # AVIF input (pillow-avif-plugin registered eagerly)
uv run main.py input.pdf -v                                   # verbose debug logging
uv run uvicorn server:app --reload --port 8000                # web UI at http://localhost:8000
```

Debug/inspection tools live in `scripts/` (`visualize_bboxes.py`, `debug_alignment.py`, `verify_output.py`, `inspect_pdf.py`, etc.) and are run with `uv run scripts/<name>.py <args>`.

### Tests

```bash
uv run pytest                     # full suite (~35s, loads Surya once; 167 tests)
uv run pytest -m "not slow"       # fast tier only, no model load (~7s; 145 tests)
uv run pytest -m slow             # integration: real Surya + example PDFs (~30s; 22 tests)
uv run pytest tests/test_aligner.py -v   # single file
```

Tests live under `tests/`. `pytest-asyncio` is in auto mode so `async def test_...` works without decorators. Markers: `slow` (loads Surya), `live_llm` (hits a real LLM endpoint — not wired yet). Integration tests use the on-disk `examples/*.pdf` with a stubbed LLM to validate the full detect→align→embed path end-to-end.

### Confidence evaluation

`scripts/confidence_eval.py` scores either pipeline path against the ground-truth fixtures in `tests/fixtures/ground_truth_*.json`. Requires a live LLM; reports per-document block recall, average IoU of matched pairs, and average text similarity against the GT content.

```bash
uv run scripts/confidence_eval.py --path grounded --grounded-model qwen/qwen3-vl-8b
uv run scripts/confidence_eval.py --path hybrid --hybrid-model allenai/olmocr-2-7b
uv run scripts/confidence_eval.py --path both    # both, side by side
```

Fixture axis-order is auto-detected (`src/pdf_ocr/evaluation.py::_detect_bbox_axis_order`) since `handwritten.pdf`'s fixture uses `[x0,y0,x1,y1]` while `hybrid.pdf` and `digital.pdf` fixtures use `[y0,x0,y1,x1]`. Normalization uses each fixture's declared page dimensions.

## Configuration

LLM endpoint is read from `.env` (or CLI overrides `--api-base` / `--model`). LM Studio must be running locally and serving the vision model.

```
LLM_API_BASE=http://localhost:1234/v1
LLM_MODEL=allenai/olmocr-2-7b
OCR_CONCURRENCY=3          # server.py — async LLM concurrency
```

## Architecture

Hybrid pipeline: **Surya** does fast layout *detection only* (no recognition), a **local vision LLM** (OlmOCR via LM Studio) produces text content, and a **Needleman-Wunsch DP** binds LLM lines to detected boxes. Boxes the DP cannot confidently populate are re-OCR'd individually via per-box image crops. **Dense pages** (>`dense_threshold` detected boxes, default 60) skip the full-page LLM call entirely and OCR each Surya box individually — bypasses the loop / hallucination failure modes that full-page OCR exhibits on dense handwritten content. PyMuPDF writes a "sandwich" PDF (rasterized page image + invisible selectable text).

```
                ┌── (sparse) ── LLM full-page OCR ── DP line→box alignment ── crop re-OCR for gaps ──┐
PDF → images ──┤                                                                                       ├── output_writer → searchable PDF
                └── (dense)  ── per-box OCR (each Surya box → one LLM crop call) ──────────────────────┘
```

### Alignment algorithm (`HybridAligner.align_text`)

Both sequences (LLM lines, Surya boxes) arrive in reading order, so alignment is a **monotonic** Needleman-Wunsch DP over `(N lines, M boxes)`:

- **Match cost**: relative character-count mismatch between the LLM line's length and each box's estimated capacity. Capacity is proportional to box area (width × height in normalized space), with total capacity scaled to total LLM chars — so what matters is each box's *share* of the page's text.
- **Skip ops**: `skip_line` (line unmatched; attached to the nearest matched box to preserve searchability) and `skip_box` (box left empty; cheap — many detected boxes are rules/decorations).
- **Output**: one `(box, text)` tuple per input box, in input order. Empty strings flag boxes the DP could not match — those are the refine-stage candidates.

### Refine stage (per-box crop re-OCR)

After DP, any empty box passing `_is_refinable` (~40pt × 6pt at 200 DPI) is cropped from the page image, upscaled to ≥256 px, and sent to the LLM with a minimal "transcribe only this text" prompt. Before the LLM call, `is_blank_crop` (`utils/image.py`) checks pixel stddev — notebook dot grids and other low-variance regions short-circuit because OlmOCR-2 hallucinates the canonical pangram ("The quick brown fox...") on blank input. The crop response is also filtered against `_HALLUCINATION_PATTERNS` so any pangram or "lorem ipsum" that slips through is dropped. This catches tables, multi-column layouts, and figure captions while staying cheap on clean prose. Disable refine entirely with `--no-refine`.

### Dense-page detection / per-box OCR

When `dense_mode` is `"auto"` (default) and a page exceeds `dense_threshold` boxes (default 60), `OCRPipeline._ocr_per_box` skips full-page OCR for that page and feeds each Surya box individually through `perform_ocr_on_crop`. Refine is also skipped for those pages — every box was already individually transcribed. `dense_mode="always"` forces this on every page (recommended for handwriting); `"never"` keeps the original full-page path. This bypasses the failure modes that full-page OCR exhibits on dense content (the model loops on `<th>Java</th>`-style runaway, or just fails to track the volume of content). Cost: N times more LLM calls per dense page, mitigated by `--concurrency`.

### LLM stability defenses (`OCRProcessor`)

Local VLMs occasionally fall into runaway-generation loops on dense or unusual pages. Without bounds the openai client default-waits 600s before erroring, so the pipeline appears to hang. Three layered defenses:

- **Token / time caps**: page calls are capped at `PAGE_MAX_TOKENS=6144` / `PAGE_TIMEOUT_S=240`; crop calls at `CROP_MAX_TOKENS=256` / `CROP_TIMEOUT_S=60`. Set per-call via `client.with_options(timeout=...)` plus `max_tokens=...`.
- **Runaway-repetition stripper** (`_strip_runaway_repetition`): caps any single response line at 20 occurrences. Admits legitimate repeated structure (HTML table rows, separators) while clipping "(6) result" × 869-style loops. Logs a warning naming the worst offender so the user knows to lower `--max-image-dim` or swap `--model`.
- **Pangram / placeholder filter**: crop responses containing `_HALLUCINATION_PATTERNS` ("the quick brown fox jumps over the lazy dog", "lorem ipsum") are dropped to empty strings. Defense in depth for crops that pass the blank check but still confuse the model.

`_chat` wraps any underlying exception in `LLMCallError` with a message naming the api-base + model so connection / model-not-loaded failures are diagnosable from the CLI output without a full stack trace.

### OlmOCR prompt / YAML parsing

`OCRProcessor.perform_ocr` uses the canonical `build_no_anchoring_v4_yaml_prompt()` string from `olmocr.prompts` (the model is RL-trained on that exact prompt). Responses are markdown with YAML front matter (`primary_language`, `is_rotation_valid`, ...); `_strip_yaml_front_matter` removes the front matter before returning the body. `perform_ocr_on_crop` uses a distinct minimal prompt since per-region metadata is nonsensical.

### The orchestration seam: `src/pdf_ocr/pipeline.py`

`OCRPipeline` is the single shared driver used by both `main.py` and `server.py`. Two execution paths:

- **Hybrid** (default): `convert → detect → ocr → refine → embed` — Surya gives boxes, LLM transcribes whole page, DP binds lines to boxes, crop re-OCR fills gaps. Per-page strategy: pages with ≤ `dense_threshold` boxes use full-page OCR; pages above the threshold (or all pages when `dense_mode="always"`) skip full-page and run per-box OCR via `_ocr_per_box`. Refine is only invoked for pages that took the full-page route.
- **Grounded** (`grounded_backend=...`): `grounded → embed` — a bbox-native VLM (Qwen2.5-VL, Qwen3-VL, etc.) returns `(bbox, text)` pairs directly. Skips Surya, DP, and refine entirely. `src/pdf_ocr/core/grounded.py::PromptedGroundedOCR` is the default implementation; it rasterizes pages, calls the VLM with `DEFAULT_GROUNDING_PROMPT` (which explicitly demands one element per visual line so wrapped phrases stay separated), parses the JSON, normalizes pixel bboxes to 0..1. Honors `LLM_API_BASE` / `LLM_MODEL` from `.env` like `OCRProcessor` does.

Both paths take the same **injected components** so extensions can swap any phase without touching the CLI or web handlers:

| Parameter       | Contract                                                               | Default                               |
|-----------------|------------------------------------------------------------------------|---------------------------------------|
| `aligner`       | `get_detected_boxes_batch(list[bytes])` + `align_text(structured, text)` | `HybridAligner` (Surya-only)          |
| `ocr_processor` | async `perform_ocr(image_base64) -> list[str]`                         | `OCRProcessor` (OpenAI-compat LLM)    |
| `pdf_handler`   | `convert_to_images(path, dpi) -> dict[int, b64]`                       | `PDFHandler`                          |
| `output_writer` | `callable(input_path, output_path, pages_data, dpi) -> None`           | `pdf_handler.embed_structured_text`   |

`OCRPipeline.run(...)` is async, uses `asyncio.as_completed` + a `Semaphore(concurrency)` for per-page LLM work (so `concurrency=1` is the degenerate sequential case), and drives an optional progress callback:

```python
async def progress(stage: str, current: int, total: int, message: str) -> None
# stages: "convert" | "detect" | "ocr" | "refine" | "embed"
```

The `"ocr"` stage label is suffixed with a dense/sparse split when both kinds of pages exist on a run (`OCR (3 dense / 17 sparse)`) so the user can see which pages took the per-box path. `main.py` maps this callback onto Rich progress tasks; `server.py` maps it onto WebSocket percent updates via `_STAGE_WEIGHTS`.

### Core classes (`src/pdf_ocr/core/`)

| Class           | File        | Role                                                                          |
|-----------------|-------------|-------------------------------------------------------------------------------|
| `PDFHandler`    | `pdf.py`    | PDF↔image conversion; builds a fresh `new_doc` and overlays invisible text with `render_mode=3`. `_draw_invisible_text` auto-sizes font per box. Eagerly imports `pillow_avif` to register `.avif` decoding with PIL — `IMAGE_EXTENSIONS` includes `.avif` alongside JPEG/PNG/TIFF/BMP/WebP. |
| `OCRProcessor`  | `ocr.py`    | `AsyncOpenAI` client against the local LLM; `perform_ocr` returns a list of lines. Per-call `max_tokens` + `timeout` caps prevent runaway generation. Output runs through `_strip_runaway_repetition` (caps any single line at 20 occurrences) and crop responses through the pangram filter. |
| `HybridAligner` | `aligner.py`| Wraps Surya's `DetectionPredictor`; `get_detected_boxes_batch` returns boxes in row-major order; `align_text` runs the DP twice (row-major + column-major from `_reading_order_indices`) and picks the lower-cost result, so the same code path matches whichever order the LLM emits. |

### Coordinate and text conventions
- Bounding boxes are normalized `[nx0, ny0, nx1, ny1]` in `0..1` and only scaled to PDF points inside `embed_structured_text`. Don't scale them anywhere else.
- `get_detected_boxes_batch` returns boxes in stable **row-major** order (top-to-bottom, left-to-right) — a deterministic default for visualization and downstream tools.
- `align_text` is **model-agnostic**: it runs the DP twice — once with row-major boxes and once with column-major (via `_reading_order_indices`) — and picks the lower-cost result. VLMs disagree on emission order (OlmOCR-2 → column-major on multi-column pages; some others → row-major), and OlmOCR's prompt is RL-locked so we can't normalize via prompt. The DP cost itself is the signal for which ordering matches the LLM's emission.
- If `align_text` receives zero detected boxes, it falls back to a single full-page box containing all LLM text so search still works.
- In `embed_structured_text`, multi-line text in a box is treated as a full-page fallback block; single-line text is placed with `insert_text(point, render_mode=3)` (PyMuPDF maintainer-recommended for invisible OCR layers — `insert_textbox` mis-sizes single-line glyphs).

### Surya progress-bar silencing
`src/pdf_ocr/utils/tqdm_patch.py` is applied at the top of `aligner.py` (`tqdm_patch.apply()`) to stop Surya's internal tqdm bars from colliding with Rich. Do not remove this import — it must run before `from surya.detection import DetectionPredictor`. `main.py` also sets `TQDM_DISABLE=1` via `os.environ.setdefault` before the lazy import path loads Surya.

### Entry points

- **`main.py`** — argparse + lazy imports of heavy modules (Surya, PyMuPDF) so `--help` stays fast; wires a Rich progress adapter into `OCRPipeline`.
- **`server.py`** — FastAPI with `POST /process`, `GET /text/{job_id}`, `WS /ws/{client_id}`. The WebSocket is for live progress; the pipeline callback is translated to a single 0-100 percent via `_STAGE_WEIGHTS`.

### Extension points

Common forks / extensions map cleanly onto the injection points above:

- **Alternative layout model** (e.g. DETR): implement an aligner exposing `get_detected_boxes_batch` + `align_text` and pass it to `OCRPipeline`.
- **Alternative output format** (e.g. EPUB): write a function `writer(input_path, output_path, pages_data, dpi)` and pass it as `output_writer=`.
- **Different OCR backend**: implement `perform_ocr(image_base64) -> list[str]` and pass as `ocr_processor=`.

No entry-point edits required in any of these cases.
