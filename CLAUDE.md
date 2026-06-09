# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync                                                       # install/sync deps
uv run local-llm-pdf-ocr input.pdf [output.pdf]                         # CLI OCR (DP align + crop-refine, auto dense-mode)
uv run local-llm-pdf-ocr input.pdf --pages 1-3,5 --dpi 300              # page range + DPI
uv run local-llm-pdf-ocr input.pdf --concurrency 3                      # parallel LLM calls
uv run local-llm-pdf-ocr input.pdf --no-refine                          # skip crop re-OCR (faster, less robust)
uv run local-llm-pdf-ocr input.pdf --max-image-dim 640                  # smaller VLM context (needed for GLM-OCR:1.1B)
uv run local-llm-pdf-ocr input.pdf --dense-mode always --concurrency 5  # force per-box OCR every page (handwriting / dense)
uv run local-llm-pdf-ocr input.pdf --dense-threshold 40                 # auto-mode kicks in earlier (default 60 boxes/page)
uv run local-llm-pdf-ocr input.pdf --api-base http://localhost:11434/v1 --model glm-ocr:latest --max-image-dim 640   # Ollama + GLM-OCR
uv run local-llm-pdf-ocr input.pdf --grounded --model qwen/qwen3-vl-8b  # grounded path: bbox-native VLM, no Surya
uv run local-llm-pdf-ocr photo.avif                                     # AVIF input (native via Pillow ≥11.3)
uv run local-llm-pdf-ocr input.pdf --no-verify-model                    # skip pre-flight model check (Ollama / non-/v1/models servers)
uv run local-llm-pdf-ocr input.pdf --format html                        # HTML output with external sidecar JPEGs (auto-named input_ocr.html + input_ocr_p<N>.jpg)
uv run local-llm-pdf-ocr input.pdf --format html --html-inline-images   # single self-contained HTML (page images as base64 data: URLs)
uv run local-llm-pdf-ocr input.pdf --format html --html-mode letter-spacing  # opt back into the original stretched-glyph overlay mode
uv run local-llm-pdf-ocr input.pdf --format html --html-invert-dark       # HTML with dark-mode page inversion (opt-in)
uv run local-llm-pdf-ocr input.pdf out.md                               # Markdown output (extension wins over --format)
uv run local-llm-pdf-ocr input.pdf -v                                   # verbose debug logging
uv run uvicorn pdf_ocr.server:app --reload --port 8000        # web UI at http://localhost:8000
```

Debug/inspection tools live in `scripts/` (`visualize_bboxes.py`, `debug_alignment.py`, `verify_output.py`, `inspect_pdf.py`, etc.) and are run with `uv run scripts/<name>.py <args>`.

### Tests

```bash
uv run pytest                     # full suite (~60s, loads Surya once; 359 fast + 29 slow)
uv run pytest -m "not slow"       # fast tier only (~25s; 359 tests)
uv run pytest -m slow             # integration: real Surya + example PDFs + geometry floors (~30s; 29 tests)
uv run pytest tests/test_aligner.py -v   # single file
```

Tests live under `tests/`. `pytest-asyncio` is in auto mode so `async def test_...` works without decorators. Markers: `slow` (loads Surya), `live_llm` (hits a real LLM endpoint — not wired yet). Integration tests use the on-disk `examples/*.pdf` with a stubbed LLM to validate the full detect→align→embed path end-to-end.

### Confidence evaluation

`scripts/confidence_eval.py` scores either pipeline path against the ground-truth fixtures in `tests/fixtures/ground_truth_*.json`. Requires a live LLM. Metrics are decomposed per axis (`ConfidenceReport.to_metrics_dict()`): geometry — block recall/precision/hmean at the IoU threshold via optimal Hungarian matching (`--matching greedy` reproduces legacy reports), recall@0.5, matched-IoU average; text — per-match CER (jiwer) plus assignment-free bag-of-words F1; structure — split/merge-tolerant pseudo-character coverage (CLEval-style PCC, in-repo); plus binary per-document checks from `evals/checks/<stem>.jsonl` (present/absent/order facts, olmOCR-bench style). Each run appends `evals/history.csv` and compares against committed `evals/baselines/<doc>__<path>.json` (tolerance 0.02 on deterministic geometry/checks axes, 0.04 on the LLM-sampled text axes; regressions exit 1); `--update-baselines` ratchets them in a reviewed commit. Detector-only geometry floors run offline in `tests/test_eval_regression.py` (slow tier).

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
OCR_CONCURRENCY=2          # server.py — in-flight LLM requests (KV-cache VRAM on parallel-slot servers; queued for free on LM Studio/Ollama defaults)
OCR_VERIFY_MODEL=0         # server.py — opt out of pre-flight model check (default: on)
```

### Pre-flight model verification (issue #7)

Both entry points hit `GET /v1/models` before any OCR work to confirm the configured model is loaded, raising `ModelNotLoadedError` (subclass of `LLMCallError`) on mismatch. LM Studio otherwise silently falls back to whatever model is loaded, producing subtly wrong OCR with no error to flag the mismatch. Implementation lives on the backends — `OCRProcessor.ensure_model_loaded()` and `PromptedGroundedOCR.ensure_model_loaded()` — both reusing the same `_list_loaded_model_ids` / `_format_model_not_loaded` helpers in `core/ocr.py`. Disable per-call with CLI `--no-verify-model` or env `OCR_VERIFY_MODEL=0` (server only).

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

When `dense_mode` is `"auto"` (default) and a page exceeds `dense_threshold` boxes (default 60), `OCRPipeline._ocr_per_box` skips full-page OCR for that page and feeds each Surya box individually through `perform_ocr_on_crop`. Refine is also skipped for those pages — every box was already individually transcribed. `dense_mode="always"` forces this on every page (recommended for handwriting); `"never"` keeps the original full-page path. In `"auto"`, a sparse page is also **retried per-box** when the DP's real match count lands under half its boxes (`pipeline.py::_DP_RETRY_*`) — the aligner reports it via `AlignmentResult.match_count`. Forms are the canonical trigger: many label boxes with near-identical DP match costs bind text to the wrong fields, and the retry replaces guessed bindings with per-region transcriptions. This bypasses the failure modes that full-page OCR exhibits on dense content (the model loops on `<th>Java</th>`-style runaway, or just fails to track the volume of content). Cost: N times more LLM calls per dense page, mitigated by `--concurrency`.

### Photo preprocessing (`utils/preprocess.py`)

Hybrid-path pages that look like camera captures (convex page outline covering 18–97% of the frame, perspective deviation ≥1.5% of the diagonal) are perspective-rectified + illumination-flattened **for detection/OCR only**; boxes are mapped back through the inverse homography right before the writers run, arriving as `OrientedBox` quads in original-photo space. Writers always rasterize the original input, so only coordinates need mapping. Flat scans / rasterized PDF pages find no page quad and pass through byte-identical. CLI `--preprocess auto|always|never` (default auto); grounded path unaffected (its backend rasterizes internally).

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
| `output_writer` | `callable(input_path, output_path, pages_data, dpi) -> None`           | resolved from output extension via `pdf_ocr.output.resolve_output_writer` (`.pdf` → `PDFHandler`, `.html`/`.htm` → `HTMLHandler`, `.md`/`.markdown` → `MarkdownHandler`); falls back to `pdf_handler.embed_structured_text` |

`OCRPipeline.run(...)` is async, uses `asyncio.as_completed` + a `Semaphore(concurrency)` for per-page LLM work (so `concurrency=1` is the degenerate sequential case), and drives an optional progress callback:

```python
async def progress(stage: str, current: int, total: int, message: str) -> None
# stages: "convert" | "detect" | "ocr" | "refine" | "embed"
```

The `"ocr"` stage label is suffixed with a dense/sparse split when both kinds of pages exist on a run (`OCR (3 dense / 17 sparse)`) so the user can see which pages took the per-box path. `main.py` maps this callback onto Rich progress tasks; `server.py` maps it onto WebSocket percent updates via `_STAGE_WEIGHTS`.

### Core classes (`src/pdf_ocr/core/`)

| Class           | File        | Role                                                                          |
|-----------------|-------------|-------------------------------------------------------------------------------|
| `PDFHandler`    | `pdf.py`    | PDF↔image conversion; builds a fresh `new_doc` and overlays invisible text with `render_mode=3`. `_draw_invisible_text` auto-sizes font per box. `IMAGE_EXTENSIONS` includes `.avif` alongside JPEG/PNG/TIFF/BMP/WebP — AVIF decoding is native to Pillow ≥11.3 (the pyproject.toml floor). |
| `HTMLHandler`   | `html.py`   | HTML output with OCR text overlaid as absolutely-positioned invisible `<span>`s. By default page images are referenced as external files (the input file itself for single-frame browser-native images JPEG/PNG/WebP/AVIF/GIF; sidecar JPEGs `<output_stem>_p<N>.jpg` — zero-padded page numbers, plain `<output_stem>.jpg` for single-page inputs — for PDFs and multi-frame inputs). `inline_images=True` (CLI: `--html-inline-images`) inlines page images as base64 data URLs for a single self-contained file. `invert_dark=True` (CLI: `--html-invert-dark`) adds CSS `filter: invert() hue-rotate(180deg)` that activates under `prefers-color-scheme: dark`, inverting scanned white-background pages for night reading. Default sizing mode is `scaled` (server-side fit + page-load `canvas.measureText` script setting per-span `--scale-x`, the PDF.js textLayer approach; degrades to the server-side fit without JS); `letter-spacing` and `full-height` are still available via `mode=` / `--html-mode`. Tilted detector quads (`OrientedBox.is_rotated`) anchor spans at the quad's top-left with `--rotate`, swung by the stylesheet's `transform: rotate(var(--rotate)) scaleX(var(--scale-x))`. Supports both PDF and image inputs. |
| `MarkdownHandler` | `markdown.py` | Plain Markdown export. `# OCR output: <input>` top-level header, `## Page N` per page, one block per non-empty box in reading order. No bbox math, no rasterization. |
| `OCRProcessor`  | `ocr.py`    | `AsyncOpenAI` client against the local LLM; `perform_ocr` returns a list of lines. Per-call `max_tokens` + `timeout` caps prevent runaway generation. Output runs through `_strip_runaway_repetition` (caps any single line at 20 occurrences) and crop responses through the pangram filter. |
| `HybridAligner` | `aligner.py`| Wraps Surya's `DetectionPredictor`; `get_detected_boxes_batch` returns `OrientedBox`es (list-compatible AABBs carrying Surya's min-area-rect `quad`, pixel-space `angle`, and per-page-normalized `confidence`) in row-major order; `align_text` runs the DP twice (row-major + column-major from `_reading_order_indices`) and picks the lower-cost result, returning an `AlignmentResult` (list + `match_count` diagnostic for the pipeline's per-box retry). |
| `OrientedBox` etc. | `geometry.py` | Shared oriented-geometry types: `OrientedBox(list)` (AABB + quad/angle/confidence, JSON-serializes as a plain array), `quad_angle_deg` / `quad_edge_lengths` (pixel-space math — normalized space distorts angles/lengths on non-square pages), and `ANGLE_FLATTEN_DEG` (writers treat tilt below it as horizontal; min-area rects wobble 1-3° on straight printed text). |
| `_layout` helpers | `_layout.py` | Pure helpers shared by every output writer: `is_full_page_fallback` (detect aligner's `[0,0,1,1]+\n` fallback bbox) and `split_multi_line_bbox` (proportional vertical split for `\n`-joined lines in one bbox). PDF and HTML writers call them so edge-case handling stays identical across formats. |

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
- **Alternative output format** (e.g. EPUB): write a function `writer(input_path, output_path, pages_data, dpi)` and pass it as `output_writer=`. The CLI / server use `pdf_ocr.output.resolve_output_writer` for extension-based dispatch — extend it to plug a new format into the standard format table (`SUPPORTED_FORMATS`).
- **Different OCR backend**: implement `perform_ocr(image_base64) -> list[str]` and pass as `ocr_processor=`.

No entry-point edits required in any of these cases.
