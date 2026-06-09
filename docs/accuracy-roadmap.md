# Geometry & Accuracy Roadmap — toward a de-facto-standard OCR text overlay

*Research synthesis, 2026-06-09. Four parallel research passes (detection geometry, embedding mechanics, font-metric congruence, metrics/benchmarks) + live baseline measurement. Every recommendation maps to an existing injection point (`aligner`, `output_writer`, `evaluation.py`) — none requires entry-point changes.*

---

## 1. Where the pipeline stands (measured 2026-06-09, live `allenai/olmocr-2-7b`)

`uv run scripts/confidence_eval.py --path hybrid` — block recall @ IoU≥0.3, greedy matching:

| Document | GT blocks | Recall | Avg IoU | Text sim | Path |
|---|---|---|---|---|---|
| digital.pdf (form) | 16 | 0.44 | 0.68 | 0.70 | full-page + DP |
| hybrid.pdf (form) | 38 | **0.24** | 0.54 | 0.62 | full-page + DP |
| handwritten.pdf | 14 | 0.93 | 0.59 | 0.71 | full-page + DP |
| dense.pdf | 365 | **0.99** | 0.92 | 0.83 | per-box |
| notes.pdf | 477 | 0.93 | 0.93 | 0.71 | per-box |

Two facts drive the whole roadmap:

1. **The per-box path is already near-ceiling** (0.93–0.99 recall, 0.92–0.93 IoU). The DP path craters specifically on **forms** — unmatched GT blocks are `Name: ____`-style label/fill-in pairs. Part of that 0.24 is a *metric artifact* (line-vs-block granularity; greedy matching), part is real DP misbinding.
2. **Rotation/skew support is latent in the installed dependencies and currently discarded:**
   - Surya computes a true rotated min-area quad per line (`cv2.minAreaRect`, CRAFT-lineage) plus a confidence — `aligner.py:56` flattens to the AABB envelope and drops both. ([surya/detection/heatmap.py](https://github.com/datalab-to/surya/blob/master/surya/detection/heatmap.py))
   - PyMuPDF's `morph=(pivot, Matrix)` supports arbitrary rotation/shear composed with invisible `render_mode=3` — and `pdf.py:_draw_invisible_text` *already uses morph* for width scaling; rotation is one matrix multiplication away. ([PyMuPDF Shape docs](https://pymupdf.readthedocs.io/en/latest/shape.html), [maintainer example](https://github.com/pymupdf/PyMuPDF/discussions/2722))

Current congruence state: the **PDF writer already implements the industry pattern at line level** — Helvetica metrics (`Font.text_length`), fontsize = box-height / (ascender−descender), baseline anchored at `y1 + descender·fontsize`, per-line morph scale-x. The **HTML writer is the laggard**: monospace 0.6-aspect heuristic, no measurement, no rotation.

---

## 2. The target model ("synonymous formatting")

Every production-grade implementation converges on the same triple pattern (Tesseract `pdfrenderer.cpp`, OCRmyPDF v17 `fpdf2` renderer, PDF.js textLayer):

1. **One invisible font with known metrics**; fontsize from line height × ascent/(ascent+descent) (≈0.78 Helvetica; 0.8 is PDF.js's no-metrics fallback).
2. **Text anchored at the baseline**, not bbox-fit — selection rectangles in viewers are reconstructed from font metrics × text matrix around the baseline pen position; near-horizontal baselines deliberately flattened (Tesseract `ClipBaseline`: "viewers like evince get really confused during copy-paste when the baseline wanders").
3. **Per-WORD horizontal scaling** so each word's advance width exactly equals its detected width (PDF `Tz` operator / PyMuPDF morph scale-x / CSS `transform: scaleX`). Rotation lives in the text matrix (`Tm`/`cm`/CSS `rotate()`); width in the scale — they compose cleanly.

Sources: [tesseract pdfrenderer.cpp](https://github.com/tesseract-ocr/tesseract/blob/main/src/api/pdfrenderer.cpp) · [OCRmyPDF fpdf renderer](https://github.com/ocrmypdf/OCRmyPDF/blob/main/src/ocrmypdf/fpdf_renderer/renderer.py) · [pdf.js text_layer.js](https://github.com/mozilla/pdf.js/blob/master/src/display/text_layer.js)

---

## 3. Ranked roadmap

### P0 — Upgrade the eval before touching the pipeline (prerequisite for "strive toward improvement")

*Injection point: `src/pdf_ocr/evaluation.py`, `scripts/confidence_eval.py`, CI.*

| Change | Why | Proof metric |
|---|---|---|
| Hungarian matching (`scipy.optimize.linear_sum_assignment` on −IoU) replacing greedy | Deterministic, order-independent, optimal; greedy under-matches | recall/precision/hmean at IoU 0.5 **and** 0.3 |
| CER via `jiwer` on matched pairs (replacing `difflib` ratio) | Standardized, alignment-inspectable | CER per doc |
| Page-level order-independent **bag-of-words F1** (overlay text vs GT text, no boxes) | Decouples transcription quality from box binding — DP can misbind while text is right | BoW-F1 |
| **CLEval** (`pip install cleval`) for char-level det/E2E with split/merge tolerance | Directly absorbs the Surya-line vs GT-block granularity mismatch that depresses form recall to 0.24 | CLEval det/E2E hmean |
| olmOCR-bench-style **binary unit tests** per fixture (JSONL: `present` / `absent` / `order` rules) | Attributable pass/fail beats fuzzy aggregates; one test per form label and per filled value | pass fraction per doc |
| Mean matched IoU kept as the **tightness** signal (TIoU later if needed) | Tracks box-quality improvements separately from recall | mean IoU |

**CI ratchet:** commit `evals/baselines/<doc>.json`; `pytest-regressions` `num_regression` with abs tolerance 0.02; nightly live-LLM run, stub-LLM on PRs; baselines move only via explicit `--force-regen` commits reviewed like code; append history to `evals/history.csv`. Report per-document per-axis, never one blended score — each improvement must be attributable.
Sources: [CLEval](https://github.com/clovaai/CLEval) · [jiwer](https://github.com/jitsi/jiwer) · [olmOCR-bench design](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/overview.html) · [OmniDocBench per-axis pattern](https://github.com/opendatalab/OmniDocBench)

*Effort: S–M. Expected effect: form-doc scores rise on CLEval/BoW axes immediately (metric truthfulness), giving honest attribution for everything below.*

### P1 — Consume Surya's polygons + confidence (free geometry upgrade)

*Injection point: `HybridAligner.get_detected_boxes_batch` + `pages_data` tuple shape.*

Stop flattening: carry `(aabb, quad, angle, confidence)` per box — angle via `atan2` on the quad's top edge. Backward-compatible if writers read positionally with optional extras. Tightness experiments: `DETECTOR_BOX_Y_EXPAND_MARGIN=0` (default pads 5% vertically each way) and raised `DETECTOR_TEXT_THRESHOLD`.
*Expected impact: matched IoU ↑ on all docs (the 5% pad alone inflates every box); enables P2. Proof: mean matched IoU; later baseline-angle error.*
Source: [heatmap.py](https://github.com/datalab-to/surya/blob/master/surya/detection/heatmap.py) (`DETECTOR_BOX_Y_EXPAND_MARGIN` in [settings.py](https://github.com/datalab-to/surya/blob/master/surya/settings.py)). Note: Surya confidence is per-page max-normalized — a ranking signal, not a probability.

### P2 — Rotated invisible text in both writers (S effort, latent in deps)

*Injection points: `PDFHandler._draw_invisible_text`, `HTMLHandler._emit_span`.*

- **PDF**: compose rotation into the existing morph: `morph = (baseline, fitz.Matrix(scale_x, 1.0) * fitz.Matrix(angle_deg))`. Flatten near-horizontal angles (|angle| < ~2°) to 0 like Tesseract's `ClipBaseline` — selection UX over micro-fidelity.
- **HTML**: PDF.js pattern — keep `%` positions and `transform-origin: 0% 0%`, add `transform: rotate(var(--rotate)) ...` per span. Browser selection/hit-testing follows transformed glyphs (production-proven by Firefox's built-in viewer).
*Expected impact: rotated/skewed scans (currently embedded as inflated horizontal AABBs) get correctly-angled selection. Proof: new rotated-fixture unit tests + baseline-angle error metric.*
Sources: [morph example](https://github.com/pymupdf/PyMuPDF/discussions/2722) · [text_layer_builder.css](https://github.com/mozilla/pdf.js/blob/master/web/text_layer_builder.css) · [OCRmyPDF baseline-matrix composition](https://github.com/ocrmypdf/OCRmyPDF/blob/main/src/ocrmypdf/fpdf_renderer/renderer.py)

### P3 — HTML width congruence: measured `scaleX`, retire the 0.6 heuristic

*Injection point: `HTMLHandler._span_sizing_style` + the existing inline-script pattern.*

Replace the monospace-aspect math in `scaled` mode with the PDF.js approach: a small inline script (sibling to the existing fit-to-viewport script) measures each span via `canvas.measureText` in the actual rendered font and sets `--scale-x = target_width / measured`. Client-side measurement is mandatory — server-side font metrics can't know the browser's font fallback.
Keep `letter-spacing` mode only as legacy: shipping browsers add a trailing letter-space after the last character (so width-matching systematically overshoots by one unit), it disables ligatures, and it breaks Arabic-script joining. `scaleX` is paint-only and exact by construction.
*Expected impact: right-edge selection drift on proportional text eliminated in HTML (currently ±30%-class error on pathological lines: "ill" vs "WWW"). Proof: rendered-overlay edge-offset check (Playwright measure of selection rects vs box rects) + visual QA screenshots.*
Sources: [pdf.js measureText/scaleX](https://github.com/mozilla/pdf.js/blob/master/src/display/text_layer.js) · [letter-spacing trailing-gap CSSWG issue](https://github.com/w3c/csswg-drafts/issues/1518) · [MDN letter-spacing](https://developer.mozilla.org/en-US/docs/Web/CSS/letter-spacing)

### P4 — Per-word placement (PDF first), word-split of line text

*Injection points: `PDFHandler._draw_invisible_text`; later `_emit_span`.*

`pdf.py` fits width per LINE — interior glyph positions still drift within the line (error zeroes only at line edges). Port the Tesseract/OCRmyPDF refinement: split line text into words, allocate horizontal extent proportionally (or from per-word detector boxes when available), one `insert_text` + morph per word, scaled spaces between (OCRmyPDF v17.4.0's regression shows the inter-word-gap trap — scale the space, don't stretch words to swallow gaps). This also retires the crude multi-line-split heuristic at `pdf.py:257` once detector granularity improves.
*Expected impact: word-level selection/highlight congruence in PDF (the unit users actually select). Proof: PDFium `FPDFText_GetLooseCharBox` extraction vs detector boxes — mean per-word horizontal offset.*
Sources: [pdfrenderer.cpp Tz math](https://github.com/tesseract-ocr/tesseract/blob/main/src/api/pdfrenderer.cpp) · [OCRmyPDF v17 notes](https://github.com/ocrmypdf/OCRmyPDF/blob/main/docs/releasenotes/version17.md)

### P5 — Smarter dense-mode default (forms fix with zero new deps)

*Injection point: `OCRPipeline` dense heuristic.*

The eval proves per-box mode is the quality ceiling (0.99/0.92 on dense.pdf) and forms fail on the DP path while sitting *under* the 60-box dense threshold (digital: 30 boxes, hybrid: 15). Box count alone is the wrong trigger. Add a second signal — e.g. median-box-width/page-width (forms = many short label boxes) or label-pattern density — and route form-like pages per-box. Quantify the cost/quality trade (N× LLM calls) on the new eval axes before changing the default; the data says the recall gain is ~0.24→0.9+.
*Proof: form-doc recall + CLEval E2E before/after; LLM-call count as the cost axis.*

### P6 — Longer horizon: granularity and backends

- **docTR** detection (`assume_straight_pages=False`): word-level rotated polygons, torch-native, Apache-2.0, relative 0..1 coords matching this repo's convention — the cleanest external aligner swap if Surya line-granularity becomes the binding constraint. ([mindee/doctr](https://github.com/mindee/doctr))
- **Kraken** blla segmenter: baseline polylines + boundary polygons — purpose-built for the handwriting/dense-notebook case; torch ~=2.4 pin is the friction. ([kraken](https://github.com/mittagessen/kraken))
- **Florence-2** `<OCR_WITH_REGION>`: the only small VLM emitting true 4-point quads — natural alternative backend for the `--grounded` path (MIT, sub-1B). ([HF card](https://huggingface.co/microsoft/Florence-2-large))
- **External benchmarks** once internal axes are stable: FUNSD test split + NAF (hand-filled historical forms — exactly the weak spot), olmOCR-bench subsets (ODC-BY; pairs with the olmocr model already used), OmniDocBench for attribute-stratified breadth. Skip scene-text sets (IC15/Total-Text/CTW1500) — wrong domain.

---

## 4. Dead ends (researched, rejected — don't re-litigate)

- **CRAFT** (upstream dead since 2019), **EAST** (GPL-3.0, obsolete), **MMOCR** (mmcv pin vs modern torch, dormant since 2023-07) as detector swaps.
- **Qwen-style VLM grounding for geometry**: AABB-only (no quads) — strictly worse than the Surya quads already computed. Watch-item: **Qwen3-VL switched bbox grounding to normalized 0–1000 coordinates** (Qwen2.5-VL was absolute pixels) — `PromptedGroundedOCR`'s pixel-coord parsing breaks silently if users point `--grounded` at Qwen3-VL with a 2.5-era assumption. Worth a defensive heuristic or doc note now. ([Qwen3-VL issue](https://github.com/QwenLM/Qwen3-VL/issues/1486))
- **Glyphless font** (Tesseract pdf.ttf / OCRmyPDF Occulta): maximum width-math purity, but documented viewer-compat pain (pdf.js/Preview mis-segmentation, `h e l l o` artifacts, uneditable output). Only revisit if real-font morph scaling proves insufficient.
- **Per-image font recognition** for the overlay font: nobody ships it; modern Tesseract (LSTM) dropped font attributes entirely; one normalized-metrics font + scaling is the industry answer.

---

## 5. Definition of done for "quantifiable and improving"

1. Every run writes per-doc, per-axis JSON (geometry / text / structure / unit-tests) + a history row.
2. CI fails on any axis dropping > 0.02 below the committed baseline; baselines ratchet only via reviewed regen commits.
3. Each roadmap item above names the single axis it must move; a change that moves no axis is reverted or re-scoped.
4. External validity: quarterly run against FUNSD/NAF/olmOCR-bench subsets to anchor internal numbers against public ground truth.
