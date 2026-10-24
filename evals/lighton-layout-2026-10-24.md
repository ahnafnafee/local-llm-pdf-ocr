# LightOnOCR-3 and layout validation — 2026-10-24

LightOnOCR-3-4B is the default for hybrid, text-only and grounded OCR. Both fresh
full-document runs pass **46/46 content assertions** and recognize **27/27 pages**.
The **24 focused live regressions** also pass across two engines and three seeds.
No page errors or timeouts occurred in the final full runs. These results resolve
the previously failing checks; they do not establish perfect transcription or
bounding-box accuracy on the dense document and notebook.

The full runs and workflow checks use runtime commit `6ec5cc7`. Inference went
through actual local HTTP requests to standalone CUDA llama.cpp on an RTX 3080
with 10 GiB VRAM. The endpoint remains configurable and defaults to
`http://localhost:1234/v1`; this evaluation used `http://127.0.0.1:28127/v1`.

## What changed

Native page grounding keeps handwriting and form fields in context. Hybrid mode
uses those coordinates, snapping only unambiguous lines to Surya detections and
preserving every unmatched native block. This removes the former length-based
alignment and isolated-word rereads from the LightOn default path. Lossless PNG
and a 2048-pixel cap preserve source detail. Other profiles retain their existing
alignment strategy and legacy library thumbnail defaults; explicit custom page
prompts retain their requested transcription behavior.

Greedy decoding preserves the form's original `mucas` spelling in the tested
cases. Dense responses can still contain invalid coordinates or truncated table
markup, so native recognition gets one sampled same-image retry before bounded
region subdivision. Every attempt shares the existing 240-second page deadline.
The final grounded run exercised both invalid-coordinate recovery and a response
that reached its 4096-token limit. Truncated output was rejected and recovered,
not accepted as a successful transcription.

Grounding sometimes changes an acronym into mathematical notation. A separate
native plain transcription of the same page supplies a second reading only when
an inline letter/subscript expression appears inside prose. Replacement requires
identical surrounding prose, a unique uppercase acronym, equal letter count and
similar letters. This recovered the notebook's `FK can have null value` without
a fixture lookup or hardcoded correction. Equations, numeric subscripts and
conflicting readings are left unchanged. HTML table cells are decoded for the
PDF/HTML text layer; Markdown keeps the original table markup.

The served model must use its [official chat template](https://huggingface.co/lightonai/LightOnOCR-3-4B/blob/main/chat_template.jinja)
with thinking disabled. The tested GGUF's inherited thinking configuration put
OCR into `reasoning_content` and left normal content empty. The adapter now
reports that configuration error explicitly; it does not treat reasoning as OCR.

## Full-document results

| Document | Mode | Blocks | Raw word F1 | Box H-mean | Character recall | Checks |
|---|---|---:|---:|---:|---:|---:|
| digital.pdf | hybrid | 18 | 0.9970 | 0.7059 | 0.9052 | 13/13 |
| hybrid.pdf | hybrid | 14 | 1.0000 | 0.3077 | 0.9542 | 13/13 |
| handwritten.pdf | hybrid | 14 | 0.9324 | 0.9286 | 0.9935 | 10/10 |
| dense.pdf | hybrid | 296 | 0.6134 | 0.1906 | 0.3798 | 5/5 |
| notes.pdf | hybrid | 225 | 0.5706 | 0.4942 | 0.7984 | 5/5 |
| digital.pdf | grounded | 18 | 0.9970 | 0.7647 | 0.9048 | 13/13 |
| hybrid.pdf | grounded | 14 | 1.0000 | 0.3462 | 0.9572 | 13/13 |
| handwritten.pdf | grounded | 7 | 0.9324 | 0.5714 | 0.9935 | 10/10 |
| dense.pdf | grounded | 286 | 0.6031 | 0.2366 | 0.3865 | 5/5 |
| notes.pdf | grounded | 190 | 0.5612 | 0.4627 | 0.8197 | 5/5 |

The five sources comprise one digital page, one form, one handwriting page,
three dense pages and 21 notebook pages. Geometry matching is page-aware.
Raw word scores retain the established normalization, including its sensitivity
to native HTML table markup. Geometry scores penalize different segmentation,
and the audited references retain some overlapping or clipped regions. These
limitations do not explain away all errors: dense-page and notebook transcription
and line placement remain weaker than the checked phrases alone suggest.

A separate output-layer check decodes HTML table cells without changing the
primary metrics. It raises hybrid dense/notebook word F1 to 0.6403/0.6092 and
grounded dense/notebook F1 to 0.6299/0.5988. Markup accounts for part of the score
loss; remaining transcription errors are substantial. This supplemental result
is retained independently and is not an acceptance threshold or baseline.

The 46 assertions are fixed document facts and ordering checks, not 46 complete
page transcriptions. No reference, assertion or tolerance changed in this work.
Existing olmOCR controls are incompatible with the new model and image cap.
The final runs therefore used a separate empty baseline directory: **content
checks passed; a cross-model metric regression gate was not run**. No new output
was promoted to a baseline. The earlier failing runs remain archived separately.

## Actual end-to-end checks

- **681 offline tests passed**, 24 live cases deselected, in 47.38 seconds. This
  includes Surya integration, real SDK transport, cancellation, native parsers,
  profile defaults, output geometry and injected-client ownership.
- **24 live cases passed**: form, handwriting, notebook and dense first pages in
  hybrid and grounded modes, each with seeds 0, 1 and 2. The form cases inspect
  actual PDF search coordinates for the question and handwritten answer. Logs
  retain the separate 9 + 9 + 3 + 3 executions.
- All three CLI modes passed on a **130-page input with `--pages 2,4`**. Only
  source indices 1 and 3 were rasterized, and output headings retained 2 and 4.
  The runs omitted `--model` and verified the actual request model. Hybrid and
  text-only used 2 and
  2 requests; grounded used
  2 full-page and
  0 recovery requests.
- Both actual web uploads returned **HTTP 200** searchable PDFs with two source
  pages and OCR text only on selected page 2. The model form field was empty;
  discovery and recognition used the configured LightOn default. The local
  `.env` model override was migrated from olmOCR; endpoint and credentials were
  preserved.
- The actual AVIF runs measured word F1 **1.0000**
  in hybrid and **1.0000** in grounded mode.
  The image's geometry reference has known alignment limitations.
- Wheel and source-distribution builds passed; packaged runtime Python and web
  assets match the working sources.

Full grounded inference made 34 requests totaling
358.01 seconds; hybrid made 28 totaling
244.79 seconds. These are summed client request durations, including
retries and literal verification, not total pipeline timings or a throughput
benchmark. Server prompt caching and CPU detection also affect comparisons.

## PP-DocLayoutV3 experiment

[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) supplies document
regions, polygons and reading order. Its official
[safetensors conversion](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors)
ran locally on CPU using Transformers 5.20.2 in an isolated environment. The
production Surya environment remains on Transformers 4.57.3.

The first pages of the form, notes, handwriting and dense examples produced
14, 18, 29 and 40 regions at threshold 0.3, respectively, in approximately
0.84–1.02 seconds per page, excluding model loading. For a controlled geometry
comparison, fixed final LightOn predictions were split into lines and anchored
to PP regions with confidence at least 0.5, using the same conservative overlap
rule as production. Image-class regions were excluded. Text and native reading
order were held fixed; this does not evaluate every possible PP integration.

| First source page | Box recall before → PP | Average matched IoU before → PP |
|---|---:|---:|
| hybrid.pdf | 0.2368 → 0.1842 | 0.5427 → 0.6043 |
| notes.pdf | 0.2258 → 0.2258 | 0.5594 → 0.5594 |
| handwritten.pdf | 0.7143 → 0.7143 | 0.6425 → 0.6712 |
| dense.pdf | 0.2214 → 0.2214 | 0.4038 → 0.4038 |

The handwriting IoU improvement did not generalize. Form recall declined, while
notes and dense-page box recall did not improve. Some notebook and diagram text
was classified as images, and some regions overlapped. Separate actual LightOn
reads of PP-derived crops still corrected `mucas` to `mucus` and misread `FK` as
notation. Region crops did not solve those recognition failures.

PP-DocLayoutV3 is therefore available through the diagnostic
[`scripts/layout_probe.py`](../scripts/layout_probe.py), with JSON and overlay
evidence, but is not a new production dependency or the default line detector.
The tested production improvement comes from retaining native page context and
coordinates, plus conservative Surya anchors.

## Reproduction and evidence

The [capture manifest](captures/lighton-layout-2026-10-24/manifest.json) records
source/reference/check hashes, model and projector hashes, server configuration,
raw completions, outputs, page indices, negative controls and test logs. Text
hashes normalize CRLF to LF; binary hashes use original bytes. The historical
[model comparison](model-comparison-2026-10.md) and
[previous validation](validation-2026-10-24.md) remain unchanged.

The model was Q8_0 (4.48 GB), with its F16 vision projector (0.67 GB), an
8192-token server context, one inference slot, image-first messages and top-p 0.9.
Primary and literal reads use temperature 0; native retries use 0.2. Full runs
leave seed unset; focused tests set 0, 1 and 2. Sampled retries inherit server
top-k 40 and min-p 0.05. Grounded rendering uses 150 dpi and a 4096-token cap;
hybrid uses 200 dpi and 6144 tokens. Both use PNG, a 2048-pixel image limit,
concurrency 1 and the unchanged 240-second page deadline.

Against a correctly configured LightOn server, run:

```powershell
uv run pytest -m "not live_llm" -q
$env:OCR_LIVE_API_BASE = 'http://127.0.0.1:28127/v1'
$env:OCR_LIVE_MODEL = 'lightonai/LightOnOCR-3-4B'
$env:OCR_LIVE_MODE = 'hybrid'
uv run pytest -m live_llm tests/test_live_quality.py -q
$env:OCR_LIVE_MODE = 'grounded'
uv run pytest -m live_llm tests/test_live_quality.py -q
uv run scripts/confidence_eval.py --path both --api-base http://127.0.0.1:28127/v1 --hybrid-model lightonai/LightOnOCR-3-4B --grounded-model lightonai/LightOnOCR-3-4B --max-image-dim 2048 --concurrency 1 --baseline-dir .cache/lightonocr-3-baselines
```

The archived workflow harnesses ran from `.cache/lighton-layout-20261024`; restore
them there to reproduce their project-relative paths. The standalone PP probe
uses its separate environment, as described in [inference configuration](../docs/inference.md#optional-layout-experiments).
