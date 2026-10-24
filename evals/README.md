# Live OCR confidence evaluation

Run `uv run scripts/confidence_eval.py --path hybrid --api-base http://localhost:1234/v1`
against a running vision server. Use `--path grounded` to exercise grounded OCR.
These are actual model calls. Unit tests use injected transports and do not establish
recognition quality. The image evaluator is `scripts/confidence_image.py`.

The [dense and notebook validation](dense-notebook-2026-10-24.md) contains the current
regional recovery results, repeated live tests and workflow checks. The preceding
[LightOnOCR-3 validation](lighton-layout-2026-10-24.md) retains the default-model
migration and PP-DocLayoutV3 experiments. The [earlier October 24 validation](validation-2026-10-24.md) preserves
the preceding models' failures. The [2026 model comparison](model-comparison-2026-10.md)
records modern candidates, native serving conventions and unsuccessful runs.

The runners retain source page indices through matching and character coverage.
Boxes on different pages cannot match, even when their normalized coordinates are
identical. `compute_report(..., pipeline_page_indices=None)` retains pooled matching
for reproducing historical reports; new multi-page evaluations should supply indices.

Missing pages, empty outputs, failed document checks and metric regressions cause a
nonzero exit. Failed runs cannot update baselines. One-to-one geometry precision
still penalizes legitimate split lines; character coverage reports the complementary
split/merge-tolerant result. Neither score establishes transcription correctness by itself.

## Reference corrections

The dense and notes fixtures were originally bootstrapped from OCR output. Visual
comparison of all 842 layout regions with the original PDFs found incorrect words,
text assigned to the wrong boxes, invented mathematics, HTML tags assigned to printed
fields, and text assigned to blank dotted paper. The version 2 corrections preserve
the original coordinates and block IDs. Ten demonstrably blank regions remain in the
layout with `empty_line` labels and are excluded from text scoring. Ambiguous readings
were not silently replaced, and the geometry still has overlapping and clipped regions.

[reference-corrections.json](reference-corrections.json) records every changed block,
its original and corrected text, page, coordinates, source SHA-256 and fixture hashes.
The source verification used PDF pixels rendered at 288 dpi, independently of the
predictions being scored. To inspect a correction, run:

```powershell
uv run scripts/review_reference_corrections.py dense 1 --output review.png
uv run scripts/review_reference_corrections.py notes 0 --output review.png
```

The red rectangle marks the original annotation region. The notes content check now
requires `FK can have null value`, as written on page 1; its former `PK` reading was
false. Model text must satisfy the corrected fact without the former edit allowance.

Dense reference version 3 adds the previously omitted printed exercise panel on
source page 3. Its 16 lines and source-pixel coordinates were transcribed from a
600 dpi rendering and are recorded in
[reference-additions-2026-10-24.json](reference-additions-2026-10-24.json).
All existing reference text and coordinates are retained. The old dense control is
archived in `baselines/previous-reference-v2/`; the active control rescores the same
frozen predictions against the expanded reference without changing tolerances.

## Comparable regression controls

The old numbers remain unchanged in `baselines/legacy-pooled-v1/`. They used earlier
references and pooled page matching, so applying them to corrected references and
page-aware scoring produced false regression failures. Active baselines now include
source, reference and check hashes, scoring protocol, matching strategy, image limit
and model ID. Text hashes normalize checkout line endings. Incompatible or unversioned
baselines fail explicitly rather than comparing unrelated scores.

The current hybrid controls are recomputed from the frozen real outputs of commit
`1cb795d`, before the crop-canvas and alignment fixes, in
`captures/before-crop-canvas/`. They are not predictions from the improved run. The
original control still fails the FK fact; all live content checks remain mandatory
even when a control failed one. Existing tolerances remain 0.02 for geometry/checks
and 0.04 for transcription metrics. Model weight hashes and sampling settings are
recorded in capture provenance; the server must use those weights for a controlled run.

Replay a capture without making model calls:

```powershell
uv run scripts/replay_confidence.py evals/captures/before-crop-canvas/notes__hybrid.json
```

That example deliberately exits nonzero because it preserves the failed pre-fix
recognition. `--write-control` explicitly migrates the comparison point from a frozen
capture after reference/scoring changes, preserving its predictions and source hash.
The live evaluator's `--update-baselines` remains a separate, explicit operation and
refuses to write when any document or content check fails.

## Repeated live recognition tests

The 28 opt-in pytest cases exercise the real pipeline on notebook, handwriting,
form and dense-page regressions. Twenty-four repeat first-page cases with seeds
0, 1 and 2; four inspect later dense and notebook pages, including diagram fields,
embedded photo text, mathematical extraction and source positions in actual PDFs.
They skip without an explicit endpoint
and mode; ordinary unit tests never start a model server.

```powershell
$env:OCR_LIVE_API_BASE = 'http://localhost:1234/v1'
$env:OCR_LIVE_MODE = 'hybrid' # or grounded, with that model served
uv run pytest -m live_llm tests/test_live_quality.py -q
```

Set `OCR_LIVE_MODEL` to override the mode's existing model default and
`OCR_LIVE_API_KEY` when authentication is needed. The hybrid form cases also inspect
the written PDF to verify that questions and answers occupy their source positions.
Full-document quality and layout are still measured by `scripts/confidence_eval.py`.
