# Live OCR confidence evaluation

Run `uv run scripts/confidence_eval.py --path hybrid --api-base http://localhost:1234/v1`
against a running vision server. Use `--path grounded` to exercise grounded OCR.
These are actual model calls. Unit tests use injected transports and do not establish
recognition quality. The image evaluator is `scripts/confidence_image.py`.

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

The committed historical baseline numbers and tolerances have not been lowered.
They were produced with earlier references and pooled page matching. Comparisons
across these changes are diagnostic, not controlled measurements of an OCR regression.
Keep historical results separate from version 2 results and review a future baseline
with its reference version, model weights, serving configuration and sampling settings.
