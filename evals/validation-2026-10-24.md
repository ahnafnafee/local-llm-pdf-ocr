# Live OCR validation — 2026-10-24

**The live recognition gate remains failing.** The final hybrid run passes
42/46 content assertions and the grounded run passes 42/46.
The application fixes and candidate comparisons must not be reported as a complete
solution to the remaining recognition errors. No newer model was promoted to the
default; the [2026 comparison](model-comparison-2026-10.md) records the alternatives.

This run exercises the runtime changes through `9b60a44` using actual local HTTP
inference. Models were served by standalone CUDA llama.cpp, with no inference
stubs and no dependency on a model-management application.
Hybrid evidence was collected after thumbnail restoration at `e115f6a`; later
runtime changes affect grounded recovery only. Grounded evidence uses `9b60a44`.

## Application validation

- **668 offline tests passed**, with 18 opt-in live cases deselected, in 60.62 seconds.
- The wheel and source distribution build successfully. All packaged runtime
  Python sources and web assets match the working sources.
- All three real CLI modes pass the 130-page source test with `--pages 2,4`:
  only source indices 1 and 3 are rasterized, and Markdown preserves source
  page headings 2 and 4. Hybrid/text use two requests each; grounded uses
  2 full-page requests and
  0 bounded recovery requests.
- Both real web uploads return HTTP 200 and searchable two-page PDFs. Only the
  selected second page receives text.
- AVIF word F1 is 1.0000 for hybrid and
  1.0000 for grounded. These are text scores; the existing
  image geometry reference has known alignment limitations.
- The browser exposes all model presets and persists their values. API tests
  verify custom aliases in hybrid, text-only and grounded modes, and HTTP 400
  for an invalid preset.

## Complete PDF results

| Document | Mode | Output blocks | Text similarity | CER ↓ | Word F1 | Content checks |
|---|---|---:|---:|---:|---:|---:|
| digital.pdf | hybrid | 35 | 0.9601 | 0.0618 | 0.9277 | 13/13 |
| hybrid.pdf | hybrid | 16 | 0.8922 | 0.2434 | 0.9423 | 13/13 |
| handwritten.pdf | hybrid | 17 | 0.9045 | 0.1303 | 0.9178 | 7/10 |
| dense.pdf | hybrid | 357 | 0.7794 | 0.3721 | 0.6775 | 5/5 |
| notes.pdf | hybrid | 458 | 0.8280 | 0.2356 | 0.7075 | 4/5 |
| digital.pdf | grounded | 34 | 0.6855 | 0.3711 | 0.9863 | 13/13 |
| hybrid.pdf | grounded | 17 | 0.8303 | 0.3557 | 0.9417 | 12/13 |
| handwritten.pdf | grounded | 15 | 0.9716 | 0.0494 | 0.9517 | 10/10 |
| dense.pdf | grounded | 177 | 0.7172 | 0.3525 | 0.5318 | 2/5 |
| notes.pdf | grounded | 306 | 0.7612 | 0.3455 | 0.7038 | 5/5 |

Recognized source pages: hybrid **27/27**, grounded
**27/27**. Grounded page-error maps are retained with the raw
responses; 0 page errors were
reported. A failed document receives no passing report, and an empty page cannot
silently become a successful output.

Remaining failed checks and comparisons:

- handwritten.pdf, hybrid: missing: 'Algorithms'
- handwritten.pdf, hybrid: missing: 'Halting'
- handwritten.pdf, hybrid: order rule operand missing: 'Algorithms' / 'Correctness'
- handwritten.pdf/hybrid: checks_pass_rate 0.700 below baseline 1.000 (-0.300)
- notes.pdf, hybrid: missing: 'FK can have null value'
- hybrid.pdf, grounded: missing: '24 Barney Lane'
- dense.pdf, grounded: missing: 'SQL in Application Code'
- dense.pdf, grounded: missing: 'statement level interface'
- dense.pdf, grounded: missing: 'cursor'

No grounded metric baseline is committed. Completing every page and passing
content assertions does not establish an absolute bounding-box quality threshold.
The captures retain geometry metrics separately from the word scores.

## Input and sampling controls

Grounded JSON that terminated normally sometimes contained only a small header
fragment. The default full-page prompt now checks whether its boxes cover at
least 20% of dark source ink, allowing a coordinate margin. Severely incomplete
responses request bounded source-region recovery. Uniform backgrounds, sparse
headers, custom prompts and native formats have explicit coverage tests. This
heuristic can request another view of an illustration; it is not a calibrated
confidence probability or a guarantee of complete recognition.

The subsequent full run exposed an empty recovery band represented by a block
with an empty `content` string. Recovery now accepts that explicit empty-text
representation as well as an empty array; malformed data and completely empty
selected pages still fail. The preceding failure is retained in the comparison
captures, and the final results above come from a fresh run after this fix.

Another dense-page replay returned thousands of bullet characters for a recovery
crop whose owned core was exactly white. Marks existed only in the adjacent
overlap. The pipeline now skips exactly white owned cores before inference;
any nonwhite source pixel, including a faint mark, retains recognition. The
captured token-limit failure remains available, and an empty selected page still
fails. This addresses the observed blank-region loop without raising the page
deadline or accepting truncated output.

The focused dense replay of the initial ink-coverage change at `55c782f` passed
5/5 assertions and completed
all three pages without errors. Its word F1 was 0.7415, compared with 0.2435 in
the preceding complete-page attempt. The final full-run results above remain
separate observations; no best seed or probe replaces them.
The final dense run still omits checked phrases despite completing every page,
so the omission heuristic does not resolve full-document recognition quality.

The final hybrid run retains the previous JPEG PDF thumbnails. Replacing those
with PNG changed both recognition and detector geometry: the exploratory full
run passed 42/46 assertions and failed several geometry floors. Restoring the
thumbnails recovered those geometry comparisons. Detailed source crops and
grounded pages use PNG; they are independent of the detector thumbnail.

The fixed-seed thumbnail replay passed two of three focused cases, including
the form geometry checks, but still missed Halting (9/10 handwriting assertions).
The unseeded full run above is authoritative for its own outcome. Neither a
passing seed nor a better crop probe substitutes for that result. Alternate crop
encoding and canvas probes did not resolve the remaining handwriting error.

Hardware: RTX 3080, 10 GiB VRAM; Python 3.12; Surya on CPU. The server is llama.cpp
CUDA build 10735 / commit 842b18804, with an 8192-token context and one inference
slot. Both evaluators use concurrency 1 and maximum image dimension 1024.

- Hybrid: olmOCR-2-7B-1025 Q4_K_M, temperature 0.1, seed unset.
- Grounded: Qwen3-VL-8B-Instruct Q4_K_M, temperature 0.7, top-p 0.8,
  presence penalty 1.5, seed unset, minimum 1024 image tokens. The request limit
  is 4096 output tokens; the 240-second page deadline includes bounded recovery.
  The server prompt cache is capped at 256 MiB.

The endpoint default remains `http://localhost:1234/v1`; this validation explicitly
used `http://127.0.0.1:28127/v1`. Endpoint and model IDs remain configurable.

## Evidence and reproduction

[Raw results, predictions, source indices and hashes](captures/validation-2026-10-24/manifest.json)
retain each final observation. The source-verified reference corrections from the
previous run are unchanged. Historical controls are archived, and current controls
are replayed from frozen pre-fix predictions against the audited references;
new candidate predictions were not used as replacement baselines. See
[evaluation provenance](README.md).

Run against each corresponding loaded model:

```powershell
uv run scripts/confidence_eval.py --path hybrid --api-base http://127.0.0.1:28127/v1 --hybrid-model allenai/olmocr-2-7b --concurrency 1
uv run scripts/confidence_eval.py --path grounded --api-base http://127.0.0.1:28127/v1 --grounded-model qwen/qwen3-vl-8b --concurrency 1
```

Failed content checks, incompatible controls, missing pages and metric regressions
return nonzero. Baselines and tolerances were not changed to turn failures into
passes. The [October 23 report](validation-2026-10-23.md) is a historical snapshot,
not the result of this final implementation.
