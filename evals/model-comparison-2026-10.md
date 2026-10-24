# OCR model comparison, October 2026

The 2026 candidates were tested through actual local HTTP inference on an RTX 3080
with 10 GiB VRAM. A newer release did not reliably improve this repository's
handwritten forms and notebook pages. The initial screen and complete application
runs are reported separately; a successful HTTP response is not a quality pass.

LightOnOCR-3 was the most promising modern grounded candidate in this corpus:
its full run completed all 27 pages and passed 45/46 content checks. Its remaining
notebook error, weaker dense/notebook word scores and seed-dependent form error
prevent a reliable default promotion. Qwen3.5 read the notebook more accurately
but failed to complete a dense page within the deadline. Existing defaults remain
for compatibility, while model profiles make the alternatives configurable.

## Candidates and native conventions

| Candidate | Release generation | Tested weights | Native request |
|---|---|---|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | March 2026 | Q4_K_M / Q6_K, approximately 5.68 / 7.46 GB plus vision encoder | Transcription / prompted JSON; thinking disabled |
| [LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B) | October 2026 | Q8_0, approximately 4.48 GB plus vision encoder | Image before instruction; empty transcription prompt or `grounding` |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | May 2026 | Official GGUF, approximately 0.94 GB plus vision encoder | `OCR:` / `Spotting:`, preserving location tokens |
| [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | June 2026 | Official BF16, approximately 6.67 GB, 3.34B total parameters | Official Transformers implementation and `document parsing.` |
| [Chandra OCR 2](https://huggingface.co/datalab-to/chandra-ocr-2) | March 2026 | Q8_0, approximately 5.16 GB plus vision encoder, 5.30B total parameters | Official HTML layout prompt, temperature 0 / top-p 0.1 |

These are serving configurations, not new application dependencies. Unlimited-OCR
used its reference implementation because stock llama.cpp did not implement its
required attention architecture. Chandra's weights have separate commercial license
conditions; its code license does not remove those conditions. The previous
[olmOCR-2 release](https://huggingface.co/allenai/olmOCR-2-7B-1025) is from October
2025, before these releases.

## Initial first-page screen

Each PDF's first page was rendered at 200 dpi, resized to a maximum dimension of
2048 pixels, and sent as PNG. The additional AVIF example used its source pixels.
Every request used the model's native prompt and output format, with seed 0. The
46 content assertions cover the five PDF examples; the AVIF has transcription
metrics but no content-assertion count.

| Candidate / grounded configuration | Content assertions | Failures that matter |
|---|---:|---|
| Qwen3.5-9B, temperature 0.7 / top-p 0.8 / presence penalty 1.5 | 45/46 | Corrected the source's `mucas` spelling; some boxes were inaccurate |
| LightOnOCR-3, temperature 0.2 / top-p 0.9 | 45/46 | Misread the notebook's FK abbreviation; some dense coordinates needed stricter validation |
| PaddleOCR-VL-1.6, greedy | 41/46 | Lost handwritten diagram words and notebook text |
| Unlimited-OCR, greedy | 35/46 | Returned only an image region for the dense page, plus handwritten/form errors |
| Chandra OCR 2, greedy | 41/46 | Form errors and diagram text represented as generated descriptions |

Chandra's diagram descriptions were excluded from literal OCR. Including those
descriptions can recover keywords while also inventing explanatory prose, which
would be unsuitable for the searchable text layer. Paddle's spotting test used
preserved special tokens and a vision pixel budget of 1,605,632; an initial run
without those settings was not used for this ranking. Qwen's greedy dense-page
requests exhausted 8192 output tokens; its sampled configuration completed them.

Word F1, raw outputs, coordinates, failures and per-request settings are retained
in [the comparison captures](captures/model-comparison-2026-10/manifest.json).
The LightOn screening parser did not initially reject out-of-range coordinates;
its production parser does. Its screen must therefore not be read as a valid
bounding-box result. Formatting also differs: native HTML/Markdown can lower word
scores independently of recognition, and full-page text mode has no useful
line-level geometry score.

## Complete application runs and diagnosed defects

LightOnOCR-3 completed all 27 PDF pages without page errors or timeouts. It passed
45/46 content assertions, but word F1 fell to 0.6021 on the three-page dense
document and 0.5929 on the 21-page notebook. It was not promoted on the strength
of the first-page screen. Repeated grounded tests also exposed a seed-dependent
form spelling failure (8/9 cases passed). Its hybrid crop pipeline failed two
of the three focused seed-0 cases. Paddle and Qwen3.5 initially failed all three
hybrid cases, despite stronger full-page transcription on some documents.

After the lossless/context fixes, Qwen3.5 Q4_K_M passed all three notebook and
all three handwriting repeats, but still failed all three form spelling checks.
Its complete grounded run recognized 26/27 pages: dense page 2 exceeded the
240-second deadline after an 8192-token response and recovery. The other four
documents passed all 41 of their content assertions; the dense document failed
as a whole and did not receive a passing report. Notebook word F1 was 0.7224,
and geometry remained poor (character recall 0.4806). This is a failed run.

Those sampled Qwen runs inherited llama.cpp's top-k 40 and min-p 0.05 defaults
in addition to the explicit standard API parameters. The model's published
non-thinking settings specify top-k 20 and min-p 0.0. These server settings and
quantization are recorded separately; greedy hybrid results are unaffected by
top-k/min-p truncation. A later serving-configuration change must not be credited
solely to a weight-precision change.

The Q6_K weights were then tested on the actual hybrid form workflow at seed 0,
using the recommended server settings. The same spelling assertion failed
(12/13), so increasing precision did not resolve that acceptance failure.

Final default-model validation also exposed valid JSON containing only the first
two lines of a dense page. A conservative source-ink coverage check now sends
severely incomplete default JSON responses through bounded region recovery. In
the focused three-page replay, dense word F1 rose from 0.2435 to 0.7415 and content
checks rose from 3/5 to 5/5, with no page errors. This is a pipeline recovery
result, separate from the modern-model ranking, and box accuracy remains limited.

Further full runs exposed two blank-band failures: an explicit empty text block
was rejected, and a crop with marks only in its overlap generated repeated
bullets until its token budget was exhausted. Empty text blocks are now accepted
only for recovery regions. An exactly white owned core is skipped before
inference; any nonwhite pixel preserves the request. Those failed runs are
retained alongside the final replay, without relaxing the page deadline.

Input encoding and the alignment fallback both affected recognition. The hybrid
thumbnail path encodes JPEG at quality 50, which loses small source strokes, but
a blanket PNG replacement also changed layout detections and regressed the
existing model. A controlled replay restored the previous PDF thumbnails: two
of the three focused cases passed, compared with one under PNG, and handwriting
improved from 7/10 to 9/10 assertions. The production hybrid thumbnail behavior
is therefore retained. Detailed source crops and grounded images use PNG.

Crop-backed text matching can retain a unique, strongly matching span from the
full-page transcription. It does not populate blank regions, change differing
numbers, or choose between similarly plausible fields. Generic prompts explicitly
preserve source spelling. These changes improve some cases, but the live failures
show that they do not guarantee a correct reading.

The earlier full LightOn run used the production 150 dpi/JPEG80 grounded path,
before the lossless correction. It is retained as diagnostic evidence, not a
controlled comparison with later lossless runs. Final application validation is
reported in [the final validation](validation-2026-10-24.md), separately from this
initial screen. No source reference, content check,
or tolerance was weakened to improve a candidate's result.
