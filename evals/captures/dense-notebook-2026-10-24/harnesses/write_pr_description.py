"""Prepare PR metadata from the completed local validation artifacts."""
import json
from pathlib import Path
import subprocess
import re

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))

assert all(row['exit_code']==0 for row in read('final-workflow-status.json').values())
tests=read('final-live-outcomes.json')
assert len(tests)==28 and all(row['outcome']=='passed' for row in tests)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
capture=ROOT/'evals/captures/dense-notebook-2026-10-24/manifest.json'
runtime=json.loads(capture.read_text())['code_commit']
rows=[]
for row in read('before-after-comparison.json')['rows']:
    before,after=[row['variants'][key]['visible'] for key in ['before','after']]
    rows.append(f"| {row['document']} | {row['mode']} | {before['bow_f1']:.4f} → {after['bow_f1']:.4f} | "
                f"{before['hmean']:.4f} → {after['hmean']:.4f} |")
images={mode:read(f'image-{mode}.json')['metrics']['bow_f1'] for mode in ['hybrid','grounded']}
offline_count=re.search(r'(\d+) passed, 28 deselected',
    (OUT/'offline-final-v18.log').read_text())[1]
body=f'''Selecting `--pages 2,4` previously rasterized the complete input in hybrid/text mode, while grounded mode ignored the selection. Resolve ranges before rendering in all three modes and multi-frame images, preserve source indices, and report errors against the original page numbers. PDF/HTML output retains document backgrounds; Markdown/text contains processed pages.

Separate inference through an injectable `VisionClient`, a configurable OpenAI-compatible adapter, and model profiles. CLI, web discovery, authentication and client ownership share configuration without server-management APIs. Use **LightOnOCR-3-4B as the default in every mode**, keeping **`http://localhost:1234/v1`**. Documentation, package metadata and repository description are server agnostic.

Share native parsing and source-region recovery across both OCR paths. Detected lines guide contextual crops, while four overlapping source views find tiny printed lines missed in dense-page thumbnails. Load detailed pixels lazily and stop crop padding before adjacent text. This recovers the dense exercise panel, printed notebook schema and labels inside an embedded photograph. Strengthen sparse faint colored strokes in inference crops, anchor nearby ordered lines and restore regular paragraph rows without changing their words. Preserve separate table cells, remove displaced duplicates and decode supported mathematical markup into searchable Unicode with PDF font fallback.

Malformed or truncated full-page reads switch to detected regions. Local sampled retries and subdivision remain inside the existing **240-second page deadline**. Completed empty regional responses—including valid native coordinate tags without text—preserve the page; malformed coordinates, reasoning-only responses and entirely unrecognized selected pages remain errors.

**Compatibility:** The model default is `lightonai/LightOnOCR-3-4B`, and the image cap defaults to 2048 pixels. Serve the official LightOn template with thinking disabled, or configure another model/profile. Other model profiles retain their alignment strategies. Custom PDF/grounded backends accept `pages`; native-layout PDF handlers also accept `lossless`. Detailed source loading and layout detection are injectable. See `docs/inference.md`.

Validation uses runtime commit `{runtime}` with standalone llama.cpp and actual LightOn Q8_0 inference:

- **{offline_count} offline tests passed**; wheel/source builds and packaged-runtime verification passed.
- **28 live cases passed on the final implementation**, 14 per mode. Three seeds cover first-page form, handwriting, notebook and dense inputs. Later-page checks inspect dense pages 2–3 and notebook pages 6, 13, 17 and 21 in actual searchable PDFs, including field positions, duplicate counts and the printed exercise panel.
- Both complete corpus runs recognize **27/27 pages** and pass **46/46 content checks**: **54 pages and 92 checks**, without page errors or timeouts. Actual PDF exports of both runs also pass the dense and notebook position assertions.
- All three real CLI modes pass the **130-page `--pages 2,4`** check. The trace distinguishes regional rereads from whole-page requests and forbids OCR renders of unselected pages. Requests use the built-in model default.
- Both actual web API uploads return HTTP 200 searchable PDFs with text only on selected page 2. AVIF word F1 is **{images['hybrid']:.4f}** in hybrid mode and **{images['grounded']:.4f}** in grounded mode.

The following comparison rescores unchanged earlier LightOn captures and final outputs against the same current references with the same HTML/LaTeX decoder:

| Document | Mode | Visible word F1 | Box H-mean |
|---|---|---:|---:|
{chr(10).join(rows)}

Geometry improves in all four comparisons. Transcription is mixed: grounded dense and hybrid notebook word F1 improve, while hybrid dense falls by 0.0201 and grounded notebook by 0.0041. Ambiguous handwriting, mathematical symbols and word positions inside photographed formulas remain imperfect. The report includes raw metrics and character-position coverage, which is not character-recognition accuracy.

The dense reference adds 16 printed lines verified against source pixels. Existing dense annotations and the notebook reference remain unchanged. The historical olmOCR dense control is rescored from frozen predictions, with its previous reference version archived. No content-matching or metric tolerance was relaxed and final outputs were not promoted to baselines. Full runs use an empty baseline directory, so these are content-check results and explicit same-reference comparisons, not a cross-model baseline-regression pass.

Earlier failing runs remain available, including omitted printed questions, duplicated schema text, empty-region page failures, the clipped-question hallucination and a model-server stall. PP-DocLayoutV3 did not show a consistent placement improvement in the earlier local experiment; its diagnostic script remains separate from production dependencies.

[Validation report and raw evidence](https://github.com/ahnafnafee/local-llm-pdf-ocr/blob/{head}/evals/dense-notebook-2026-10-24.md) · [Source addition audit](https://github.com/ahnafnafee/local-llm-pdf-ocr/blob/{head}/evals/reference-additions-2026-10-24.json) · [Earlier model comparison](https://github.com/ahnafnafee/local-llm-pdf-ocr/blob/{head}/evals/model-comparison-2026-10.md)

BREAKING CHANGE: The default model is LightOnOCR-3-4B and the default image cap is 2048 pixels. Custom PDF/grounded backends must accept page selection, and native-layout PDF handlers must accept lossless rendering. The endpoint default remains unchanged.

Fixes #26
Fixes #27
'''
(OUT/'pr-body.md').write_text(body,encoding='utf-8')
print('Prepared PR description from verified local results')
