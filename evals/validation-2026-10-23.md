# Live OCR validation — 2026-10-23

Grounded page timeouts and empty dense output are resolved in this run, but **the OCR quality gate is still failing**. Both modes pass 45 of 46 document checks, with different recognition errors. Historical hybrid metric regressions remain visible; no baseline numbers or tolerances were lowered.

The tested implementation is `c2d3f07`; source-reference corrections are in `6fe8cdf`. This snapshot uses real SDK requests to a standalone CUDA llama.cpp server, with no inference stubs or LM Studio service.

## Verification

- **605 local tests passed**, including Surya integration tests (`pytest -m "not live_llm" -q`, 43.67 seconds).
- `uv build --offline` built the 0.3.0 wheel and source distribution. Their final runtime sources were inspected.
- Diff validation and redacted secret scans passed.
- All three real CLI modes passed a 130-page source test with `--pages 2,4`: only source indices 1 and 3 were rasterized, exactly two real inference calls were made, and Markdown retained source page headings 2 and 4.
- Both real web uploads returned HTTP 200 and searchable two-page PDFs. Only selected source page 2 received text.
- Both AVIF workflows produced text. Word F1 was 1.000 for each; this is a measured content score, not an absolute geometry-quality pass.

## Final PDF confidence results

Matching and character coverage retain source page IDs. Dense/notes use source-corrected version 2 references. Other references are unchanged.

| Document | Mode | Output blocks | Text similarity | CER ↓ | Word F1 | Content checks |
|---|---|---:|---:|---:|---:|---:|
| digital.pdf | hybrid | 35 | 0.9601 | 0.0618 | 0.9251 | 13/13 |
| hybrid.pdf | hybrid | 15 | 0.6159 | 0.5895 | 0.9765 | 13/13 |
| handwritten.pdf | hybrid | 17 | 0.9324 | 0.0909 | 0.9452 | 10/10 |
| dense.pdf | hybrid | 355 | 0.7530 | 0.4194 | 0.6507 | 5/5 |
| notes.pdf | hybrid | 451 | 0.7263 | 0.3639 | 0.6529 | 4/5 |
| digital.pdf | grounded | 21 | 0.8532 | 0.1812 | 0.9909 | 13/13 |
| hybrid.pdf | grounded | 16 | 0.9300 | 0.1857 | 0.9712 | 12/13 |
| handwritten.pdf | grounded | 19 | 0.9308 | 0.1136 | 0.9452 | 10/10 |
| dense.pdf | grounded | 267 | 0.4758 | 0.6599 | 0.7877 | 5/5 |
| notes.pdf | grounded | 344 | 0.7802 | 0.3491 | 0.7644 | 5/5 |

Grounded OCR populated **all 27 PDF source pages**, including all three dense pages and all 21 notebook pages. The AVIF image is an additional input. Every final grounded response has an empty `page_errors` map. Truncated dense responses triggered bounded source-region recovery and were not accepted as complete JSON.

Remaining failures are concrete:

- Hybrid misses the corrected notebook fact `FK can have null value`, transcribing the letters as mathematical text. The handwritten example now passes all 10 checks, including its heading and Halting label.
- Grounded transcribes the form address as `24 Bancy Lane`, failing the `24 Barney Lane` check. All five corrected notebook checks pass, including FK.
- Hybrid still trips historical geometry/text floors on handwritten, dense and notes. Its handwritten result matches all 14 reference blocks, but 17 output blocks include three legitimate continuation lines, reducing one-to-one precision to 0.8235. Text similarity improved from the initial run’s 0.7148 to 0.9324; CER improved from 0.3243 to 0.0909.
- No committed grounded metric baseline exists. Completing every page and passing content checks does not establish acceptable absolute bounding-box accuracy.
- An AVIF overlay shows reference rectangles displaced from visible handwriting. The unchanged image geometry reference needs its own audit; the perfect word scores do not validate those rectangles.

## Reference integrity and comparison limits

All 842 dense/notebook layout regions were compared with source pixels rendered at 288 dpi. Correct 640 transcriptions and classify ten source-verified blank regions as non-content. Preserve original block IDs and coordinates. The audit records before/after text, source pages, boxes, PDF hashes and fixture hashes. It does not copy predictions into the references.

See [reference corrections](reference-corrections.json) and [review instructions](README.md). Overlapping and clipped historical annotation regions remain, and ambiguous readings were not silently replaced. The new strict FK fact also removes an old edit allowance that accepted PK.

The five historical baseline files are byte-for-byte unchanged. Their earlier reference text and pooled page matching differ from this evaluation, so cross-version score differences are diagnostic rather than controlled regression measurements. Both final evaluator commands correctly returned **exit status 1**; the passing unit suite must not be presented as a green live quality gate.

## Runtime configuration

The application retains `http://localhost:1234/v1`, `allenai/olmocr-2-7b` and `qwen/qwen3-vl-8b` as its endpoint/model defaults. This test explicitly used `http://127.0.0.1:28127/v1`. Documentation and package metadata remain server agnostic.

Hardware: RTX 3080 (10 GiB), Python 3.12.2, PyTorch 2.9.1+cpu and Surya. The server binary was llama.cpp CUDA build 10735 / commit 842b18804 (Prism distribution). Hybrid used one 8192-token slot and concurrency 1. Grounded used two 8192-token slots, concurrency 2, and `--image-min-tokens 1024`, following the server warning and [Qwen guidance](https://github.com/ggml-org/llama.cpp/issues/16842#issuecomment-3476399330). KV caches were q8_0, flash attention enabled, batch 256 and microbatch 128.

- Hybrid: OlmOCR-2-7B-1025 Q4_K_M; temperature 0.1; seed unset.
- Grounded: Qwen3-VL-8B-Instruct Q4_K_M; temperature 0.7, top-p 0.8, presence penalty 1.5; seed unset.
- Grounded request limit: 4096 tokens; page deadline: 240 seconds including bounded recovery.

Exact weight SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| hybrid_weights_sha256 | `821ed0a31450d7d265bcc05519a28ebb43eb3cd2738ca543fa11b78f35153e25` |
| hybrid_projector_sha256 | `fe90b9db826810aa37a17b53bd575ee15a9947e9075d73a8f68e284a38ab5eb4` |
| grounded_weights_sha256 | `43e5606b05c5df8dfad9894cf631045b395f605c8181fbdd074dc4e7c59d09a0` |
| grounded_projector_sha256 | `30cc93c7375b529b2abe165d1597e12f6717bf87565655a546727c4bd2bb5ada` |

Grounded weights came from public revision `2e40238691f776f0d2ac8c01fbf2942942f763d8` of `lmstudio-community/Qwen3-VL-8B-Instruct-GGUF`; the repository name does not require its publisher’s serving application.

Reproduce the final confidence commands against the corresponding loaded model:

```powershell
uv run scripts/confidence_eval.py --path hybrid --api-base http://127.0.0.1:28127/v1 --hybrid-model allenai/olmocr-2-7b --concurrency 1
uv run scripts/confidence_eval.py --path grounded --api-base http://127.0.0.1:28127/v1 --grounded-model qwen/qwen3-vl-8b --concurrency 2
```

An initial grounded attempt started before model readiness and received loading/503 errors; it is preserved separately as `grounded-startup-*` and excluded from final results. The final run began only after health returned 200. Earlier 4B and 8B probes use different models/settings and are not a controlled speed comparison.

A focused fixed-seed (42) form replay with the current prompt passed 13/13; the alternative task prompt passed 12/13 under the same settings and was not adopted. This demonstrates sampling sensitivity and is not substituted for the unseeded final run.
