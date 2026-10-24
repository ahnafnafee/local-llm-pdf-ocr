# Configuring and extending inference

The OCR pipeline does not start, load, or manage an inference server. Run a vision model on a server of your choice, using the defaults below or configuring its OpenAI-compatible API base URL and exact model ID. The adapter uses image chat completions and, when verification is enabled, the model catalog. It does not use server-specific management APIs.

## Configuration

The defaults are `http://localhost:1234/v1` for the API endpoint and `lightonai/LightOnOCR-3-4B` for every OCR mode. These values do not require a particular server application. `LLMConfig.from_env()` resolves the same model used by `OCRProcessor` and `PromptedGroundedOCR`.

To override them, set `LLM_API_BASE` and `LLM_MODEL` in the environment or a local `.env`, or pass `--api-base` and `--model`. Explicit values take precedence over environment settings, which take precedence over defaults. `LLM_MODEL`, when set, overrides the model in every mode. An endpoint includes its API prefix, such as `http://localhost:11434/v1` or `http://localhost:8000/v1`; custom prefixes are supported. The same configuration is used for full-page, crop, and grounded OCR, and for the web model picker.

Set `LLM_API_KEY` if the endpoint requires bearer authentication. Otherwise the SDK receives a neutral placeholder, which local servers without authentication can ignore. Keys are excluded from the configuration object's representation. The CLI and web server close their adapter after each run; library callers should close adapters they create.

For web requests, the environment key is used only for the configured `LLM_API_BASE`, or the built-in endpoint when that variable is unset. Changing the endpoint in the browser does not forward that secret to another server. To use an authenticated endpoint from the web UI, configure its URL and key together in the server environment.

Existing explicit settings and the `ensure_model_loaded()` compatibility method continue to work. That method now delegates to `ensure_model_available()`: the catalog advertises usable models and does not establish whether their weights are resident in memory. Servers with incomplete or absent catalogs can use `--no-verify-model` or the web verification checkbox.

The adapter targets the vision API documented by [Ollama](https://docs.ollama.com/api/openai-compatibility), [vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/), and [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), as well as other implementations of the same protocol. Model capabilities and serving configuration still determine OCR quality. Tests exercise requests through the real SDK with an in-memory HTTP transport; they do not certify every server/model combination.

## Library use and ownership

`VisionClient` exposes two async methods: `complete(prompt, image_base64, *, model, timeout, max_tokens, temperature)` returns text, and `list_models()` returns model IDs. OCR components use this interface, so an application can supply its own transport without depending on the SDK. Injected clients remain owned by the caller.

```python
from pdf_ocr import (
    LLMConfig, OCRPipeline, OCRProcessor, OpenAICompatibleClient, PDFHandler,
)
from pdf_ocr.core.text import TextHandler
from pdf_ocr.core.profiles import resolve_profile


async def extract_text(input_path, output_path):
    config = LLMConfig.from_env()
    profile = resolve_profile(config.model)
    async with OpenAICompatibleClient(
        config, image_first=profile.image_first,
        generation_options=dict(profile.sampling),
    ) as client:
        processor = OCRProcessor(model=config.model, client=client)
        await processor.ensure_model_available()
        pipeline = OCRPipeline(
            ocr_processor=processor,
            pdf_handler=PDFHandler(),
            output_writer=TextHandler().embed_structured_text,
        )
        return await pipeline.run(
            input_path, output_path, text_only=True, pages="1-3,5",
        )
```

`PromptedGroundedOCR(model=..., client=client)` accepts the same adapter for bounding-box OCR. `OpenAICompatibleClient(config, sdk_client=...)` also accepts a caller-owned SDK client for custom HTTP transports. Internally created adapters reuse one SDK client for discovery and inference. Call `aclose()` on an OCR component when it created its own adapter; that method leaves injected clients open.

Prompt and output conventions belong to `ModelProfile`, separately from the transport. Known model IDs select their profile automatically; unknown IDs use general transcription and JSON grounding prompts. `--model-profile`, `LLM_MODEL_PROFILE`, or Advanced → Model preset in the web UI selects a profile explicitly when the server uses an alias. Library callers can pass `profile=` to either OCR component, and web API callers can send `model_profile`. Explicit `page_prompt`, `crop_prompt`, and grounding `prompt` values override the profile's prompts, including an intentionally empty string. Use the `generic` profile when supplying your own JSON-grounding prompt.

LightOnOCR-3 uses its native image-only transcription and `grounding` modes. Its owned adapter sends the image before the instruction. Native grounding markers are parsed on the documented 0–1000 grid and validated before use; malformed coordinates trigger the same bounded region recovery as malformed JSON. Generated image/chart descriptions are excluded from recognized text. Its native mode bypasses JSON-schema negotiation. An injected adapter retains its caller's message ordering and sampling configuration; `OpenAICompatibleClient(..., image_first=True)` supports image-first model inputs.

PaddleOCR-VL uses `OCR:` for transcription and `Spotting:` for grounded output. Spotting requires the server to preserve its special location tokens; each eight-token polygon becomes a normalized enclosing box. A server that strips the location tokens produces an explicit configuration error. Native output formats belong to the model profile and use the same HTTP adapter as JSON-capable models.

Crop transcription requests literal text rather than page-level figure descriptions or table markup. OlmOCR first reads the ordinary strip crop. If it returns unexpected mathematics or code formatting, it retries on a square paper canvas with the source ink size retained, capped at 1024 pixels. Padding every crop lost small details, so this alternate view is limited to format failures. The model preference is exposed as `OCRProcessor.crop_canvas_size`; custom processors may supply this optional attribute. Blank detection and neighbor masking happen before canvas padding, and the region deadline includes both views and truncation retries.

For model profiles without native coordinates, automatic hybrid mode uses table/diagram flags to select region OCR because linear alignment cannot reliably place cells or diagram labels. For single-column pages with at least eight detected boxes, any skipped line or box also triggers region OCR: a mostly complete sequence alignment can still shift every later field. Multi-column pages keep their page context unless fewer than half their boxes match; column wrapping alone should not replace a sound transcription with isolated-word reads.

After region OCR anchors the positions, strongly agreeing crop words can identify a unique span in the full-page transcription, including part of a longer line. That span retains context lost by isolated crops. Blank regions and short labels are not populated from page text, differing numbers are not substituted, and similarly plausible competing spans keep the crop reading. Agreement is a heuristic rather than a calibrated confidence probability; the live checks exercise both recognition and field placement.

Crop images are rendered lazily from the selected source page at up to 2048 pixels, independently of the full-page thumbnail. Custom PDF handlers may implement `render_page_for_crops(path, page_index, dpi)`; otherwise the pipeline uses their existing images. Rectified photographs keep their rectified image so crop coordinates remain in the correct frame.

Grounded pages, native-layout hybrid pages, recovery regions and detailed crop sources use lossless PNG. Other hybrid profiles retain their established JPEG preprocessing, validated with the earlier olmOCR model. Source crops render independently, avoiding enlargement of thumbnail compression artifacts. The image dimension cap applies, and the transport identifies JPEG or PNG from its bytes. Output PDF background compression is independent of inference preprocessing.

## Generation limits and grounded recovery

Inference calls have a wall-clock deadline and disable automatic SDK retries. A response ending with `finish_reason=length` raises `IncompleteCompletionError`; it is never accepted as a complete transcription. Automatic hybrid mode retries truncated pages per box. A truncated crop is retried once with the page prompt, then left unrecognized with a warning if it still exhausts its budget.

Grounded OCR normally requests one page at a time with a 4096-token limit. The Qwen3.5 profile allows 8192 tokens because dense pages can legitimately exceed 4096 tokens of line-level JSON; an explicit `max_tokens` still overrides the profile. JSON prompts request compact output without indentation. Malformed or truncated responses trigger subdivision into overlapping thirds, with at most two subdivision levels. Retries use source pixels, map coordinates back to the original page, and keep only each region's owned centers at seams. The 240-second page deadline includes recovery calls. Cancellation drains outstanding page tasks, and the pipeline rejects missing pages before writing an output. Failures are retained in `GroundedResponse.page_errors`, keyed by source index, and included in pipeline and evaluator errors.

`StructuredVisionClient` adds an optional `complete_json(..., schema=...)` capability to the base interface. The default adapter sends a standard JSON schema response format and falls back to ordinary completions when the endpoint explicitly rejects that format. Ordinary injected clients need only implement `VisionClient`; their responses are parsed and validated. The default grounding prompt explicitly requests a 0–1000 coordinate grid. Custom prompts retain automatic coordinate-convention detection.

A valid JSON response can still stop after a small fragment of a page. With the default full-page JSON prompt, boxes covering less than 20% of the source's dark ink trigger the same bounded region recovery. A small box margin allows coordinate error; uniform backgrounds and pages with too little dark ink provide no evidence. This is a conservative omission check, not a recognition-confidence score: illustrations can also trigger another view. It applies only to the initial page request, and custom prompts and native model formats retain their own scope.

Recovery regions may contain no text. Both an empty JSON array and explicit blocks with only empty or whitespace text fields are accepted for those regions. A completed empty LightOn native response, including valid coordinate markers with no text content, is also valid for a recovery region: detectors can select notebook dots or list bullets. Nonempty malformed output, truncated responses and reasoning-only responses still fail. An entire selected page without recognized text also fails.

An exactly white owned region is skipped before inference, even if its overlap contains a fragment from the adjacent region. Every nonwhite source pixel prevents this skip, including faint marks. This avoids requesting text from blank source areas that can otherwise produce long hallucinated repetitions; it does not apply a broad density threshold or make an empty selected page successful.

For Qwen3-VL and Qwen3.5 IDs, the built-in grounded adapter uses temperature 0.7, top-p 0.8 and presence penalty 1.5. Greedy decoding repeated text on dense pages during the [2026 model comparison](../evals/model-comparison-2026-10.md). Qwen3.5 also uses image-first message ordering; disable thinking in its server configuration for these OCR settings. An injected adapter controls its own optional sampling settings through `OpenAICompatibleClient(config, generation_options={...})`; the OCR component still supplies its per-request temperature and budgets. These settings depend on the model rather than the server product.

Image preprocessing in the chosen server also affects grounding accuracy. For example, Qwen recommends roughly 1024–2048 image tokens for OCR and grounding; llama.cpp exposes the lower bound as `--image-min-tokens 1024`. See the [model team's guidance](https://github.com/ggml-org/llama.cpp/issues/16842#issuecomment-3476399330). Configure the equivalent vision settings in your server and wait for model readiness before evaluation; advertising a catalog entry during loading does not establish inference readiness.

Run the live confidence evaluator with concurrency matching the endpoint's inference capacity. Its default is one request at a time. Missing pages, empty documents, failed content checks, incompatible baselines and metric regressions return nonzero; failed runs cannot update baselines.

See [evaluation provenance](../evals/README.md) for page-aware scoring, source-verified reference corrections and the limits of comparisons with historical baselines.

## Page selection contracts

`PDFHandler.convert_to_images(..., pages="2,4")` selects before rasterizing and returns images keyed by original zero-based source indices, here `1` and `3`. TIFF and other multi-frame image inputs use the same selection rule. Custom PDF handlers should accept the optional `pages` keyword and preserve source indices.

Native-layout model profiles additionally request `lossless=True` from the PDF handler. Their hybrid path calls `OCRProcessor.perform_ocr_with_layout()`, which shares parsing and recovery through `PromptedGroundedOCR.ocr_image()`. The initial page read retains the configured DPI and thumbnail cap. Only pages needing regional recovery load a second source canvas at 400 dpi, capped at 4096 pixels; every inference image still respects the request cap. On dense pages, four overlapping views supplement detection with tiny lines missed in the full-page thumbnail. Existing detections remain unchanged, and overlapping tile detections are deduplicated. Unselected pages are never rendered for OCR. Completed grounded pages release their source canvases, and rectified hybrid pages keep their existing coordinate frame.

Custom PDF handlers can implement `render_page_source(path, page_index, dpi, *, max_image_dim)` to return a detailed RGB Pillow image in the same source coordinate frame. Without this optional method, hybrid recovery uses the existing page image. An injected `layout_detector` receives a list of encoded image bytes and returns one list of normalized `[x0, y0, x1, y1]` rectangles per image; it supplies positions only. Detection and source loading run in worker threads.

Regional views can strengthen sparse, faint colored handwriting using minimum-channel ink contrast. A light-background and stroke-shape check preserves monochrome images, colored panels and filled highlights. This changes only inference pixels; output document backgrounds retain their source colors.

Local native coordinates anchor to detected source lines when spatial overlap is unambiguous. Equal counts of nearby, vertically separated lines in a single column can also use their row order to correct a small coordinate drift. This requires horizontal overlap and centers within one line height for every pair. Competing columns, missing lines and distant rows keep spatial matching.

When native paragraphs join several regular source rows, their relative heights allocate those rows and source widths distribute the original words. Inline math remains intact. This requires a nearby, ordered column with similar line heights; tables, display math, code and irregular layouts retain their existing treatment. Word placement within a restored paragraph remains approximate.

LightOnOCR-3 uses image-first messages, its native empty/`grounding` prompts, a 2048-pixel request cap, and greedy decoding. Both execution paths use the same detector-guided recovery for dense pages and omitted text inside images. Regions split at whitespace gutters while keeping neighboring lines together; crop padding stops before adjacent detected text so a cut-off fragment cannot enter the next view. Native image descriptions are excluded; only new readings of the source pixels supply diagram labels. A failed full-page read switches directly to detected regions when available. A local region gets one sampled retry before subdivision, with smaller token budgets and at most two subdivision levels. All attempts share the 240-second page deadline. An injected `layout_detector` callable can replace the default Surya detector in `PromptedGroundedOCR`; hybrid callers reuse their existing detector and already computed boxes.

Serve the official LightOn chat template with thinking disabled; llama.cpp accepts `--chat-template-file path/to/chat_template.jinja --reasoning off`. A reasoning-only response raises a configuration error instead of treating internal reasoning as recognized document text. Injected clients must supply their model's image ordering and optional sampling settings, as in the example above. Explicit custom prompts retain their own scope, and `--dense-mode always` still requests the existing per-box hybrid path.

An inline letter/subscript expression inside prose gets a second reading of the same page through native plain transcription. A uniquely matching prose line can restore an uppercase acronym with the same length and similar letters. This addresses grounding responses that invent mathematical notation for abbreviations; it does not use reference documents, a correction dictionary, or substitutions without matching source context. Numeric subscripts, equations and conflicting alternatives retain the original read. Verification shares the page deadline, and an unavailable verification request leaves the original valid blocks intact. Explicit custom prompts disable this verification.

PDF and HTML position simple table cells separately, and regional reads can anchor those cells to detected text. Plain text and searchable layers decode supported LaTeX expressions; code, currency and unknown commands remain literal. PDF font fallback preserves Unicode operators. Markdown retains markup in enclosing native blocks; tables recovered as individual source cells become separate blocks. This is an approximation of mathematical typesetting for search and extraction, not a formula renderer. The fast `--text-only` path continues to skip detection and regional recovery.

Grounded backends accept the same `pages` keyword and select before rendering or inference. A `GroundedResponse` includes `page_indices`, aligned with `page_sizes`, for every selected page even when recognition fails. Its blocks use original source indices. This lets the pipeline distinguish an unselected page from a selected page with missing text; older full-document backends may leave `page_indices=None` to retain the contiguous-page contract. The experimental `ZAIHostedOCR` skeleton rejects page selections because its service-side selection API has not been implemented.

PDF and HTML writers retain the complete document and render all background pages when writing output. Only selected pages receive new OCR text; Markdown and text writers include only processed pages. This preserves the existing output behavior while limiting OCR rendering, detection, and inference to the requested pages.

## Optional layout experiments

[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) predicts document
regions, polygons and reading order. The [local evaluation](../evals/lighton-layout-2026-10-24.md)
found no consistent line-placement improvement from using its regions to anchor
LightOn text. It remains a diagnostic option; Surya supplies the production line
anchors. A layout region detector does not itself transcribe text inside figures.

Run the standalone probe in a separate environment with `transformers>=5.5,<6`,
`torch`, `torchvision`, `Pillow` and `PyMuPDF`; the production Surya environment
does not need an upgrade. The probe saves region coordinates, reading order and
a PNG overlay without changing OCR output:

```powershell
python scripts/layout_probe.py examples/hybrid.pdf layout-results --page 1
```

The default weights are the official
[safetensors conversion](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors).
Use `--model` for a downloaded directory and `--device` to change the default CPU.
