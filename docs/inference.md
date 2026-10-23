# Configuring and extending inference

The OCR pipeline does not start, load, or manage an inference server. Run a vision model on a server of your choice and provide its OpenAI-compatible API base URL and exact model ID. The adapter uses image chat completions and, when verification is enabled, the model catalog. It does not use server-specific management APIs.

## Configuration

Set `LLM_API_BASE` and `LLM_MODEL` in the environment or a local `.env`, or pass `--api-base` and `--model`. Explicit values take precedence. An endpoint includes its API prefix, such as `http://localhost:11434/v1` or `http://localhost:8000/v1`; custom prefixes are supported. The same configuration is used for full-page, crop, and grounded OCR, and for the web model picker.

Set `LLM_API_KEY` if the endpoint requires bearer authentication. Otherwise the SDK receives a neutral placeholder, which local servers without authentication can ignore. Keys are excluded from the configuration object's representation. The CLI and web server close their adapter after each run; library callers should close adapters they create.

For web requests, the environment key is used only for the configured `LLM_API_BASE`. Changing the endpoint in the browser does not forward that secret to another server. To use an authenticated endpoint from the web UI, configure its URL and key together in the server environment.

There are no implicit endpoint or model defaults. If an older installation used the former defaults, set `LLM_API_BASE` and `LLM_MODEL` explicitly before upgrading. Existing explicit settings and the `ensure_model_loaded()` compatibility method continue to work. That method now delegates to `ensure_model_available()`: the catalog advertises usable models and does not establish whether their weights are resident in memory. Servers with incomplete or absent catalogs can use `--no-verify-model` or the web verification checkbox.

The adapter targets the vision API documented by [Ollama](https://docs.ollama.com/api/openai-compatibility), [vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/), and [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), as well as other implementations of the same protocol. Model capabilities and serving configuration still determine OCR quality. Tests exercise requests through the real SDK with an in-memory HTTP transport; they do not certify every server/model combination.

## Library use and ownership

`VisionClient` exposes two async methods: `complete(prompt, image_base64, *, model, timeout, max_tokens, temperature)` returns text, and `list_models()` returns model IDs. OCR components use this interface, so an application can supply its own transport without depending on the SDK. Injected clients remain owned by the caller.

```python
from pdf_ocr import (
    LLMConfig, OCRPipeline, OCRProcessor, OpenAICompatibleClient, PDFHandler,
)
from pdf_ocr.core.text import TextHandler


async def extract_text(input_path, output_path):
    config = LLMConfig.from_env()
    async with OpenAICompatibleClient(config) as client:
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

Prompt selection belongs to the OCR component. `OCRProcessor` uses the canonical OlmOCR prompt for model IDs containing `olmocr`, a general transcription prompt for other IDs, and an explicit `page_prompt` when supplied. `PromptedGroundedOCR` retains its separate JSON/bounding-box prompt. Server identity never determines a prompt.

## Page selection contracts

`PDFHandler.convert_to_images(..., pages="2,4")` selects before rasterizing and returns images keyed by original zero-based source indices, here `1` and `3`. TIFF and other multi-frame image inputs use the same selection rule. Custom PDF handlers should accept the optional `pages` keyword and preserve source indices.

Grounded backends accept the same `pages` keyword and select before rendering or inference. A `GroundedResponse` includes `page_indices`, aligned with `page_sizes`, for every selected page even when recognition fails. Its blocks use original source indices. This lets the pipeline distinguish an unselected page from a selected page with missing text; older full-document backends may leave `page_indices=None` to retain the contiguous-page contract. The experimental `ZAIHostedOCR` skeleton rejects page selections because its service-side selection API has not been implemented.

PDF and HTML writers retain the complete document and render all background pages when writing output. Only selected pages receive new OCR text; Markdown and text writers include only processed pages. This preserves the existing output behavior while limiting OCR rendering, detection, and inference to the requested pages.
