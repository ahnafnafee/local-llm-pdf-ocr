"""
FastAPI web server: thin wrapper around OCRPipeline with WebSocket progress.

The browser exposes the full CLI option surface (engine, model, endpoint,
and the per-path tuning knobs). This module is the trust boundary: every
bound the UI presents is *re-validated* here, because a direct caller
(curl + a client_id) never runs the frontend's checks.
"""

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from pdf_ocr import (
    HybridAligner,
    OCRPipeline,
    OCRProcessor,
    PDFHandler,
    PromptedGroundedOCR,
    SUPPORTED_FORMATS,
    media_type_for,
    resolve_output_writer,
    suffix_for_format,
)
from pdf_ocr.core.pdf import IMAGE_EXTENSIONS

# Resolve the bundled static directory relative to this module so the server
# works regardless of the user's CWD when launched via the installed
# `local-llm-pdf-ocr-server` entry point.
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# High-level progress shape sent to the browser. We translate the pipeline's
# fine-grained (stage, current, total) tuples into a single 0-100 percent.
# Grounded and text-only paths only emit a subset of these stages; the
# unmentioned ones are simply skipped, which is fine — the bar advances
# through whichever stages actually fire.
_STAGE_WEIGHTS = {
    "convert": (0, 15),    # 0-15% => PDF rasterization
    "detect": (15, 25),    # 15-25% => Surya batch detection
    "ocr": (25, 75),       # 25-75% => per-page LLM OCR + DP alignment
    "refine": (75, 90),    # 75-90% => per-box crop re-OCR (if any)
    "embed": (90, 100),    # 90-100% => PDF assembly
}


def stage_to_percent(stage: str, current: int, total: int) -> int:
    lo, hi = _STAGE_WEIGHTS.get(stage, (0, 100))
    if total <= 0:
        return lo
    return lo + int((current / total) * (hi - lo))


# --- option parsing / validation -------------------------------------------

_ENGINES = ("hybrid", "grounded", "text")
_DENSE_MODES = ("auto", "always", "never")
_PREPROCESS_MODES = ("auto", "always", "never")
_HTML_MODES = ("scaled", "letter-spacing", "full-height")
_INPUT_SUFFIXES = frozenset({".pdf"}) | IMAGE_EXTENSIONS


class _OptionError(ValueError):
    """A client-supplied option is out of range → surfaced as HTTP 400."""


def _clamp_int(value: int, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _parse_min_confidence(raw: str):
    """`""` → None (keep all boxes); otherwise a float clamped to 0..1."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        raise _OptionError(
            f"min_box_confidence must be a number in 0..1; got {raw!r}"
        )
    return max(0.0, min(1.0, v))


class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)

    async def send_progress(self, client_id: str, message: str, percent: int):
        ws = self.active.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json({"status": message, "percent": percent})
        except Exception:
            self.disconnect(client_id)


manager = ConnectionManager()

# Process-wide Surya aligner. Constructing HybridAligner loads the
# detection model (seconds of latency plus GPU/CPU memory), and the
# predictor is reusable across requests — paying that cost per upload
# serializes users behind redundant model loads. Built lazily off the
# event loop on first use; the lock keeps a burst of first requests
# from loading the model multiple times. Model/endpoint choices never
# touch this — they only construct the per-request LLM backend.
_aligner_lock = asyncio.Lock()
_shared_aligner = None


async def _get_aligner():
    global _shared_aligner
    if _shared_aligner is None:
        async with _aligner_lock:
            if _shared_aligner is None:
                _shared_aligner = await asyncio.to_thread(HybridAligner)
    return _shared_aligner


@app.get("/")
async def read_index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/models")
async def list_models(api_base: str = ""):
    """Model IDs loaded on the configured (or supplied) LLM endpoint.

    Powers the web UI's model picker. **Fails soft**: when the endpoint
    is unreachable or doesn't implement `/v1/models`, returns an empty
    list plus the error string, so the UI can still let the user type a
    model name by hand and see which endpoint it's pointed at. `api_base`
    defaults to `LLM_API_BASE` from the environment.
    """
    from openai import AsyncOpenAI

    from pdf_ocr.core.ocr import _list_loaded_model_ids

    base = api_base.strip() or os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
    default_model = os.getenv("LLM_MODEL", "")
    client = AsyncOpenAI(base_url=base, api_key="lm-studio")
    try:
        models = await _list_loaded_model_ids(client, base)
        return {"endpoint": base, "default": default_model, "models": models}
    except Exception as e:
        return {"endpoint": base, "default": default_model, "models": [], "error": str(e)}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    engine: str = Form("hybrid"),
    format: str = Form("pdf"),
    model: str = Form(""),
    api_base: str = Form(""),
    dpi: int = Form(200),
    pages: str = Form(""),
    concurrency: int = Form(0),
    max_image_dim: int = Form(1024),
    refine: bool = Form(True),
    dense_mode: str = Form("auto"),
    dense_threshold: int = Form(60),
    preprocess: str = Form("auto"),
    min_box_confidence: str = Form(""),
    html_mode: str = Form("scaled"),
    html_invert_dark: bool = Form(False),
    html_hover_text: bool = Form(False),
    verify_model: bool = Form(True),
    text_only: bool = Form(False),
):
    """OCR an uploaded file with the full CLI option surface.

    `engine` selects the pipeline path:
      * `hybrid`   — Surya layout + full-page LLM + DP alignment + refine
      * `grounded` — a bbox-native VLM returns text + coordinates in one call
      * `text`     — fast full-page plain-text dump (no Surya, no alignment)

    `text_only=true` is accepted as a synonym for `engine=text` (keeps the
    older frontend working). `model` / `api_base` override the `.env`
    defaults per request; empty strings fall back to the environment.

    Every enum and numeric bound is validated here rather than trusted
    from the client — a direct caller bypasses the browser entirely.
    """
    # --- normalize engine (text_only is the legacy synonym) ---
    if text_only:
        engine = "text"
    if engine not in _ENGINES:
        return JSONResponse(
            status_code=400,
            content={"error": f"unknown engine {engine!r}; expected one of {_ENGINES}"},
        )
    is_text = engine == "text"
    is_grounded = engine == "grounded"

    # --- validate output format + enum knobs ---
    if is_text:
        format = "txt"  # text-only always dumps plain text
    if format not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={"error": f"unsupported format {format!r}; "
                              f"expected one of {SUPPORTED_FORMATS}"},
        )
    if dense_mode not in _DENSE_MODES:
        return JSONResponse(status_code=400, content={"error": f"invalid dense_mode {dense_mode!r}"})
    if preprocess not in _PREPROCESS_MODES:
        return JSONResponse(status_code=400, content={"error": f"invalid preprocess {preprocess!r}"})
    if html_mode not in _HTML_MODES:
        return JSONResponse(status_code=400, content={"error": f"invalid html_mode {html_mode!r}"})
    try:
        min_conf = _parse_min_confidence(min_box_confidence)
    except _OptionError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    # --- clamp numeric bounds server-side (frontend limits are courtesy only) ---
    dpi = _clamp_int(dpi, 72, 600, 200)
    max_image_dim = _clamp_int(max_image_dim, 256, 4096, 1024)
    dense_threshold = _clamp_int(dense_threshold, 1, 100000, 60)
    if concurrency and concurrency > 0:
        concurrency = max(1, min(16, concurrency))
    else:
        concurrency = max(1, min(16, int(os.getenv("OCR_CONCURRENCY", 2))))

    model_arg = model.strip() or None
    api_base_arg = api_base.strip() or None
    pages_arg = pages.strip() or None

    output_suffix = suffix_for_format(format)

    # Preserve the real input extension so PDFHandler routes images to its
    # image path instead of trying to parse them as a PDF (issue: images
    # were previously saved with a hardcoded `.pdf` suffix and misrouted).
    in_suffix = Path(file.filename or "").suffix.lower()
    if in_suffix not in _INPUT_SUFFIXES:
        in_suffix = ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix) as tmp_input:
        shutil.copyfileobj(file.file, tmp_input)
        input_path = tmp_input.name
    output_path = os.path.join(
        tempfile.gettempdir(), f"output_{uuid.uuid4()}{output_suffix}",
    )

    # HTML in the web path must be a single self-contained file — the
    # FileResponse can't ship sidecar JPEGs to the browser — so page
    # images are always inlined. The other HTML knobs are honored.
    output_writer = resolve_output_writer(
        output_path,
        html_mode=html_mode,
        html_inline_images=True,
        html_invert_dark=html_invert_dark,
        html_hover_text=html_hover_text,
    )

    try:
        await manager.send_progress(client_id, "Initializing...", 5)

        if is_text:
            # Fast path: no Surya — skip the (lazy) aligner load entirely.
            pipeline = OCRPipeline(
                ocr_processor=OCRProcessor(api_base=api_base_arg, model=model_arg),
                pdf_handler=PDFHandler(),
                output_writer=output_writer,
            )
        elif is_grounded:
            # Grounded path: bbox-native VLM returns text + positions in one
            # call. No Surya, no DP, no refine.
            pipeline = OCRPipeline(
                pdf_handler=PDFHandler(),
                grounded_backend=PromptedGroundedOCR(
                    api_base=api_base_arg, model=model_arg, max_image_dim=max_image_dim,
                ),
                output_writer=output_writer,
            )
        else:
            pipeline = OCRPipeline(
                aligner=await _get_aligner(),
                ocr_processor=OCRProcessor(api_base=api_base_arg, model=model_arg),
                pdf_handler=PDFHandler(),
                output_writer=output_writer,
            )

        # Fail fast on model mismatch (issue #7). The client can opt out per
        # request; OCR_VERIFY_MODEL=0 forces it off server-wide (for servers
        # that don't expose /v1/models).
        if verify_model and os.getenv("OCR_VERIFY_MODEL", "1") != "0":
            backend = getattr(pipeline, "grounded_backend", None) or pipeline.ocr_processor
            await backend.ensure_model_loaded()

        async def on_progress(stage, current, total, message):
            await manager.send_progress(client_id, message, stage_to_percent(stage, current, total))

        pages_text = await pipeline.run(
            input_path, output_path,
            dpi=dpi,
            pages=pages_arg,
            concurrency=concurrency,
            refine=refine,
            max_image_dim=max_image_dim,
            dense_threshold=dense_threshold,
            dense_mode=dense_mode,
            preprocess=preprocess,
            min_box_confidence=min_conf,
            text_only=is_text,
            progress=on_progress,
        )

        # Save per-page raw LLM text so the UI's "View Text" preview can fetch it.
        text_path = os.path.join(tempfile.gettempdir(), f"text_{client_id}.json")
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pages_text.items()}, f)

        await manager.send_progress(client_id, "Done! Preparing download...", 100)
        # Strip the original extension from `file.filename` so the download
        # is named with the chosen format's suffix instead of ending up like
        # `ocr_scan.pdf.html`.
        base_name = Path(file.filename or "ocr_output").stem
        return FileResponse(
            output_path,
            media_type=media_type_for(output_path),
            filename=f"ocr_{base_name}{output_suffix}",
            background=BackgroundTask(_cleanup, input_path),
        )
    except Exception as e:
        await manager.send_progress(client_id, f"Error: {e}", 0)
        _cleanup(input_path)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/text/{job_id}")
async def get_text(job_id: str):
    text_path = os.path.join(tempfile.gettempdir(), f"text_{job_id}.json")
    if os.path.exists(text_path):
        return FileResponse(text_path, media_type="application/json")
    return JSONResponse(status_code=404, content={"error": "Text not found"})


def _cleanup(*paths):
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def main() -> None:
    """Entry point for the `local-llm-pdf-ocr-server` console script."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Local LLM PDF OCR web server (FastAPI + WebSocket progress).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    args = parser.parse_args()

    uvicorn.run(
        "pdf_ocr.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
