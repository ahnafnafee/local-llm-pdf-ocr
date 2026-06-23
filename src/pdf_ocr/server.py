"""
FastAPI web server: thin wrapper around OCRPipeline with WebSocket progress.
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
    SUPPORTED_FORMATS,
    media_type_for,
    resolve_output_writer,
    suffix_for_format,
)

# Resolve the bundled static directory relative to this module so the server
# works regardless of the user's CWD when launched via the installed
# `local-llm-pdf-ocr-server` entry point.
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# High-level progress shape sent to the browser. We translate the pipeline's
# fine-grained (stage, current, total) tuples into a single 0-100 percent.
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
# from loading the model multiple times.
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
    format: str = Form("pdf"),
    text_only: bool = Form(False),
):
    """OCR an uploaded file and return the result in `format`.

    `format` is one of `pdf` (default), `html`, `md`, `txt`. The file is
    named `ocr_<original-name-with-original-extension>.<format-suffix>`
    for download, and the response's Content-Type tracks the format.

    `text_only` is the fast path: it skips Surya layout detection, DP
    alignment, and crop re-OCR, OCR'ing each page's full text and dumping
    it as plain text. It forces `format` to `txt`.
    """
    if text_only:
        format = "txt"
    if format not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={"error": f"unsupported format {format!r}; "
                              f"expected one of {SUPPORTED_FORMATS}"},
        )
    output_suffix = suffix_for_format(format)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
        shutil.copyfileobj(file.file, tmp_input)
        input_path = tmp_input.name
    output_path = os.path.join(
        tempfile.gettempdir(), f"output_{uuid.uuid4()}{output_suffix}",
    )

    try:
        await manager.send_progress(client_id, "Initializing...", 5)

        if text_only:
            # Fast path: no Surya — skip the (lazy) aligner load entirely.
            pipeline = OCRPipeline(
                ocr_processor=OCRProcessor(),
                pdf_handler=PDFHandler(),
                output_writer=resolve_output_writer(output_path),
            )
        else:
            pipeline = OCRPipeline(
                aligner=await _get_aligner(),
                ocr_processor=OCRProcessor(),
                pdf_handler=PDFHandler(),
                # Inline images so the single-file FileResponse below is
                # self-contained — sidecar JPEGs would not reach the client.
                output_writer=resolve_output_writer(
                    output_path, html_inline_images=True,
                ),
            )
        # Conservative default: in-flight requests cost KV-cache VRAM on
        # parallel-slot servers (vLLM, num_parallel>1); queuing servers
        # (LM Studio / Ollama defaults) hold extras for free. Raise via
        # OCR_CONCURRENCY when the serving side has headroom.
        concurrency = int(os.getenv("OCR_CONCURRENCY", 2))

        # Fail fast on model mismatch (issue #7). Set OCR_VERIFY_MODEL=0 to
        # skip if your server doesn't expose /v1/models.
        if os.getenv("OCR_VERIFY_MODEL", "1") != "0":
            await pipeline.ocr_processor.ensure_model_loaded()

        async def on_progress(stage, current, total, message):
            await manager.send_progress(client_id, message, stage_to_percent(stage, current, total))

        pages_text = await pipeline.run(
            input_path, output_path,
            concurrency=concurrency, text_only=text_only, progress=on_progress,
        )

        # Save per-page raw LLM text so the UI's "View Text" preview can fetch it.
        text_path = os.path.join(tempfile.gettempdir(), f"text_{client_id}.json")
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pages_text.items()}, f)

        await manager.send_progress(client_id, "Done! Preparing download...", 100)
        # Strip the original extension from `file.filename` so the
        # download is named with the chosen format's suffix instead of
        # ending up like `ocr_scan.pdf.html`.
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
