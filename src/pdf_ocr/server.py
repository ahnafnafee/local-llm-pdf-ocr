"""
FastAPI web server: thin wrapper around OCRPipeline with WebSocket progress.
"""

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

from pdf_ocr import HybridAligner, OCRPipeline, OCRProcessor, PDFHandler

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
async def process_pdf(file: UploadFile = File(...), client_id: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
        shutil.copyfileobj(file.file, tmp_input)
        input_path = tmp_input.name
    output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4()}.pdf")

    try:
        await manager.send_progress(client_id, "Initializing...", 5)

        pipeline = OCRPipeline(
            aligner=HybridAligner(),
            ocr_processor=OCRProcessor(),
            pdf_handler=PDFHandler(),
        )
        concurrency = int(os.getenv("OCR_CONCURRENCY", 3))

        async def on_progress(stage, current, total, message):
            await manager.send_progress(client_id, message, stage_to_percent(stage, current, total))

        pages_text = await pipeline.run(
            input_path, output_path,
            concurrency=concurrency, progress=on_progress,
        )

        # Save per-page raw LLM text so the UI's "View Text" preview can fetch it.
        text_path = os.path.join(tempfile.gettempdir(), f"text_{client_id}.json")
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pages_text.items()}, f)

        await manager.send_progress(client_id, "Done! Preparing download...", 100)
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"ocr_{file.filename}",
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
