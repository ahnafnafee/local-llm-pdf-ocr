from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import shutil
import os
import uuid
import json
import asyncio
from pathlib import Path
from pdf_utils import PDFHandler
from ocr import OCRProcessor
from hybrid_aligner import HybridAligner
import tempfile
import base64

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_progress(self, client_id: str, message: str, percent: int):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({
                    "status": message,
                    "percent": percent
                })
            except:
                self.disconnect(client_id)

manager = ConnectionManager()

# Serve index at root
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...), 
    client_id: str = Form(...)
):
    # 1. Save uploaded file
    # Create temp file in system temp dir
    
    # Input file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
        shutil.copyfileobj(file.file, tmp_input)
        input_path = tmp_input.name
        
    # Output file path (just a name, we'll create it later)
    # We use mkstemp to reserve a name, or just derive it. 
    # Let's derive it to keep it simple but safe.
    output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4()}.pdf")

        
    try:
        await manager.send_progress(client_id, "Initializing...", 5)

        # 2. Initialize orchestration
        pdf_handler = PDFHandler()
        ocr_processor = OCRProcessor()
        
        await manager.send_progress(client_id, "Converting PDF to images...", 10)

        # 3. Convert to Images
        # This might block event loop if large, but for local tool it's okay-ish. 
        # Ideally run in threadpool but PDFHandler is blocking.
        images_dict = pdf_handler.convert_to_images(input_path)
        
        page_nums = sorted(images_dict.keys())
        total_pages = len(page_nums)
        
        await manager.send_progress(client_id, f"Converted {total_pages} pages. Starting OCR...", 20)
        
        
        # 4. Perform Hybrid OCR
        hybrid_aligner = HybridAligner()
        
        pages_text_for_preview = {} # Clean LLM text
        pages_structured_for_pdf = {} # SuryaOCR boxes
        
        for idx, page_num in enumerate(page_nums):
            image_base64 = images_dict[page_num]
            # Decode for SuryaOCR which needs bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Progress update per page
            current_percent = 20 + int((idx / total_pages) * 70) 
            await manager.send_progress(
                client_id, 
                f"Processing Page {page_num + 1}/{total_pages} (OCR + Layout)...", 
                current_percent
            )
            
            # Parallel-ish: Run LLM for meaning, run SuryaOCR for layout
            # 1. LLM OCR (for preview)
            # Perform OCR (LLM + Hybrid)
            # We'll use the Hybrid/SuryaOCR text for the preview as well, 
            # because it's faster and cleaner than waiting for the LLM if the user just wants to see what's there.
            
            structured_data = await asyncio.to_thread(hybrid_aligner.get_structured_text, image_bytes)
            pages_structured_for_pdf[page_num] = structured_data
            
            # Extract plain text from structured data for the preview
            # structured_data is list of ([coords], text)
            page_text_lines = [item[1] for item in structured_data]
            pages_text_for_preview[page_num] = page_text_lines
            
            # 2. LLM OCR (Content)
            # Crucial for Hybrid Alignment
            llm_lines = await asyncio.to_thread(ocr_processor.perform_ocr, image_base64)
            
            if llm_lines:
                 # Update preview with better text
                 pages_text_for_preview[page_num] = llm_lines
            
            # --- Perform Alignment ---
            # Try to improve the SuryaOCR text (which might be garbled) using the LLM text
            if llm_lines:
                aligned_structure = await asyncio.to_thread(hybrid_aligner.align_text, structured_data, llm_lines)
                pages_structured_for_pdf[page_num] = aligned_structure
            else:
                pages_structured_for_pdf[page_num] = structured_data
            
        await manager.send_progress(client_id, "Embedding text into PDF...", 90)

        # 5. Embed Text (Structured)
        # We use the aligned data if available, otherwise raw SuryaOCR
        pdf_handler.embed_structured_text(input_path, output_path, pages_structured_for_pdf)
        
        # Save text for preview (Clean LLM text)
        text_path = os.path.join(tempfile.gettempdir(), f"text_{client_id}.json")
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump(pages_text_for_preview, f)
        
        await manager.send_progress(client_id, "Done! Preparing download...", 100)
        
        # 6. Return Result
        return FileResponse(
            output_path, 
            media_type="application/pdf", 
            filename=f"ocr_{file.filename}",
            # We don't auto-clean output/text immediately so user can fetch text/download again if we added that. 
            # But here we clean input. Output/Text clean via cron or just OS temp clean. 
            # Actually, let's keep it simple.
            background=BackgroundTask(cleanup, input_path) 
        )
        
    except Exception as e:
        await manager.send_progress(client_id, f"Error: {str(e)}", 0)
        cleanup(input_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/text/{job_id}")
async def get_text(job_id: str):
    text_path = os.path.join(tempfile.gettempdir(), f"text_{job_id}.json")
    if os.path.exists(text_path):
        return FileResponse(text_path, media_type="application/json")
    return JSONResponse(status_code=404, content={"error": "Text not found"})

def cleanup(*paths):
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass


