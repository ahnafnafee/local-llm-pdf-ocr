"""Upload an actual PDF through the running web API and inspect its output."""

import argparse
import json
import sys
import time
from pathlib import Path

import fitz
import httpx

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["hybrid", "grounded"], required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--use-default", action="store_true")
args = parser.parse_args()
started = time.monotonic()
with httpx.Client(base_url="http://127.0.0.1:28129", timeout=240, trust_env=False) as client:
    catalog = client.get("/models", params={"api_base": "http://127.0.0.1:28127/v1"})
    catalog.raise_for_status()
    assert args.model in catalog.json()["models"], catalog.text
    if args.use_default:
        assert catalog.json()["default"] == args.model, catalog.text
    with (ROOT / "examples/multipage.pdf").open("rb") as source:
        response = client.post("/process", files={"file": ("multipage.pdf", source, "application/pdf")}, data={
            "client_id": "confidence-e2e", "engine": args.mode, "format": "pdf",
            "api_base": "http://127.0.0.1:28127/v1", "model": "" if args.use_default else args.model,
            "pages": "2", "dpi": "72", "concurrency": "1", "refine": "false",
            "dense_mode": "never", "preprocess": "never", "verify_model": "true",
        })
    response.raise_for_status()

output = OUTPUT / f"web-{args.mode}-selected-page.pdf"
output.write_bytes(response.content)
with fitz.open(stream=response.content, filetype="pdf") as pdf:
    page_text = [page.get_text() for page in pdf]
result = {
    "mode": args.mode, "model": args.model, "status_code": response.status_code,
    "explicit_model_argument": not args.use_default,
    "content_type": response.headers.get("content-type"), "output_pages": len(page_text),
    "text_characters_per_page": [len(text.strip()) for text in page_text],
    "elapsed_seconds": round(time.monotonic() - started, 2), "output": str(output),
    "passed": len(page_text) == 2 and not page_text[0].strip() and len(page_text[1].strip()) > 100,
}
(OUTPUT / f"web-{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
