"""Trace real CLI rendering and inference on a long source document."""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import fitz
from PIL import Image
from pdf_ocr.cli import main
from pdf_ocr.inference import OpenAICompatibleClient

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["hybrid", "text", "grounded"], required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--use-default", action="store_true")
args = parser.parse_args()
source = OUTPUT / "selection-source-130-pages.pdf"
if not source.exists():
    with fitz.open(ROOT / "examples/digital.pdf") as original, fitz.open() as doc:
        for _ in range(130):
            doc.insert_pdf(original, from_page=0, to_page=0)
        doc.save(source, garbage=4, deflate=True)

rendered = []
render_events = []
requests = []
full_page_hashes = set()
original_render = fitz.Page.get_pixmap
original_complete = OpenAICompatibleClient._complete


def traced_render(page, *positional, **keywords):
    rendered.append(page.number)
    pixmap = original_render(page, *positional, **keywords)
    render_events.append({"source_page_index": page.number, "dpi": keywords.get("dpi"),
                          "size": [pixmap.width, pixmap.height]})
    if args.mode in {"grounded", "hybrid"}:
        expected = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        expected.thumbnail((2048, 2048))
        full_page_hashes.add(hashlib.sha256(expected.tobytes()).hexdigest())
    return pixmap


async def traced_complete(client, *positional, **keywords):
    with Image.open(io.BytesIO(base64.b64decode(positional[1]))) as request_image:
        requests.append({"model": keywords["model"], "image_size": list(request_image.size),
                         "pixel_sha256": hashlib.sha256(request_image.convert("RGB").tobytes()).hexdigest()})
    return await original_complete(client, *positional, **keywords)


fitz.Page.get_pixmap = traced_render
OpenAICompatibleClient._complete = traced_complete
output = OUTPUT / f"selection-{args.mode}.md"
sys.argv = [
    "local-llm-pdf-ocr", str(source), str(output),
    "--api-base", "http://127.0.0.1:28127/v1",
    "--pages", "2,4", "--concurrency", "1", "--no-refine", "--dense-mode", "never",
]
if not args.use_default:
    sys.argv.extend(["--model", args.model])
if args.mode == "text":
    sys.argv.append("--text-only")
elif args.mode == "grounded":
    sys.argv.append("--grounded")
start = time.monotonic()
exit_code = 0
try:
    main()
except SystemExit as exc:
    exit_code = exc.code
finally:
    fitz.Page.get_pixmap = original_render
    OpenAICompatibleClient._complete = original_complete

text = output.read_text(encoding="utf-8") if output.exists() else ""
output_pages = [int(page) for page in re.findall(r"^## Page (\d+)$", text, re.MULTILINE)]
result = {
    "mode": args.mode, "model": args.model, "source_pages": 130,
    "explicit_model_argument": not args.use_default,
    "selection": "2,4", "rendered_zero_based_indices": rendered,
    "render_events": render_events,
    "real_inference_requests": len(requests), "exit_code": exit_code,
    "inference_requests": requests,
    "elapsed_seconds": round(time.monotonic() - start, 2),
    "output": str(output), "output_characters": len(text),
    "output_page_numbers": output_pages,
}
# Identify the initial requests by source-pixel hashes, since a smaller region
# can coincidentally have the same aspect ratio as a whole page.
full_page_requests = (sum(request["pixel_sha256"] in full_page_hashes for request in requests)
                      if args.mode in {"grounded", "hybrid"} else len(requests))
result["full_page_inference_requests"] = full_page_requests
result["recovery_inference_requests"] = len(requests) - full_page_requests
request_counts_valid = (full_page_requests == 2 and 2 <= len(requests) <= 26
                        if args.mode in {"grounded", "hybrid"} else len(requests) == 2)
# Native recovery can lazily render each selected page once more at 400 dpi.
# Keep the selection guarantee strict: no unselected page may be rasterized.
render_counts_valid = (set(rendered) == {1, 3}
                       and all(rendered.count(page) <= 2 for page in {1, 3})
                       and all(event["dpi"] == 400 for event in render_events[2:]))
result["passed"] = (exit_code == 0 and render_counts_valid
                    and request_counts_valid and output_pages == [2, 4] and "CS 701" in text
                    and all(request["model"] == args.model for request in requests))
(OUTPUT / f"selection-{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
