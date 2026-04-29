#!/usr/bin/env python3
"""
Build a ground-truth JSON fixture from a grounded VLM run.

Used to bootstrap fixtures for examples that are too dense to hand-build
(dense.pdf, notes.pdf, ...). The grounded path produces accurate
(bbox, text) pairs in one shot which we serialize into the fixture
format that scripts/confidence_eval.py understands. The fixture is then
useful for *regression* testing — confidence drops on future runs flag
that something got worse, even if the absolute baseline is biased
toward whichever model produced the fixture.

Usage:
    uv run scripts/build_fixture.py examples/dense.pdf tests/fixtures/ground_truth_dense.json
    uv run scripts/build_fixture.py examples/notes.pdf tests/fixtures/ground_truth_notes.json \\
        --model qwen/qwen3-vl-4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_ocr import PromptedGroundedOCR  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input PDF or image")
    parser.add_argument("output", help="Output fixture JSON path")
    parser.add_argument(
        "--api-base", default=None,
        help="Defaults to LLM_API_BASE env var, then localhost:1234",
    )
    parser.add_argument("--model", default="qwen/qwen3-vl-4b")
    parser.add_argument("--max-image-dim", type=int, default=1024)
    args = parser.parse_args()

    api_base = args.api_base or os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"Building fixture from {in_path.name} via grounded ({args.model})")
    print(f"  api_base={api_base}")

    backend = PromptedGroundedOCR(
        api_base=api_base, model=args.model,
        max_image_dim=args.max_image_dim, max_tokens=8192, concurrency=3,
    )
    response = await backend.ocr_document(str(in_path))

    if not response.blocks:
        print(f"  ERROR: grounded returned 0 blocks; refusing to write empty fixture")
        sys.exit(1)

    layout = []
    block_id = 0
    for b in response.blocks:
        # GroundedBlock.bbox is normalized [nx0, ny0, nx1, ny1].
        nx0, ny0, nx1, ny1 = b.bbox
        pw, ph = response.page_sizes[b.page_index]
        # Fixture format used by digital/hybrid: [y0, x0, y1, x1] in pixels.
        bbox_yx = [
            int(round(ny0 * ph)),
            int(round(nx0 * pw)),
            int(round(ny1 * ph)),
            int(round(nx1 * pw)),
        ]
        layout.append({
            "block_content": b.text,
            "bbox": bbox_yx,
            "block_id": block_id,
            "page_index": b.page_index,
            "block_label": b.label or "text",
            "score": 1.0,
        })
        block_id += 1

    fixture = {
        "data": {
            "file_name": in_path.name,
            "file_type": "pdf" if in_path.suffix.lower() == ".pdf" else "image",
            "layout": layout,
            "data_info": {
                "pages": [{"width": int(w), "height": int(h)} for (w, h) in response.page_sizes],
                "num_pages": len(response.page_sizes),
            },
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)

    print(f"  wrote {len(layout)} blocks across {len(response.page_sizes)} pages")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
