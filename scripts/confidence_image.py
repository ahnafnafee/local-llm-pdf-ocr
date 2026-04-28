#!/usr/bin/env python3
"""
Confidence comparison: hybrid (no --grounded) vs grounded (--grounded)
on a single image input, scored against a hand-built ground-truth fixture.

Usage:
    uv run scripts/confidence_image.py
    uv run scripts/confidence_image.py --image examples/image.avif
    uv run scripts/confidence_image.py --hybrid-model allenai/olmocr-2-7b \
        --grounded-model qwen/qwen3-vl-4b

Reports per-path block recall (matched GT blocks at IoU >= threshold),
average IoU of matched pairs, and average text similarity. Prints the
worst unmatched ground-truth blocks for each path so we can see WHICH
content each path drops.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.stdout.reconfigure(encoding="utf-8")  # Windows console safety

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from src.pdf_ocr import (  # noqa: E402
    HybridAligner,
    OCRPipeline,
    OCRProcessor,
    PDFHandler,
    PromptedGroundedOCR,
)
from src.pdf_ocr.evaluation import (  # noqa: E402
    compute_report,
    load_ground_truth,
)


async def run_grounded(image: Path, api_base: str, model: str, max_dim: int):
    backend = PromptedGroundedOCR(
        api_base=api_base, model=model,
        max_image_dim=max_dim, max_tokens=8192,
    )
    response = await backend.ocr_document(str(image))
    return [(b.bbox, b.text) for b in response.blocks]


async def run_hybrid(image: Path, api_base: str, model: str, max_dim: int):
    captured: dict = {}

    def capture(_in, _out, pages_data, _dpi):
        captured["pages_data"] = pages_data

    pipe = OCRPipeline(
        aligner=HybridAligner(),
        ocr_processor=OCRProcessor(api_base=api_base, model=model),
        pdf_handler=PDFHandler(),
        output_writer=capture,
    )
    # Throwaway output path — we use the captured in-memory data.
    await pipe.run(
        str(image), str(image.with_suffix(".scoring.pdf")),
        max_image_dim=max_dim, concurrency=3, refine=True,
    )
    blocks = []
    for _page_num, page_blocks in captured.get("pages_data", {}).items():
        for bbox, text in page_blocks:
            if text.strip():
                blocks.append((bbox, text))
    return blocks


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "examples" / "image.avif"))
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "tests" / "fixtures" / "ground_truth_image.json"),
    )
    parser.add_argument(
        "--api-base", default=None,
        help="Defaults to LLM_API_BASE env var, then localhost:1234",
    )
    parser.add_argument("--hybrid-model", default="allenai/olmocr-2-7b")
    parser.add_argument("--grounded-model", default="qwen/qwen3-vl-4b")
    parser.add_argument("--max-image-dim", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    api_base = args.api_base or os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
    image = Path(args.image)
    fixture = Path(args.fixture)
    console = Console()

    gt, (fw, fh) = load_ground_truth(fixture)
    console.print(
        f"[bold]>> {image.name}[/]  GT: {len(gt)} blocks, ref dims {fw}x{fh}\n"
        f"   api_base={api_base}\n"
        f"   hybrid_model={args.hybrid_model}\n"
        f"   grounded_model={args.grounded_model}\n"
    )

    reports: list[tuple[str, object]] = []

    console.print("[cyan]hybrid (no --grounded)...[/]")
    try:
        out = await run_hybrid(image, api_base, args.hybrid_model, args.max_image_dim)
        report = compute_report(image.name, gt, out, iou_threshold=args.iou_threshold)
        reports.append(("hybrid", report))
        console.print(f"   {report.summary_line()}")
    except Exception as e:
        console.print(f"   [red]hybrid failed: {type(e).__name__}: {e}[/]")

    console.print("\n[cyan]grounded (--grounded)...[/]")
    try:
        out = await run_grounded(image, api_base, args.grounded_model, args.max_image_dim)
        report = compute_report(image.name, gt, out, iou_threshold=args.iou_threshold)
        reports.append(("grounded", report))
        console.print(f"   {report.summary_line()}")
    except Exception as e:
        console.print(f"   [red]grounded failed: {type(e).__name__}: {e}[/]")

    if not reports:
        console.print("[red]Both paths failed; nothing to report.[/]")
        return

    table = Table(title="Confidence: hybrid vs grounded on " + image.name, show_lines=True)
    table.add_column("path")
    table.add_column("GT", justify="right")
    table.add_column("Pipeline", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Avg IoU", justify="right")
    table.add_column("Avg TextSim", justify="right")

    for path, report in reports:
        table.add_row(
            path,
            str(report.gt_count),
            str(report.pipeline_count),
            str(len(report.matched)),
            f"{report.block_recall:.2f}",
            f"{report.avg_iou:.2f}",
            f"{report.avg_text_similarity:.2f}",
        )
    console.print()
    console.print(table)

    for path, report in reports:
        unmatched = [m for m in report.matches if m.iou < report.iou_threshold]
        if not unmatched:
            continue
        console.print(f"\n[yellow]Unmatched GT blocks ({path}):[/]")
        for m in unmatched[:10]:
            snippet = m.gt_text[:60].replace("\n", " ")
            console.print(f"  - {snippet!r}  (best_iou={m.iou:.2f})")
        if len(unmatched) > 10:
            console.print(f"  ... and {len(unmatched) - 10} more.")


if __name__ == "__main__":
    asyncio.run(main())
