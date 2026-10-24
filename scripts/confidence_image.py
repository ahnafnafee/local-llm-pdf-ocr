#!/usr/bin/env python3
"""
Confidence comparison: hybrid (no --grounded) vs grounded (--grounded)
on a single image input, scored against a hand-built ground-truth fixture.

Usage:
    uv run scripts/confidence_image.py
    uv run scripts/confidence_image.py --image examples/image.avif
    uv run scripts/confidence_image.py --hybrid-model your-served-model \
        --grounded-model your-served-model

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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console safety

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from pdf_ocr.evaluation import (  # noqa: E402
    compute_report,
    load_ground_truth,
)
from pdf_ocr.inference import DEFAULT_API_BASE, DEFAULT_MODEL, DEFAULT_GROUNDED_MODEL  # noqa: E402


from scripts.confidence_eval import run_grounded, run_hybrid  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "examples" / "image.avif"))
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "tests" / "fixtures" / "ground_truth_image.json"),
    )
    parser.add_argument(
        "--api-base", default=None,
        help=f"API base URL; defaults to LLM_API_BASE or {DEFAULT_API_BASE}",
    )
    parser.add_argument("--hybrid-model", default=DEFAULT_MODEL)
    parser.add_argument("--grounded-model", default=DEFAULT_GROUNDED_MODEL)
    parser.add_argument("--max-image-dim", type=int, default=2048)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    api_base = args.api_base or os.getenv("LLM_API_BASE", DEFAULT_API_BASE)
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
    failures = False

    console.print("[cyan]hybrid (no --grounded)...[/]")
    try:
        out = await run_hybrid(image, api_base, args.hybrid_model, args.max_image_dim)
        if not out:
            raise RuntimeError("OCR returned no text blocks")
        report = compute_report(image.name, gt, out, iou_threshold=args.iou_threshold,
                                pipeline_page_indices=getattr(out, "page_indices", None))
        reports.append(("hybrid", report))
        console.print(f"   {report.summary_line()}")
    except Exception as e:
        failures = True
        console.print(f"   [red]hybrid failed: {type(e).__name__}: {e}[/]")

    console.print("\n[cyan]grounded (--grounded)...[/]")
    try:
        out = await run_grounded(image, api_base, args.grounded_model, args.max_image_dim)
        if not out:
            raise RuntimeError("OCR returned no text blocks")
        report = compute_report(image.name, gt, out, iou_threshold=args.iou_threshold,
                                pipeline_page_indices=getattr(out, "page_indices", None))
        reports.append(("grounded", report))
        console.print(f"   {report.summary_line()}")
    except Exception as e:
        failures = True
        console.print(f"   [red]grounded failed: {type(e).__name__}: {e}[/]")

    if not reports:
        console.print("[red]Both paths failed; nothing to report.[/]")
        return 1

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

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
