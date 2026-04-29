#!/usr/bin/env python3
"""
Local LLM PDF OCR - Command Line Interface.

Process PDF documents through a local LLM vision model to create searchable PDFs.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR with Local LLM - turn scanned PDFs or raw images into searchable PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.pdf output_ocr.pdf
  %(prog)s scan.png                       # Image input - auto-generates scan_ocr.pdf
  %(prog)s pages.tiff                     # Multi-frame TIFF - one output page per frame
  %(prog)s input.pdf                      # Auto-generates input_ocr.pdf
  %(prog)s input.pdf output.pdf --verbose
  %(prog)s input.pdf output.pdf --pages 1-3,5
  %(prog)s input.pdf output.pdf --dpi 300 --api-base http://localhost:1234/v1
        """,
    )
    parser.add_argument(
        "input_pdf",  # kwarg name kept for internal stability; accepts PDFs *and* images.
        metavar="input",
        help="Path to a PDF or image file (JPEG/PNG/TIFF/BMP/WebP/AVIF). "
             "Multi-frame TIFFs expand to multiple output pages.",
    )
    parser.add_argument(
        "output_pdf", nargs="?",
        metavar="output",
        help="Path to output PDF (always a PDF, even for image inputs; "
             "defaults to <input_stem>_ocr.pdf).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for image rendering (default: 200)")
    parser.add_argument("--pages", help="Page range to process, e.g., '1-3,5' (default: all)")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel LLM requests (default: 1)")
    parser.add_argument(
        "--no-refine", dest="refine", action="store_false",
        help="Skip per-box crop re-OCR for low-confidence boxes (faster, less accurate on complex layouts)",
    )
    parser.add_argument(
        "--max-image-dim", type=int, default=1024,
        help="Longest-edge px cap for page images sent to the LLM (default: 1024). "
             "Drop to ~640 for small local VLMs like GLM-OCR:1.1B that crash on larger inputs.",
    )
    parser.add_argument(
        "--dense-threshold", type=int, default=60,
        help="In --dense-mode auto, pages with more than this many detected boxes "
             "use per-box OCR instead of full-page (default: 60). Per-box is more "
             "accurate on dense handwritten content where the LLM otherwise loops "
             "or hallucinates.",
    )
    parser.add_argument(
        "--dense-mode", choices=("auto", "always", "never"), default="auto",
        help="auto (default): per-box OCR for pages above --dense-threshold. "
             "always: per-box for every page (slow but most accurate). "
             "never: original full-page OCR everywhere.",
    )
    parser.add_argument(
        "--grounded", action="store_true",
        help="Use a bbox-native VLM (Qwen2.5-VL / Qwen3-VL / etc.) that returns text WITH "
             "bounding boxes in one call. Skips Surya + DP + refine. Requires --model to be "
             "a vision LLM that supports grounded output.",
    )
    parser.add_argument("--api-base", help="Override LLM API base URL")
    parser.add_argument("--model", help="Override LLM model name")
    parser.set_defaults(refine=True)
    return parser


def configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def resolve_output_path(input_pdf: str, output_pdf: str | None) -> str:
    if output_pdf:
        return output_pdf
    p = Path(input_pdf)
    return str(p.parent / f"{p.stem}_ocr.pdf")


async def run(args: argparse.Namespace, console: Console) -> None:
    # Lazy imports: heavy modules load AFTER argparse so --help stays fast.
    os.environ.setdefault("TQDM_DISABLE", "1")
    from pdf_ocr import (
        HybridAligner, OCRPipeline, OCRProcessor, PDFHandler,
        PromptedGroundedOCR,
    )

    pdf_handler = PDFHandler()
    if args.grounded:
        # Grounded path: bbox-native VLM returns text + positions in one call.
        backend_kwargs = {"max_image_dim": args.max_image_dim}
        if args.api_base:
            backend_kwargs["api_base"] = args.api_base
        if args.model:
            backend_kwargs["model"] = args.model
        pipeline = OCRPipeline(
            pdf_handler=pdf_handler,
            grounded_backend=PromptedGroundedOCR(**backend_kwargs),
        )
    else:
        pipeline = OCRPipeline(
            aligner=HybridAligner(),
            ocr_processor=OCRProcessor(api_base=args.api_base, model=args.model),
            pdf_handler=pdf_handler,
        )

    output_path = resolve_output_path(args.input_pdf, args.output_pdf)
    console.print(f"[bold cyan]Processing '{args.input_pdf}'...[/bold cyan]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        tasks: dict[str, int] = {}

        async def on_progress(stage: str, current: int, total: int, message: str) -> None:
            if stage not in tasks:
                tasks[stage] = progress.add_task(f"[cyan]{message}", total=total)
            progress.update(tasks[stage], total=total, completed=current, description=f"[cyan]{message}")

        try:
            await pipeline.run(
                args.input_pdf, output_path,
                dpi=args.dpi, pages=args.pages,
                concurrency=args.concurrency,
                refine=args.refine,
                max_image_dim=args.max_image_dim,
                dense_threshold=args.dense_threshold,
                dense_mode=args.dense_mode,
                progress=on_progress,
            )
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise

    console.print(f"[bold green]Done! Saved to '{output_path}'[/bold green]")


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose, args.quiet)
    console = Console(quiet=args.quiet)
    try:
        asyncio.run(run(args, console))
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
