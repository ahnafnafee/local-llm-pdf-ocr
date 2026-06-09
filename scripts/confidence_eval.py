#!/usr/bin/env python3
"""
Confidence evaluation: run each pipeline path against the example PDFs and
score the output against the ground-truth fixtures.

Usage:
    uv run scripts/confidence_eval.py                            # both paths
    uv run scripts/confidence_eval.py --path grounded            # just grounded
    uv run scripts/confidence_eval.py --path hybrid              # just hybrid
    uv run scripts/confidence_eval.py --model allenai/olmocr-2-7b

Assumes LM Studio / Ollama is serving the target model at --api-base.
Prints a per-document summary and flags the worst unmatched GT blocks.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from pdf_ocr import (  # noqa: E402
    HybridAligner,
    OCRPipeline,
    OCRProcessor,
    PDFHandler,
    PromptedGroundedOCR,
)
from pdf_ocr.evaluation import (  # noqa: E402
    compute_report,
    load_doc_checks,
    load_ground_truth,
    run_doc_checks,
)

FIXTURES = ROOT / "tests" / "fixtures"
EXAMPLES = ROOT / "examples"
CHECKS_DIR = ROOT / "evals" / "checks"
BASELINES_DIR = ROOT / "evals" / "baselines"
HISTORY_CSV = ROOT / "evals" / "history.csv"

# A metric may drop this much below its committed baseline before the
# run is declared a regression. Baselines move only via
# --update-baselines, reviewed like code.
REGRESSION_TOLERANCE = 0.02
_LOWER_IS_BETTER = {"avg_cer"}

JOBS = [
    ("digital.pdf", "ground_truth_digital.json"),
    ("hybrid.pdf", "ground_truth_hybrid.json"),
    ("handwritten.pdf", "ground_truth_handwritten.json"),
    # Dense and notes fixtures were bootstrapped from the hybrid pipeline's
    # own output_*.pdf via scripts/fixture_from_output.py — too dense to
    # hand-build, and the grounded VLM hits "context size exceeded" on
    # them so build_fixture.py couldn't be used either. Useful for
    # *regression* testing: if a future change degrades the hybrid output
    # against this baseline, recall drops will flag it. Less useful for
    # absolute hybrid-vs-grounded comparison since the bar is set by the
    # hybrid path itself.
    ("dense.pdf", "ground_truth_dense.json"),
    ("notes.pdf", "ground_truth_notes.json"),
]


# ---------------------------------------------------------------------------


async def run_grounded(pdf: Path, api_base: str, model: str, max_dim: int,
                       concurrency: int = 4):
    backend = PromptedGroundedOCR(
        api_base=api_base,
        model=model,
        max_image_dim=max_dim,
        max_tokens=16384,
        concurrency=concurrency,
    )
    # Use ocr_document directly — no need to write an output PDF for scoring.
    response = await backend.ocr_document(str(pdf))
    return [(b.bbox, b.text) for b in response.blocks]


async def run_hybrid(pdf: Path, api_base: str, model: str, max_dim: int,
                     concurrency: int = 4):
    # Run pipeline to a throwaway file; we only need pages_structured data.
    # Smuggle the alignment result out via a custom output_writer.
    captured: dict = {}

    def capture_writer(_in, _out, pages_data, _dpi):
        captured["pages_data"] = pages_data

    pipe = OCRPipeline(
        aligner=HybridAligner(),
        ocr_processor=OCRProcessor(api_base=api_base, model=model),
        pdf_handler=PDFHandler(),
        output_writer=capture_writer,
    )
    await pipe.run(
        str(pdf), str(pdf.with_suffix(".scoring.pdf")),
        max_image_dim=max_dim, concurrency=concurrency, refine=True,
    )
    blocks = []
    for _page_num, page_blocks in captured.get("pages_data", {}).items():
        for bbox, text in page_blocks:
            if text.strip():
                blocks.append((bbox, text))
    return blocks


# ---------------------------------------------------------------------------


def render_report(console: Console, reports: list):
    table = Table(title="Confidence evaluation", show_lines=True)
    table.add_column("document")
    table.add_column("path")
    table.add_column("GT", justify="right")
    table.add_column("Pipe", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("R@0.5", justify="right")
    table.add_column("Hmean", justify="right")
    table.add_column("IoU", justify="right")
    table.add_column("Sim", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("BoW F1", justify="right")
    table.add_column("ChrR/P", justify="right")
    table.add_column("Checks", justify="right")

    for path, report, checks in reports:
        table.add_row(
            report.document,
            path,
            str(report.gt_count),
            str(report.pipeline_count),
            f"{report.block_recall:.2f}",
            f"{report.recall_at(0.5):.2f}",
            f"{report.hmean:.2f}",
            f"{report.avg_iou:.2f}",
            f"{report.avg_text_similarity:.2f}",
            f"{report.avg_cer:.2f}",
            f"{report.bow_f1:.2f}",
            f"{report.char_recall:.2f}/{report.char_precision:.2f}",
            "-" if checks is None else f"{checks[0]}/{checks[1]}",
        )
    console.print(table)

    # Unmatched-blocks detail. Misses are marked by a missing pipeline
    # pairing — their `iou` field is a best-available diagnostic that can
    # legitimately exceed the threshold under optimal matching.
    for path, report, checks in reports:
        unmatched = [m for m in report.matches if m.pipeline_text is None]
        if unmatched:
            console.print(
                f"\n[yellow]Unmatched GT blocks in {report.document} ({path}):[/]"
            )
            for m in unmatched[:6]:
                snippet = m.gt_text[:80].replace("\n", " ")
                console.print(f"  - {snippet!r}  (best_iou={m.iou:.2f})")
            if len(unmatched) > 6:
                console.print(f"  ... and {len(unmatched) - 6} more.")
        if checks is not None and checks[2]:
            console.print(
                f"[yellow]Failed checks in {report.document} ({path}):[/]"
            )
            for why in checks[2]:
                console.print(f"  - {why}")


# --- baseline ratchet + history --------------------------------------------


def _metrics_row(report, checks) -> dict[str, float]:
    row = report.to_metrics_dict()
    if checks is not None and checks[1] > 0:
        row["checks_pass_rate"] = round(checks[0] / checks[1], 4)
    return row


def _baseline_path(document: str, path_label: str) -> Path:
    stem = Path(document).stem
    return BASELINES_DIR / f"{stem}__{path_label}.json"


def compare_to_baseline(document: str, path_label: str, metrics: dict) -> list[str]:
    """Return regression descriptions vs the committed baseline, if any."""
    bp = _baseline_path(document, path_label)
    if not bp.exists():
        return []
    baseline = json.loads(bp.read_text(encoding="utf-8"))
    regressions = []
    for key, base_val in baseline.items():
        if key not in metrics:
            continue
        cur = metrics[key]
        if key in _LOWER_IS_BETTER:
            if cur > base_val + REGRESSION_TOLERANCE:
                regressions.append(
                    f"{document}/{path_label}: {key} {cur:.3f} worse than "
                    f"baseline {base_val:.3f} (+{cur - base_val:.3f})"
                )
        elif cur < base_val - REGRESSION_TOLERANCE:
            regressions.append(
                f"{document}/{path_label}: {key} {cur:.3f} below "
                f"baseline {base_val:.3f} (-{base_val - cur:.3f})"
            )
    return regressions


def write_baseline(document: str, path_label: str, metrics: dict) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    bp = _baseline_path(document, path_label)
    bp.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return bp


def append_history(rows: list[dict]) -> None:
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "document", "path"] + sorted(
        {k for r in rows for k in r if k not in ("timestamp", "document", "path")}
    )
    new_file = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["both", "grounded", "hybrid"], default="both")
    parser.add_argument("--api-base", default="http://localhost:1234/v1")
    parser.add_argument("--grounded-model", default="qwen/qwen3-vl-8b")
    parser.add_argument("--hybrid-model", default="allenai/olmocr-2-7b")
    parser.add_argument("--max-image-dim", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--matching", choices=["hungarian", "greedy"], default="hungarian",
        help="Box-assignment strategy. hungarian (default): globally "
             "optimal one-to-one matching. greedy: legacy order-dependent "
             "matching, for comparison against historical reports.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Parallel LLM requests per document (crop calls dominate "
             "per-box pages; local servers queue excess requests).",
    )
    parser.add_argument(
        "--update-baselines", action="store_true",
        help="Write this run's per-axis metrics to evals/baselines/ as the "
             "new regression floor. Commit the result — baselines move "
             "only through reviewed updates.",
    )
    args = parser.parse_args()

    console = Console()
    reports = []

    def doc_checks(pdf_name: str, out) -> tuple[int, int, list[str]] | None:
        rules_path = CHECKS_DIR / f"{Path(pdf_name).stem}.jsonl"
        if not rules_path.exists():
            return None
        text = "\n".join(t for _b, t in out if t.strip())
        return run_doc_checks(text, load_doc_checks(rules_path))

    for pdf_name, fixture_name in JOBS:
        pdf = EXAMPLES / pdf_name
        fixture = FIXTURES / fixture_name
        gt, (fw, fh) = load_ground_truth(fixture)
        console.print(f"\n[bold]>> {pdf_name}[/]  (GT: {len(gt)} blocks, {fw}x{fh})")

        if args.path in ("both", "grounded"):
            console.print("   [cyan]grounded[/]...")
            try:
                out = await run_grounded(pdf, args.api_base, args.grounded_model,
                                         args.max_image_dim, args.concurrency)
                report = compute_report(pdf_name, gt, out, iou_threshold=args.iou_threshold,
                                        matching=args.matching)
                reports.append(("grounded", report, doc_checks(pdf_name, out)))
                console.print(f"   {report.summary_line()}")
            except Exception as e:
                console.print(f"   [red]grounded failed: {type(e).__name__}: {e}[/]")

        if args.path in ("both", "hybrid"):
            console.print("   [cyan]hybrid[/]...")
            try:
                out = await run_hybrid(pdf, args.api_base, args.hybrid_model,
                                       args.max_image_dim, args.concurrency)
                report = compute_report(pdf_name, gt, out, iou_threshold=args.iou_threshold,
                                        matching=args.matching)
                reports.append(("hybrid", report, doc_checks(pdf_name, out)))
                console.print(f"   {report.summary_line()}")
            except Exception as e:
                console.print(f"   [red]hybrid failed: {type(e).__name__}: {e}[/]")

    console.print()
    render_report(console, reports)

    # Baseline ratchet + run history.
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    history_rows = []
    all_regressions: list[str] = []
    for path_label, report, checks in reports:
        metrics = _metrics_row(report, checks)
        history_rows.append(
            {"timestamp": timestamp, "document": report.document,
             "path": path_label, **metrics}
        )
        if args.update_baselines:
            bp = write_baseline(report.document, path_label, metrics)
            console.print(f"[green]baseline written:[/] {bp.relative_to(ROOT)}")
        else:
            all_regressions.extend(
                compare_to_baseline(report.document, path_label, metrics)
            )
    if history_rows:
        append_history(history_rows)

    if all_regressions:
        console.print("\n[red bold]REGRESSIONS vs committed baselines:[/]")
        for r in all_regressions:
            console.print(f"  [red]- {r}[/]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
