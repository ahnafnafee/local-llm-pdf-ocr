"""Run the existing live evaluator with results kept outside tracked baselines."""

import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]
os.environ["TQDM_DISABLE"] = "1"

spec = importlib.util.spec_from_file_location("confidence_eval", ROOT / "scripts/confidence_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
path_label = sys.argv[sys.argv.index("--path") + 1]
run_label = os.environ.get("CONFIDENCE_RUN_LABEL", path_label)
if os.environ.get("CONFIDENCE_DOCUMENTS"):
    selected = set(os.environ["CONFIDENCE_DOCUMENTS"].split(","))
    evaluation.JOBS = [job for job in evaluation.JOBS if Path(job[0]).stem in selected]
evaluation.HISTORY_CSV = OUTPUT / f"{run_label}-history.csv"
original_render = evaluation.render_report
original_grounded = evaluation.PromptedGroundedOCR.ocr_document
original_compute = evaluation.compute_report
empty_grounded_pages = []


def retain_inference_output(document, gt, output, **kwargs):
    (OUTPUT / f"{run_label}-{Path(document).stem}-page-indices.json").write_text(
        json.dumps(kwargs.get("pipeline_page_indices")) + "\n", encoding="utf-8",
    )
    (OUTPUT / f"{run_label}-{Path(document).stem}-output.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return original_compute(document, gt, output, **kwargs)


evaluation.compute_report = retain_inference_output


async def retain_grounded_response(backend, pdf_path, *args, **kwargs):
    response = await original_grounded(backend, pdf_path, *args, **kwargs)
    selected = response.page_indices or list(range(len(response.page_sizes)))
    recognized = {block.page_index for block in response.blocks if block.text.strip()}
    missing = [index + 1 for index in selected if index not in recognized]
    if missing:
        empty_grounded_pages.append({"document": Path(pdf_path).name, "page_numbers": missing})
    (OUTPUT / f"{run_label}-{Path(pdf_path).stem}-blocks.json").write_text(
        json.dumps(asdict(response), indent=2) + "\n", encoding="utf-8",
    )
    return response


evaluation.PromptedGroundedOCR.ocr_document = retain_grounded_response


def retain_reports(console, reports):
    original_render(console, reports)
    result = []
    for mode, report, checks in reports:
        metrics = evaluation._metrics_row(report, checks)
        result.append({
            "document": report.document,
            "mode": mode,
            "gt_count": report.gt_count,
            "pipeline_count": report.pipeline_count,
            "metrics": metrics,
            "checks": checks,
            "regressions": evaluation.compare_to_baseline(report.document, mode, metrics, context=evaluation.evaluation_context(
                ROOT / 'examples' / report.document,
                ROOT / 'tests/fixtures' / f'ground_truth_{Path(report.document).stem}.json',
                ROOT / 'evals/checks' / f'{Path(report.document).stem}.jsonl',
                model=sys.argv[sys.argv.index(f'--{mode}-model')+1],
                max_dim=int(sys.argv[sys.argv.index('--max-image-dim')+1]) if '--max-image-dim' in sys.argv else 1024,
            )),
        })
    (OUTPUT / f"{run_label}-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


evaluation.render_report = retain_reports
# Record the real SDK response, including completion status and timing.
import time
from openai.resources.chat.completions import AsyncCompletions
original_create = AsyncCompletions.create
request_number = 0
async def record_create(resource, **kwargs):
    global request_number
    request_number += 1
    number = request_number
    started = time.monotonic()
    result = await original_create(resource, **kwargs)
    elapsed = time.monotonic() - started
    payload = result.model_dump()
    (OUTPUT / f"{run_label}-completion-{number:04d}.json").write_text(
        json.dumps({"elapsed_s": elapsed, "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"), "response": payload},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"request {number}: {elapsed:.2f}s, finish={result.choices[0].finish_reason}, "
          f"tokens={result.usage.completion_tokens if result.usage else None}", flush=True)
    return result
AsyncCompletions.create = record_create
status = asyncio.run(evaluation.main())
results = json.loads((OUTPUT / f"{run_label}-results.json").read_text(encoding="utf-8"))
if len(results) != len(evaluation.JOBS) or any(result["pipeline_count"] == 0 for result in results):
    print("Evaluation failed: a document failed or produced no recognized blocks.")
    status = status or 2
if empty_grounded_pages:
    print("Grounded pages without recognized blocks:", empty_grounded_pages)
    (OUTPUT / "grounded-empty-pages.json").write_text(json.dumps(empty_grounded_pages, indent=2) + "\n", encoding="utf-8")
    status = status or 2
raise SystemExit(status)
