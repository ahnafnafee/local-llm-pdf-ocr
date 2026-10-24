"""Run the repository's image evaluator with the currently served model."""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("confidence_image", ROOT / "scripts/confidence_image.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
parser = argparse.ArgumentParser()
parser.add_argument("--path", choices=["hybrid", "grounded"], required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()


async def main():
    started = time.monotonic()
    runner = evaluation.run_hybrid if args.path == "hybrid" else evaluation.run_grounded
    blocks = await runner(ROOT / "examples/image.avif", "http://127.0.0.1:28127/v1", args.model, 2048)
    gt, _ = evaluation.load_ground_truth(ROOT / "tests/fixtures/ground_truth_image.json")
    report = evaluation.compute_report("image.avif", gt, blocks)
    result = {
        "document": "image.avif", "mode": args.path, "model": args.model,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "gt_count": report.gt_count, "pipeline_count": report.pipeline_count,
        "metrics": report.to_metrics_dict(), "output_blocks": blocks,
    }
    (OUTPUT / f"image-{args.path}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(report.summary_line())
    print(json.dumps({k: v for k, v in result.items() if k != "output_blocks"}, indent=2))


asyncio.run(main())
