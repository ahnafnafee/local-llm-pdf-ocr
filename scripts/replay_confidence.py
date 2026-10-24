#!/usr/bin/env python3
"""Rescore a frozen inference capture against the current audited references.

This performs no inference. --write-control records a pre-change comparison
point, including its known failures; live evaluation still requires every
content check to pass. It never replaces the original capture's predictions.
"""

import argparse
import hashlib
import json
from pathlib import Path

from confidence_eval import (
    ROOT, CHECKS_DIR, EXAMPLES, FIXTURES, _metrics_row, evaluation_context,
    compute_report, load_ground_truth, load_doc_checks, run_doc_checks, write_baseline,
)


def replay(capture_path: Path, *, write_control=False):
    raw = capture_path.read_text(encoding="utf-8")
    capture = json.loads(raw)
    document = capture["document"]
    # Captures describe repository examples, never arbitrary filesystem paths.
    if Path(document).name != document:
        raise ValueError("Capture document must be a filename")
    pdf = EXAMPLES / document
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != capture["document_sha256"]:
        raise ValueError("Capture and source document hashes differ")
    fixture = FIXTURES / f"ground_truth_{Path(document).stem}.json"
    checks = CHECKS_DIR / f"{Path(document).stem}.jsonl"
    gt, _ = load_ground_truth(fixture)
    report = compute_report(document, gt, capture["blocks"],
                            pipeline_page_indices=capture["page_indices"])
    check_result = run_doc_checks(
        "\n".join(t for _, t in capture["blocks"]), load_doc_checks(checks),
    )
    metrics = _metrics_row(report, check_result)
    context = evaluation_context(pdf, fixture, checks, model=capture["model"],
                                 max_dim=capture["max_image_dim"])
    if write_control:
        write_baseline(document, capture["mode"], metrics, context=context, provenance={
            "kind": "frozen-control", "source_commit": capture["source_commit"],
            "capture": capture_path.resolve().relative_to(ROOT).as_posix(),
            "capture_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "runtime": capture["runtime"],
        })
    return {"document": document, "metrics": metrics, "checks": check_result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--write-control", action="store_true",
                        help="Explicitly migrate comparison baselines from these frozen outputs")
    args = parser.parse_args()
    results = [replay(path, write_control=args.write_control) for path in args.captures]
    print(json.dumps({"inference_performed": False, "results": results}, indent=2))
    return int(any(r["checks"][0] != r["checks"][1] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
