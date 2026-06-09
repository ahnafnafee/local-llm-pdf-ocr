"""Inspect Surya detection geometry across the example documents.

Prints, per page: box count, median normalized box width, the polygon
angle distribution (min-area-rect tilt of each detected line), and the
detection-only recall against the ground-truth fixtures (Hungarian,
IoU>=0.3). Used to calibrate the form-routing signal in the pipeline's
dense-mode auto heuristic, the writers' angle-flattening threshold, and
the slow-tier geometry regression floors.

Usage:
    uv run scripts/inspect_detection_geometry.py
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pdf_ocr import HybridAligner, PDFHandler  # noqa: E402
from pdf_ocr.evaluation import compute_report, load_ground_truth  # noqa: E402

DOCS = [
    ("digital.pdf", "ground_truth_digital.json"),
    ("hybrid.pdf", "ground_truth_hybrid.json"),
    ("handwritten.pdf", "ground_truth_handwritten.json"),
    ("dense.pdf", "ground_truth_dense.json"),
    ("notes.pdf", "ground_truth_notes.json"),
    ("multipage.pdf", None),
    ("image.png", None),
]


def main() -> int:
    import base64

    aligner = HybridAligner()
    handler = PDFHandler()

    for doc, fixture in DOCS:
        path = ROOT / "examples" / doc
        images = handler.convert_to_images(str(path), 200)
        page_nums = sorted(images.keys())
        image_bytes = [base64.b64decode(images[p]) for p in page_nums]
        batch = aligner.get_detected_boxes_batch(image_bytes)

        for p, boxes in zip(page_nums, batch):
            if not boxes:
                print(f"{doc} p{p + 1}: no boxes")
                continue
            widths = sorted(b[2] - b[0] for b in boxes)
            median_w = widths[len(widths) // 2]
            angles = [abs(getattr(b, "angle", 0.0)) for b in boxes]
            print(
                f"{doc} p{p + 1}: boxes={len(boxes)} "
                f"median_w={median_w:.3f} "
                f"angles: max={max(angles):.2f} "
                f"p90={sorted(angles)[int(0.9 * (len(angles) - 1))]:.2f} "
                f"mean={statistics.mean(angles):.2f}"
            )

        if fixture:
            gt, _dims = load_ground_truth(ROOT / "tests" / "fixtures" / fixture)
            flat = [(list(b), "x") for b in batch[0]]
            report = compute_report(doc, gt, flat, iou_threshold=0.3)
            print(
                f"  detection-only vs GT: recall={report.block_recall:.2f} "
                f"prec={report.precision:.2f} avg_iou={report.avg_iou:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
