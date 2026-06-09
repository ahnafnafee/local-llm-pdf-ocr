"""Geometry regression floors — detector-only, no LLM.

Surya detection is deterministic, so the geometry axis of the eval can
be regression-gated offline: detected boxes for the hand-built fixtures
must keep clearing the floors below (measured values minus a small
margin). A drop means a detector upgrade, a preprocessing change, or an
aligner edit moved real geometry quality, independent of any model.

The angle assertions tie the writers' ANGLE_FLATTEN_DEG constant to
reality: straight printed documents must keep producing only sub-
threshold polygon tilt, otherwise the overlay would wobble with
detector noise.
"""

from __future__ import annotations

import base64

import pytest

from pdf_ocr.core.geometry import ANGLE_FLATTEN_DEG
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.evaluation import compute_report, load_ground_truth

pytestmark = pytest.mark.slow

# document -> (block-recall floor, matched-IoU floor) at IoU >= 0.3.
FLOORS = {
    "digital.pdf": ("ground_truth_digital.json", 0.38, 0.60),
    "hybrid.pdf": ("ground_truth_hybrid.json", 0.20, 0.48),
    "handwritten.pdf": ("ground_truth_handwritten.json", 0.90, 0.50),
}

# Straight printed documents; handwriting legitimately tilts.
STRAIGHT_DOCS = ("digital.pdf", "hybrid.pdf")


@pytest.fixture(scope="module")
def detected_boxes(surya_aligner, example_pdfs):
    handler = PDFHandler()
    out = {}
    for name in FLOORS:
        images = handler.convert_to_images(str(example_pdfs[name]), 200)
        page_bytes = [base64.b64decode(images[p]) for p in sorted(images)]
        out[name] = surya_aligner.get_detected_boxes_batch(page_bytes)[0]
    return out


@pytest.mark.parametrize("doc", sorted(FLOORS))
def test_detection_geometry_floor(doc, detected_boxes):
    fixture, recall_floor, iou_floor = FLOORS[doc]
    gt, _dims = load_ground_truth(f"tests/fixtures/{fixture}")
    report = compute_report(
        doc, gt, [(list(b), "x") for b in detected_boxes[doc]],
        iou_threshold=0.3,
    )
    assert report.block_recall >= recall_floor, report.summary_line()
    assert report.avg_iou >= iou_floor, report.summary_line()


@pytest.mark.parametrize("doc", STRAIGHT_DOCS)
def test_straight_documents_stay_under_flatten_threshold(doc, detected_boxes):
    angles = [abs(getattr(b, "angle", 0.0)) for b in detected_boxes[doc]]
    assert angles, "no boxes detected"
    assert max(angles) < ANGLE_FLATTEN_DEG, (
        f"{doc}: max polygon tilt {max(angles):.2f}deg reaches the "
        f"flatten threshold — straight text would render rotated"
    )


def test_detected_boxes_carry_oriented_geometry(detected_boxes):
    # The aligner must not silently fall back to plain lists.
    boxes = detected_boxes["digital.pdf"]
    assert all(getattr(b, "quad", None) is not None for b in boxes)
    assert all(getattr(b, "confidence", None) is not None for b in boxes)
