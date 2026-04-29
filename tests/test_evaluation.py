"""Unit tests for the confidence evaluation module (no LLM required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_ocr.evaluation import (
    GTBlock,
    _detect_bbox_axis_order,
    _swap_axes,
    compute_report,
    iou,
    load_ground_truth,
    text_similarity,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestIoU:
    def test_identical_boxes(self):
        assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0

    def test_disjoint_boxes(self):
        assert iou([0, 0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]) == 0.0

    def test_half_overlap(self):
        # Two unit squares overlapping by exactly half horizontally.
        score = iou([0, 0, 1, 1], [0.5, 0, 1.5, 1])
        assert 0.33 < score < 0.34  # 0.5 / 1.5

    def test_containment(self):
        # Inner 0.5x0.5 inside outer 1x1: intersection 0.25, union 1.0 → 0.25.
        score = iou([0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75])
        assert 0.24 < score < 0.26


class TestAxisDetection:
    def test_detects_standard_xyxy(self):
        # Landscape-shaped boxes (wide > tall) — standard OCR output.
        boxes = [
            [100, 50, 900, 100],
            [100, 120, 700, 170],
            [100, 200, 800, 240],
        ]
        assert _detect_bbox_axis_order(boxes) == "xyxy"

    def test_detects_swapped_yxyx(self):
        # Portrait-shaped if read as xyxy → must be yxyx in reality.
        boxes = [
            [50, 100, 100, 900],
            [120, 100, 170, 700],
            [200, 100, 240, 800],
        ]
        assert _detect_bbox_axis_order(boxes) == "yxyx"

    def test_mixed_majority_wins(self):
        # Two portrait, one landscape → declare portrait (yxyx).
        boxes = [
            [50, 100, 100, 900],  # portrait
            [120, 100, 170, 900],  # portrait
            [100, 50, 900, 100],   # landscape
        ]
        assert _detect_bbox_axis_order(boxes) == "yxyx"

    def test_empty_input_defaults_to_xyxy(self):
        assert _detect_bbox_axis_order([]) == "xyxy"


class TestSwapAxes:
    def test_swaps_correctly(self):
        assert _swap_axes([10, 20, 30, 40]) == [20, 10, 40, 30]


class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_markdown_tolerant(self):
        # Bold markdown shouldn't count against us.
        assert text_similarity("**Algorithm**", "Algorithm") > 0.9

    def test_case_insensitive(self):
        assert text_similarity("HELLO", "hello") == 1.0

    def test_partial_overlap(self):
        score = text_similarity("the quick brown fox", "quick brown")
        assert 0.6 < score < 1.0

    def test_empty_comparison(self):
        assert text_similarity("", "hello") == 0.0


class TestLoadGroundTruth:
    def test_handwritten_fixture_loads(self):
        blocks, (fw, fh) = load_ground_truth(FIXTURES / "ground_truth_handwritten.json")
        assert (fw, fh) == (1654, 2170)
        # 15 layout entries, 1 `list_marker` ("-") filtered → 14 blocks.
        assert len(blocks) == 14
        title = blocks[0]
        assert "Algorithm" in title.text
        assert title.bbox[1] < 0.2, "title should be near top"

    def test_hybrid_fixture_loads(self):
        blocks, (fw, fh) = load_ground_truth(FIXTURES / "ground_truth_hybrid.json")
        assert (fw, fh) == (1654, 1962)
        # 45 entries, 7 are `empty_line` → 38 content blocks.
        assert len(blocks) == 38
        portrait = sum(
            1 for b in blocks
            if (b.bbox[3] - b.bbox[1]) > 1.5 * (b.bbox[2] - b.bbox[0])
        )
        assert portrait == 0

    def test_digital_fixture_loads(self):
        blocks, (fw, fh) = load_ground_truth(FIXTURES / "ground_truth_digital.json")
        assert (fw, fh) == (1700, 2200)
        # 17 entries, 1 `signature_line` filtered → 16 blocks.
        assert len(blocks) == 16
        portrait = sum(
            1 for b in blocks
            if (b.bbox[3] - b.bbox[1]) > 1.5 * (b.bbox[2] - b.bbox[0])
        )
        assert portrait == 0

    def test_coordinates_are_normalized(self):
        blocks, _ = load_ground_truth(FIXTURES / "ground_truth_handwritten.json")
        for b in blocks:
            assert 0.0 <= b.bbox[0] < b.bbox[2] <= 1.0
            assert 0.0 <= b.bbox[1] < b.bbox[3] <= 1.0

    def test_skips_non_content_labels(self):
        """Image, empty_line, signature_line, list_marker must be filtered."""
        # hybrid has empty_line blocks, handwritten has list_marker,
        # digital has signature_line — none should appear post-load.
        for fixture in [
            "ground_truth_handwritten.json",
            "ground_truth_hybrid.json",
            "ground_truth_digital.json",
        ]:
            blocks, _ = load_ground_truth(FIXTURES / fixture)
            labels = {b.label for b in blocks}
            assert not labels & {"image", "empty_line", "signature_line", "list_marker"}


class TestComputeReport:
    def test_perfect_match(self):
        gt = [
            GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="alpha"),
            GTBlock(bbox=[0.5, 0.5, 1.0, 1.0], text="beta"),
        ]
        pipeline = [
            ([0.0, 0.0, 0.5, 0.5], "alpha"),
            ([0.5, 0.5, 1.0, 1.0], "beta"),
        ]
        report = compute_report("x", gt, pipeline)
        assert report.gt_count == 2
        assert len(report.matched) == 2
        assert report.block_recall == 1.0
        assert report.avg_text_similarity == 1.0
        assert report.avg_iou == 1.0

    def test_missed_block(self):
        gt = [
            GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="alpha"),
            GTBlock(bbox=[0.5, 0.5, 1.0, 1.0], text="beta"),
        ]
        pipeline = [([0.0, 0.0, 0.5, 0.5], "alpha")]  # beta is missing
        report = compute_report("x", gt, pipeline)
        assert report.gt_count == 2
        assert len(report.matched) == 1
        assert report.block_recall == 0.5

    def test_no_double_matching(self):
        # Two GT blocks, only one pipeline block — it must pair with
        # exactly one of them, not both.
        gt = [
            GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="alpha"),
            GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="beta"),  # identical bbox
        ]
        pipeline = [([0.0, 0.0, 0.5, 0.5], "alpha")]
        report = compute_report("x", gt, pipeline)
        assert len(report.matched) == 1

    def test_low_iou_unmatched(self):
        gt = [GTBlock(bbox=[0.0, 0.0, 0.2, 0.2], text="alpha")]
        # Pipeline box at a far corner — near-zero IoU with GT.
        pipeline = [([0.8, 0.8, 1.0, 1.0], "alpha")]
        report = compute_report("x", gt, pipeline, iou_threshold=0.3)
        assert len(report.matched) == 0
        assert report.block_recall == 0.0

    def test_text_similarity_with_minor_differences(self):
        gt = [GTBlock(bbox=[0.0, 0.0, 1.0, 0.1], text="Halting: the algo needs to finish in finite time.")]
        pipeline = [([0.0, 0.0, 1.0, 0.1], "Halting the algo needs to finish in finite time")]
        report = compute_report("x", gt, pipeline)
        assert len(report.matched) == 1
        assert report.avg_text_similarity > 0.9

    def test_summary_line_contains_metrics(self):
        gt = [GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="x")]
        pipeline = [([0.0, 0.0, 0.5, 0.5], "x")]
        report = compute_report("sample.pdf", gt, pipeline)
        line = report.summary_line()
        assert "sample.pdf" in line
        assert "recall=1.00" in line


def test_report_handles_empty_pipeline():
    gt = [GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="alpha")]
    report = compute_report("x", gt, [])
    assert report.pipeline_count == 0
    assert report.block_recall == 0.0
    assert report.avg_text_similarity == 0.0


def test_report_handles_empty_ground_truth():
    report = compute_report("x", [], [([0, 0, 0.5, 0.5], "alpha")])
    assert report.gt_count == 0
    assert report.block_recall == 0.0  # 0/max(1, 0) = 0


@pytest.mark.parametrize("fixture", [
    "ground_truth_handwritten.json",
    "ground_truth_hybrid.json",
    "ground_truth_digital.json",
])
def test_all_fixtures_loadable(fixture):
    """Smoke test: every shipped fixture loads cleanly."""
    blocks, (fw, fh) = load_ground_truth(FIXTURES / fixture)
    assert len(blocks) > 0
    assert fw > 0 and fh > 0
    for b in blocks:
        assert b.text, f"empty text in {fixture}"
        assert all(0.0 <= c <= 1.0 for c in b.bbox), f"unnormalized bbox in {fixture}"
