"""Unit tests for the confidence evaluation module (no LLM required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_ocr.evaluation import (
    GTBlock,
    _detect_bbox_axis_order,
    _swap_axes,
    bag_of_words_f1,
    char_coverage,
    compute_report,
    iou,
    load_doc_checks,
    load_ground_truth,
    run_doc_checks,
    text_cer,
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


class TestMatchingStrategies:
    # Two GT blocks competing for one pipeline box: optimal assignment
    # gives the box to the better-fitting GT block; greedy gives it to
    # whichever GT block comes first.
    _GT_CONTESTED = [
        GTBlock(bbox=[0.0, 0.0, 0.5, 0.2], text="alpha"),    # IoU 0.9 w/ box
        GTBlock(bbox=[0.05, 0.0, 0.5, 0.2], text="beta"),    # IoU 1.0 w/ box
    ]
    _PIPE_ONE_BOX = [([0.05, 0.0, 0.5, 0.2], "beta")]

    def test_hungarian_assigns_contested_box_optimally(self):
        report = compute_report("x", self._GT_CONTESTED, self._PIPE_ONE_BOX)
        assert len(report.matched) == 1
        assert report.matched[0].gt_text == "beta"
        assert report.matched[0].iou == 1.0

    def test_greedy_still_available_and_order_dependent(self):
        report = compute_report(
            "x", self._GT_CONTESTED, self._PIPE_ONE_BOX, matching="greedy",
        )
        assert len(report.matched) == 1
        assert report.matched[0].gt_text == "alpha"  # first GT wins

    def test_unmatched_diagnostic_iou_not_counted_as_match(self):
        # The losing GT block's miss entry carries its best IoU (0.9) as
        # a diagnostic — above the threshold, but it must not inflate
        # recall because the box went to the other GT block.
        report = compute_report("x", self._GT_CONTESTED, self._PIPE_ONE_BOX)
        assert report.block_recall == 0.5
        misses = [m for m in report.matches if m.pipeline_text is None]
        assert len(misses) == 1
        assert misses[0].iou > report.iou_threshold

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown matching strategy"):
            compute_report("x", [], [], matching="psychic")

    def test_precision_and_hmean(self):
        gt = [GTBlock(bbox=[0.0, 0.0, 0.5, 0.5], text="alpha")]
        pipeline = [
            ([0.0, 0.0, 0.5, 0.5], "alpha"),
            ([0.6, 0.6, 0.9, 0.9], "spurious"),
        ]
        report = compute_report("x", gt, pipeline)
        assert report.block_recall == 1.0
        assert report.precision == 0.5
        assert report.hmean == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    def test_recall_at_stricter_threshold(self):
        # Pair matches at IoU ~0.45 (inter 0.05 / union 0.11): counted
        # at the 0.3 bar, not at 0.5.
        gt = [GTBlock(bbox=[0.0, 0.0, 0.4, 0.2], text="alpha")]
        pipeline = [([0.15, 0.0, 0.55, 0.2], "alpha")]
        report = compute_report("x", gt, pipeline, iou_threshold=0.3)
        assert report.block_recall == 1.0
        assert report.recall_at(0.5) == 0.0


class TestTextCer:
    def test_perfect_is_zero(self):
        assert text_cer("Hello world", "hello   world") == 0.0  # normalized

    def test_empty_reference_and_hypothesis(self):
        assert text_cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert text_cer("", "ghost text") == 1.0

    def test_missing_hypothesis_is_total_error(self):
        assert text_cer("real text", "") == 1.0

    def test_capped_at_one(self):
        # A runaway hypothesis can exceed CER 1.0 uncapped; the cap keeps
        # one bad block from dominating a document average.
        assert text_cer("ab", "x" * 50) == 1.0

    def test_single_substitution(self):
        # "1530" -> "1500": one substituted char of four.
        assert text_cer("1530", "1500") == pytest.approx(0.25)


class TestBagOfWordsF1:
    def test_identical_sets(self):
        assert bag_of_words_f1(["alpha beta"], ["alpha beta"]) == 1.0

    def test_order_independent(self):
        # The whole point: reading order must not affect the score.
        gt = ["left column first", "right column second"]
        pipe = ["right column second", "left column first"]
        assert bag_of_words_f1(gt, pipe) == 1.0

    def test_binding_independent(self):
        # Same tokens distributed across different block boundaries.
        assert bag_of_words_f1(["a b c", "d"], ["a", "b c d"]) == 1.0

    def test_disjoint_is_zero(self):
        assert bag_of_words_f1(["alpha"], ["omega"]) == 0.0

    def test_both_empty_is_one(self):
        assert bag_of_words_f1([], []) == 1.0

    def test_one_side_empty_is_zero(self):
        assert bag_of_words_f1(["text"], []) == 0.0


class TestCharCoverage:
    def test_perfect_alignment(self):
        gt = [GTBlock(bbox=[0.0, 0.0, 0.5, 0.1], text="hello")]
        pipe = [([0.0, 0.0, 0.5, 0.1], "hello")]
        assert char_coverage(gt, pipe) == (1.0, 1.0)

    def test_split_tolerant(self):
        # One GT block detected as two half-boxes: an IoU pairing fails
        # both halves, but every GT character center is still covered.
        gt = [GTBlock(bbox=[0.0, 0.0, 0.8, 0.1], text="0123456789")]
        pipe = [
            ([0.0, 0.0, 0.4, 0.1], "01234"),
            ([0.4, 0.0, 0.8, 0.1], "56789"),
        ]
        recall, precision = char_coverage(gt, pipe)
        assert recall == 1.0
        assert precision == 1.0

    def test_merge_tolerant(self):
        # Two GT label blocks merged into one wide detected box.
        gt = [
            GTBlock(bbox=[0.0, 0.0, 0.3, 0.1], text="Name:"),
            GTBlock(bbox=[0.5, 0.0, 0.9, 0.1], text="Sally"),
        ]
        pipe = [([0.0, 0.0, 0.9, 0.1], "Name: Sally")]
        recall, _precision = char_coverage(gt, pipe)
        assert recall == 1.0

    def test_uncovered_region_counts_against_recall(self):
        gt = [GTBlock(bbox=[0.0, 0.0, 1.0, 0.1], text="0123456789")]
        pipe = [([0.0, 0.0, 0.5, 0.1], "01234")]  # right half undetected
        recall, _ = char_coverage(gt, pipe)
        assert recall == pytest.approx(0.5)

    def test_empty_inputs(self):
        assert char_coverage([], []) == (0.0, 0.0)


class TestDocChecks:
    def test_present_exact(self):
        passed, total, fails = run_doc_checks(
            "HEALTH INTAKE FORM\nSally Walker",
            [{"type": "present", "text": "Sally Walker"}],
        )
        assert (passed, total, fails) == (1, 1, [])

    def test_present_case_insensitive_default(self):
        passed, _, _ = run_doc_checks(
            "health intake form", [{"type": "present", "text": "HEALTH INTAKE"}],
        )
        assert passed == 1

    def test_present_whitespace_collapsed(self):
        passed, _, _ = run_doc_checks(
            "Student Name\n(Last, First):",
            [{"type": "present", "text": "Student Name (Last, First):"}],
        )
        assert passed == 1

    def test_present_fuzzy_within_budget(self):
        passed, _, _ = run_doc_checks(
            "Vyrance (25mg) daily",
            [{"type": "present", "text": "Vyvanse (25mg)", "max_diffs": 2}],
        )
        assert passed == 1

    def test_present_fuzzy_over_budget_fails(self):
        passed, total, fails = run_doc_checks(
            "Vigrahe (25mg) daily",
            [{"type": "present", "text": "Vyvanse", "max_diffs": 1}],
        )
        assert passed == 0
        assert "missing" in fails[0]

    def test_absent_passes_when_clean(self):
        passed, _, _ = run_doc_checks(
            "ordinary content", [{"type": "absent", "text": "lorem ipsum"}],
        )
        assert passed == 1

    def test_absent_fails_on_hallucination(self):
        passed, _, fails = run_doc_checks(
            "The Quick Brown Fox jumps",
            [{"type": "absent", "text": "the quick brown fox"}],
        )
        assert passed == 0
        assert "forbidden" in fails[0]

    def test_order_pass_and_fail(self):
        rules = [{"type": "order", "first": "Signatures", "then": "Notes"}]
        assert run_doc_checks("Signatures ... Notes", rules)[0] == 1
        assert run_doc_checks("Notes ... Signatures", rules)[0] == 0

    def test_order_missing_operand_fails_with_reason(self):
        passed, _, fails = run_doc_checks(
            "only Notes here", [{"type": "order", "first": "Signatures", "then": "Notes"}],
        )
        assert passed == 0
        assert "missing" in fails[0]

    def test_unknown_rule_type_fails(self):
        passed, total, fails = run_doc_checks("x", [{"type": "psychic", "text": "x"}])
        assert (passed, total) == (0, 1)

    def test_jsonl_loader_skips_comments_and_blanks(self, tmp_path: Path):
        p = tmp_path / "rules.jsonl"
        p.write_text(
            '// comment line\n'
            '\n'
            '{"type": "present", "text": "alpha"}\n'
            '{"type": "absent", "text": "beta"}\n',
            encoding="utf-8",
        )
        rules = load_doc_checks(p)
        assert [r["type"] for r in rules] == ["present", "absent"]

    def test_shipped_check_files_load(self):
        checks_dir = Path(__file__).parent.parent / "evals" / "checks"
        files = sorted(checks_dir.glob("*.jsonl"))
        assert len(files) >= 5
        for f in files:
            rules = load_doc_checks(f)
            assert rules, f"{f.name} has no rules"
            for r in rules:
                assert r["type"] in ("present", "absent", "order")


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
