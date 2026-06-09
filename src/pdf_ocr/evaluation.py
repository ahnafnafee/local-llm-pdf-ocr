"""
Confidence evaluation: compare pipeline output against ground-truth fixtures.

Ground-truth fixtures are in Z.AI hosted GLM-OCR response format:
    {"data": {"layout": [{"block_content": "...", "bbox": [...],
                          "block_label": "text", "page_index": 0}, ...],
              "data_info": {"pages": [{"width": W, "height": H}, ...]}}}

Bbox convention auto-detection: the captured fixtures sometimes use
`[x0, y0, x1, y1]` (handwritten.pdf) and sometimes `[y0, x0, y1, x1]`
(hybrid.pdf / digital.pdf). `_detect_bbox_axis_order` infers which by
aspect ratio across the fixture's boxes and normalizes to `[x0, y0, x1, y1]`
before comparison.

Metrics per document, decomposed by axis so an improvement to one
pipeline component shows up in exactly one place:
    - geometry: block_recall / precision / hmean at the IoU threshold,
      recall_at(0.5) as the tightness bar, avg matched IoU
    - text (assignment-dependent): avg difflib similarity and avg CER
      over matched pairs
    - text (assignment-free): bag-of-words F1 over the full GT and
      pipeline text sets — right even when boxes bind wrongly
    - structure (split/merge tolerant): pseudo-character coverage
      (char_recall / char_precision), CLEval-inspired
    - doc checks: binary present/absent/order rules per fixture
      (olmOCR-bench style), via :func:`run_doc_checks`

Matching is a globally optimal one-to-one assignment on the IoU matrix
(`scipy.optimize.linear_sum_assignment`), so results do not depend on GT
ordering. The legacy greedy strategy (best available pipeline box per GT
box, in GT order) remains available via ``matching="greedy"`` for
comparison against historical reports.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import jiwer

BBox = list[float]


# --- data classes ----------------------------------------------------------


@dataclass
class GTBlock:
    bbox: BBox       # normalized [x0, y0, x1, y1] in 0..1
    text: str
    page_index: int = 0
    label: str = "text"


# Labels that describe *structural* regions rather than selectable text —
# skipping them keeps the confidence eval focused on content blocks.
NON_CONTENT_LABELS = frozenset({
    "image",
    "empty_line",      # underline placeholder for unfilled fields
    "signature_line",  # "_______________________ _______________________"
    "list_marker",     # bare "-" / bullet glyphs
})


@dataclass
class BlockMatch:
    gt_text: str
    gt_bbox: BBox
    pipeline_text: str | None
    pipeline_bbox: BBox | None
    iou: float
    text_similarity: float  # 0..1, difflib ratio
    cer: float = 1.0        # character error rate vs GT (0 = perfect)


@dataclass
class ConfidenceReport:
    document: str
    iou_threshold: float
    gt_count: int
    pipeline_count: int
    matches: list[BlockMatch] = field(default_factory=list)
    bow_f1: float = 0.0          # assignment-free bag-of-words F1
    char_recall: float = 0.0     # PCC coverage of GT chars by pipeline boxes
    char_precision: float = 0.0  # PCC coverage of pipeline chars by GT boxes

    @property
    def matched(self) -> list[BlockMatch]:
        # `pipeline_text is not None` distinguishes real assignments from
        # miss entries, whose `iou` field holds the best-available IoU as
        # a diagnostic — under optimal matching that diagnostic can exceed
        # the threshold when the box was assigned to a different GT block.
        return [
            m for m in self.matches
            if m.pipeline_text is not None and m.iou >= self.iou_threshold
        ]

    @property
    def block_recall(self) -> float:
        return len(self.matched) / max(1, self.gt_count)

    @property
    def precision(self) -> float:
        return len(self.matched) / max(1, self.pipeline_count)

    @property
    def hmean(self) -> float:
        p, r = self.precision, self.block_recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    def recall_at(self, threshold: float) -> float:
        """Recall against a stricter IoU bar than the matching threshold.

        Derived from the existing assignment (pairs are fixed; only the
        acceptance bar moves), so ``recall_at(0.5)`` is comparable across
        runs without re-matching.
        """
        hits = [
            m for m in self.matches
            if m.pipeline_text is not None and m.iou >= threshold
        ]
        return len(hits) / max(1, self.gt_count)

    @property
    def avg_text_similarity(self) -> float:
        ms = self.matched
        if not ms:
            return 0.0
        return sum(m.text_similarity for m in ms) / len(ms)

    @property
    def avg_iou(self) -> float:
        ms = self.matched
        if not ms:
            return 0.0
        return sum(m.iou for m in ms) / len(ms)

    @property
    def avg_cer(self) -> float:
        """Average character error rate over matched pairs (lower = better)."""
        ms = self.matched
        if not ms:
            return 1.0
        return sum(m.cer for m in ms) / len(ms)

    def to_metrics_dict(self) -> dict[str, float]:
        """One flat row of every per-axis metric, for baselines/history."""
        return {
            "recall": round(self.block_recall, 4),
            "recall_at_05": round(self.recall_at(0.5), 4),
            "precision": round(self.precision, 4),
            "hmean": round(self.hmean, 4),
            "avg_iou": round(self.avg_iou, 4),
            "text_sim": round(self.avg_text_similarity, 4),
            "avg_cer": round(self.avg_cer, 4),
            "bow_f1": round(self.bow_f1, 4),
            "char_recall": round(self.char_recall, 4),
            "char_precision": round(self.char_precision, 4),
        }

    def summary_line(self) -> str:
        return (
            f"{self.document:<20} "
            f"gt={self.gt_count:<3} "
            f"pipeline={self.pipeline_count:<3} "
            f"matched={len(self.matched):<3} "
            f"recall={self.block_recall:.2f} "
            f"prec={self.precision:.2f} "
            f"hmean={self.hmean:.2f} "
            f"iou_avg={self.avg_iou:.2f} "
            f"text_sim_avg={self.avg_text_similarity:.2f}"
        )


# --- bbox helpers ----------------------------------------------------------


def iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _detect_bbox_axis_order(raw_boxes: list[BBox]) -> str:
    """
    Return "xyxy" if boxes look like [x0,y0,x1,y1], else "yxyx".

    Heuristic: if the majority of boxes have (v3-v1) >> (v2-v0) — i.e.
    they're heavily "portrait" when interpreted as xyxy — they're almost
    certainly yxyx (height and width got swapped in the source).
    """
    portrait = 0
    counted = 0
    for b in raw_boxes:
        if len(b) != 4:
            continue
        w_xy = abs(b[2] - b[0])
        h_xy = abs(b[3] - b[1])
        if w_xy <= 0 or h_xy <= 0:
            continue
        counted += 1
        if h_xy > 1.5 * w_xy:
            portrait += 1
    if counted == 0:
        return "xyxy"
    return "yxyx" if portrait > counted / 2 else "xyxy"


def _swap_axes(b: BBox) -> BBox:
    """[y0, x0, y1, x1] -> [x0, y0, x1, y1]."""
    return [b[1], b[0], b[3], b[2]]


# --- fixture loader --------------------------------------------------------


def load_ground_truth(fixture_path: Path | str) -> tuple[list[GTBlock], tuple[int, int]]:
    """
    Load a fixture JSON and return (blocks, (fixture_width, fixture_height)).

    Blocks are normalized to `[x0, y0, x1, y1]` in 0..1 space. Non-text
    blocks (label == "image") are skipped. Axis order is auto-detected.
    """
    with open(fixture_path) as f:
        data = json.load(f)

    d = data.get("data", data)
    raw_layout = d.get("layout", [])
    raw_items = [b for b in raw_layout if b.get("block_label") not in NON_CONTENT_LABELS]
    raw_boxes = [b["bbox"] for b in raw_items]

    order = _detect_bbox_axis_order(raw_boxes)
    pages = d.get("data_info", {}).get("pages", [])
    if not pages:
        raise ValueError(f"{fixture_path}: missing data_info.pages")
    fw = int(pages[0]["width"])
    fh = int(pages[0]["height"])

    blocks: list[GTBlock] = []
    for item in raw_items:
        bbox = item["bbox"]
        if order == "yxyx":
            bbox = _swap_axes(bbox)
        x0, y0, x1, y1 = bbox
        # Clamp and normalize. Use fixture-declared page dims — that's the
        # coord frame the bboxes were written against.
        blocks.append(GTBlock(
            bbox=[
                max(0.0, min(1.0, x0 / fw)),
                max(0.0, min(1.0, y0 / fh)),
                max(0.0, min(1.0, x1 / fw)),
                max(0.0, min(1.0, y1 / fh)),
            ],
            text=(item.get("block_content") or "").strip(),
            page_index=item.get("page_index", 0),
            label=item.get("block_label", "text"),
        ))
    return blocks, (fw, fh)


# --- text similarity -------------------------------------------------------


_WS = re.compile(r"\s+")
_MD_PUNCT = re.compile(r"[*_`#\-]+")


def _normalize_text(s: str) -> str:
    """Lowercase, strip markdown-ish punctuation, collapse whitespace."""
    s = s.lower()
    s = _MD_PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def text_cer(reference: str, hypothesis: str) -> float:
    """Character error rate of `hypothesis` against `reference`.

    Computed on normalized text so punctuation-style differences score
    the same as in :func:`text_similarity`. Capped at 1.0 so a single
    runaway hypothesis can't dominate a document average.
    """
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return min(1.0, float(jiwer.cer(ref, hyp)))


def bag_of_words_f1(
    gt_texts: list[str], pipeline_texts: list[str],
) -> float:
    """Order-independent token-multiset F1 between two text sets.

    Box-assignment-free: scores transcription content even when the
    aligner bound lines to the wrong boxes, which keeps the text axis
    honest on documents where geometry is the weak link.
    """
    gt_tokens = Counter(t for s in gt_texts for t in _normalize_text(s).split())
    pipe_tokens = Counter(t for s in pipeline_texts for t in _normalize_text(s).split())
    n_gt, n_pipe = sum(gt_tokens.values()), sum(pipe_tokens.values())
    if n_gt == 0 and n_pipe == 0:
        return 1.0
    if n_gt == 0 or n_pipe == 0:
        return 0.0
    inter = sum((gt_tokens & pipe_tokens).values())
    p, r = inter / n_pipe, inter / n_gt
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


# --- pseudo-character coverage (CLEval-inspired) ----------------------------


def char_coverage(
    ground_truth: list[GTBlock],
    pipeline_output: list[tuple[BBox, str]],
) -> tuple[float, float]:
    """Split/merge-tolerant character-level coverage, returns
    ``(char_recall, char_precision)``.

    Each block contributes one pseudo-character center per character of
    its text, evenly spaced along the block's horizontal midline (the
    CLEval PCC construction). A GT character counts as recalled when its
    center falls inside ANY pipeline box; a pipeline character counts as
    precise when its center falls inside ANY GT box. No one-to-one
    assignment, so a GT block split across several detected boxes — or
    several GT blocks merged into one detected box — scores by the
    characters actually covered instead of failing an IoU pairing. This
    is the axis that stays fair on forms, where detector granularity
    rarely matches GT block granularity.
    """
    gt_boxes = [g.bbox for g in ground_truth]
    pipe_boxes = [b for b, _t in pipeline_output]
    recall = _pcc_covered(
        ((g.bbox, g.text) for g in ground_truth), pipe_boxes,
    )
    precision = _pcc_covered(pipeline_output, gt_boxes)
    return recall, precision


def _pcc_covered(blocks, target_boxes: list[BBox]) -> float:
    covered = 0
    total = 0
    for bbox, text in blocks:
        n = len(text.strip())
        if n == 0:
            continue
        x0, y0, x1, y1 = bbox
        y_mid = (y0 + y1) / 2
        for k in range(n):
            x = x0 + (k + 0.5) / n * (x1 - x0)
            total += 1
            for t in target_boxes:
                if t[0] <= x <= t[2] and t[1] <= y_mid <= t[3]:
                    covered += 1
                    break
    return covered / total if total else 0.0


# --- matching --------------------------------------------------------------


def compute_report(
    document: str,
    ground_truth: list[GTBlock],
    pipeline_output: list[tuple[BBox, str]],
    iou_threshold: float = 0.3,
    matching: str = "hungarian",
) -> ConfidenceReport:
    """
    Match GT blocks to pipeline blocks one-to-one and score the pairing.

    matching:
        "hungarian" (default): globally optimal assignment maximizing
            total IoU (`scipy.optimize.linear_sum_assignment`); pairs
            below `iou_threshold` are discarded after assignment, per
            the standard detection-eval protocol. Deterministic and
            independent of GT ordering.
        "greedy": legacy first-come best-IoU matching in GT order;
            order-dependent and can under-match, kept only so historical
            reports remain reproducible.

    Miss entries carry the GT block's best IoU against *any* pipeline
    box as a diagnostic (`pipeline_text is None` marks them as misses).
    Small enough inputs (hundreds of blocks) that O(N×M) is fine.
    """
    n_pipe = len(pipeline_output)
    iou_rows = [
        [iou(gt.bbox, pbox) for (pbox, _ptext) in pipeline_output]
        for gt in ground_truth
    ]

    assigned: dict[int, int] = {}  # gt index -> pipeline index
    if matching == "hungarian":
        if ground_truth and pipeline_output:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-np.asarray(iou_rows))
            assigned = {
                int(g): int(p)
                for g, p in zip(rows, cols)
                if iou_rows[g][p] >= iou_threshold
            }
    elif matching == "greedy":
        used: set[int] = set()
        for gi in range(len(ground_truth)):
            best_i, best_iou = -1, 0.0
            for pi in range(n_pipe):
                if pi in used:
                    continue
                if iou_rows[gi][pi] > best_iou:
                    best_iou, best_i = iou_rows[gi][pi], pi
            if best_i >= 0 and best_iou >= iou_threshold:
                used.add(best_i)
                assigned[gi] = best_i
    else:
        raise ValueError(
            f"unknown matching strategy {matching!r}; "
            f"expected 'hungarian' or 'greedy'"
        )

    matches: list[BlockMatch] = []
    for gi, gt in enumerate(ground_truth):
        pi = assigned.get(gi)
        if pi is not None:
            pbox, ptext = pipeline_output[pi]
            matches.append(BlockMatch(
                gt_text=gt.text, gt_bbox=gt.bbox,
                pipeline_text=ptext, pipeline_bbox=pbox,
                iou=iou_rows[gi][pi],
                text_similarity=text_similarity(gt.text, ptext),
                cer=text_cer(gt.text, ptext),
            ))
        else:
            best = max(iou_rows[gi], default=0.0)
            matches.append(BlockMatch(
                gt_text=gt.text, gt_bbox=gt.bbox,
                pipeline_text=None, pipeline_bbox=None,
                iou=best, text_similarity=0.0,
            ))

    recall_pcc, precision_pcc = char_coverage(ground_truth, pipeline_output)
    return ConfidenceReport(
        document=document,
        iou_threshold=iou_threshold,
        gt_count=len(ground_truth),
        pipeline_count=n_pipe,
        matches=matches,
        bow_f1=bag_of_words_f1(
            [g.text for g in ground_truth],
            [t for _b, t in pipeline_output],
        ),
        char_recall=recall_pcc,
        char_precision=precision_pcc,
    )


# --- binary document checks (olmOCR-bench style) ----------------------------
#
# Each fixture document can carry a JSONL rule file; every rule is a
# binary pass/fail judgement against the overlay's concatenated text, so
# a regression is attributable to a specific named fact instead of a
# fuzzy aggregate drop. Rule shapes:
#
#   {"type": "present", "text": "...", "max_diffs": 0, "case_sensitive": false}
#       The text must appear (within `max_diffs` edits, semiglobal).
#   {"type": "absent", "text": "..."}
#       The text must NOT appear anywhere (hallucination guard).
#   {"type": "order", "first": "...", "then": "..."}
#       Both strings must appear, `first` before `then` (reading order).


def load_doc_checks(path: Path | str) -> list[dict]:
    """Load a JSONL rule file; blank lines and `//` comments skipped."""
    rules: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rules.append(json.loads(line))
    return rules


def run_doc_checks(
    text: str, rules: list[dict],
) -> tuple[int, int, list[str]]:
    """Evaluate rules against a document's overlay text.

    Returns ``(passed, total, failure_descriptions)``. Matching is
    whitespace-collapsed and (by default) case-insensitive, but keeps
    punctuation — unlike the similarity normalizer — because checks
    often pin exact tokens like phone numbers and e-mail addresses.
    """
    passed = 0
    failures: list[str] = []
    hay_cs = _checks_normalize(text, case_sensitive=True)
    hay_ci = _checks_normalize(text, case_sensitive=False)

    for rule in rules:
        kind = rule.get("type")
        ok = False
        why = ""
        if kind == "present":
            cs = bool(rule.get("case_sensitive", False))
            hay = hay_cs if cs else hay_ci
            needle = _checks_normalize(rule["text"], case_sensitive=cs)
            max_diffs = int(rule.get("max_diffs", 0))
            if max_diffs <= 0:
                ok = needle in hay
            else:
                ok = _min_substring_edits(needle, hay) <= max_diffs
            why = f"missing: {rule['text']!r}"
        elif kind == "absent":
            needle = _checks_normalize(rule["text"], case_sensitive=False)
            ok = needle not in hay_ci
            why = f"forbidden text present: {rule['text']!r}"
        elif kind == "order":
            first = _checks_normalize(rule["first"], case_sensitive=False)
            then = _checks_normalize(rule["then"], case_sensitive=False)
            i, j = hay_ci.find(first), hay_ci.find(then)
            if i < 0 or j < 0:
                why = (
                    f"order rule operand missing: "
                    f"{rule['first']!r} / {rule['then']!r}"
                )
            elif i >= j:
                why = f"out of order: {rule['first']!r} !< {rule['then']!r}"
            else:
                ok = True
        else:
            why = f"unknown rule type {kind!r}"

        if ok:
            passed += 1
        else:
            failures.append(why)
    return passed, len(rules), failures


_CHECKS_WS = re.compile(r"\s+")


def _checks_normalize(s: str, case_sensitive: bool) -> str:
    s = _CHECKS_WS.sub(" ", s).strip()
    return s if case_sensitive else s.lower()


def _min_substring_edits(needle: str, haystack: str) -> int:
    """Minimum edits for `needle` to match ANY substring of `haystack`
    (semiglobal alignment: deletions before/after the match are free)."""
    n, h = len(needle), len(haystack)
    if n == 0:
        return 0
    if h == 0:
        return n
    prev = [0] * (h + 1)  # free start anywhere in the haystack
    for i in range(1, n + 1):
        cur = [i] + [0] * h
        for j in range(1, h + 1):
            cost = 0 if needle[i - 1] == haystack[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev)  # free end anywhere in the haystack
