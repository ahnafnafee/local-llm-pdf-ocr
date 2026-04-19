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

Metrics per document:
    - block_recall: fraction of GT blocks matched with IoU >= threshold
    - text_similarity: avg difflib ratio over matched (text-normalized) pairs
    - unmatched: GT blocks with no sufficient pipeline counterpart

Blocks are matched greedily by IoU (best available pipeline box per GT box),
without replacement. Not optimal in the Hungarian sense, but deterministic
and close enough for a confidence summary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

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


@dataclass
class ConfidenceReport:
    document: str
    iou_threshold: float
    gt_count: int
    pipeline_count: int
    matches: list[BlockMatch] = field(default_factory=list)

    @property
    def matched(self) -> list[BlockMatch]:
        return [m for m in self.matches if m.iou >= self.iou_threshold]

    @property
    def block_recall(self) -> float:
        return len(self.matched) / max(1, self.gt_count)

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

    def summary_line(self) -> str:
        return (
            f"{self.document:<20} "
            f"gt={self.gt_count:<3} "
            f"pipeline={self.pipeline_count:<3} "
            f"matched={len(self.matched):<3} "
            f"recall={self.block_recall:.2f} "
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


# --- matching --------------------------------------------------------------


def compute_report(
    document: str,
    ground_truth: list[GTBlock],
    pipeline_output: list[tuple[BBox, str]],
    iou_threshold: float = 0.3,
) -> ConfidenceReport:
    """
    Greedy best-IoU matching of GT blocks to pipeline blocks, no re-use.

    Small enough inputs (tens of blocks per page) that O(N×M) is fine.
    """
    used: set[int] = set()
    matches: list[BlockMatch] = []
    for gt in ground_truth:
        best_i, best_iou = -1, 0.0
        for i, (pbox, _ptext) in enumerate(pipeline_output):
            if i in used:
                continue
            score = iou(gt.bbox, pbox)
            if score > best_iou:
                best_iou, best_i = score, i
        if best_i >= 0 and best_iou >= iou_threshold:
            used.add(best_i)
            pbox, ptext = pipeline_output[best_i]
            matches.append(BlockMatch(
                gt_text=gt.text, gt_bbox=gt.bbox,
                pipeline_text=ptext, pipeline_bbox=pbox,
                iou=best_iou,
                text_similarity=text_similarity(gt.text, ptext),
            ))
        else:
            matches.append(BlockMatch(
                gt_text=gt.text, gt_bbox=gt.bbox,
                pipeline_text=None, pipeline_bbox=None,
                iou=best_iou, text_similarity=0.0,
            ))
    return ConfidenceReport(
        document=document,
        iou_threshold=iou_threshold,
        gt_count=len(ground_truth),
        pipeline_count=len(pipeline_output),
        matches=matches,
    )
