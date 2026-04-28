"""
HybridAligner - Detection-only OCR aligner.

Uses Surya's `DetectionPredictor` (fast) for bounding boxes and binds
LLM-produced text to those boxes using a monotonic Needleman-Wunsch DP
over (LLM lines, detected boxes). Skipping Surya's recognition step is
~10-21x faster; the LLM supplies the text quality.
"""

import io
import logging

from src.pdf_ocr.utils import tqdm_patch

# Silence Surya's progress bars so they don't collide with Rich.
tqdm_patch.apply()

from PIL import Image  # noqa: E402
from surya.detection import DetectionPredictor  # noqa: E402


BBox = list[float]  # [nx0, ny0, nx1, ny1], normalized to 0..1


class HybridAligner:
    """
    Detection-only aligner with DP-based line-to-box alignment.

    Phase 1 (detect): Surya produces layout boxes, no text.
    Phase 2 (align): LLM-produced lines are mapped to boxes using
    Needleman-Wunsch, with cost proportional to how well each line's
    character count matches each box's estimated capacity (area-based),
    monotonic so reading order is preserved. Skip ops on both sides
    handle missing/extra lines and boxes. Unmatched LLM lines are
    attached to the nearest matched box so no text is lost.
    """

    def __init__(self):
        self.detection_predictor = DetectionPredictor()

    def get_detected_boxes_batch(self, images_bytes_list: list[bytes]) -> list[list[BBox]]:
        """
        Run Surya detection on a batch of page images in a single call.
        Returns one list of normalized boxes per page, sorted in reading order.
        """
        if not images_bytes_list:
            return []

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes_list]
        sizes = [img.size for img in images]
        predictions = self.detection_predictor(images)

        all_boxes: list[list[BBox]] = []
        for (img_w, img_h), pred in zip(sizes, predictions):
            boxes: list[BBox] = []
            for bbox in (pred.bboxes or []):
                x0, y0, x1, y1 = bbox.bbox
                boxes.append([
                    _clamp(x0 / img_w),
                    _clamp(y0 / img_h),
                    _clamp(x1 / img_w),
                    _clamp(y1 / img_h),
                ])
            all_boxes.append(_reading_order_sort(boxes))
        return all_boxes

    def align_text(
        self,
        structured_data: list[tuple[BBox, str]],
        llm_text,
    ) -> list[tuple[BBox, str]]:
        """
        Bind LLM text to detected boxes via Needleman-Wunsch line-to-box DP.

        Always returns one (box, text) tuple per input box, in input order.
        `text` is "" for boxes the algorithm could not confidently match —
        the pipeline uses that signal to trigger per-box crop re-OCR.
        """
        lines = _normalize_lines(llm_text)
        boxes = [item[0] for item in structured_data]

        if not boxes and not lines:
            return []
        if not boxes:
            # Degenerate: LLM produced text but Surya found nothing.
            # Embed all text in a full-page box so search still works.
            return [([0.0, 0.0, 1.0, 1.0], "\n".join(lines))]
        if not lines:
            return [(box, "") for box in boxes]

        mapping = _dp_align(lines, boxes)
        result: list[tuple[BBox, str]] = []
        for i, box in enumerate(boxes):
            texts = mapping.get(i, [])
            result.append((box, " ".join(texts).strip()))

        matched = sum(1 for _, t in result if t)
        logging.debug(
            f"DEBUG: DP aligned {len(lines)} lines → {matched}/{len(boxes)} boxes"
        )
        return result


# --- module-level helpers ---------------------------------------------------


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# Threshold (in normalized x-center space) above which a gap between
# consecutive boxes is treated as a column break. 2-column page gutters
# tend to push x-center distance well past 0.2 (typical column-center
# separation is ~0.3-0.5). Lower values risk treating a single marginal
# box (page number, sidebar note) as its own column and re-ordering it
# ahead of body text.
_COLUMN_GAP_THRESHOLD = 0.2


def _reading_order_sort(boxes: list[BBox]) -> list[BBox]:
    """
    Sort boxes in column-major reading order.

    Vision LLMs (OlmOCR-2, Qwen-VL, etc.) typically emit text column by
    column on multi-column pages — the entire left column first, then the
    next column. Surya's natural row-major output interleaves rows across
    columns, which mis-aligns LLM lines to boxes in the DP. This function
    detects column structure via a gap-based split on box x-centers and
    re-orders each column independently so the resulting sequence matches
    what the LLM produced.

    Heuristic: split at the largest x-center gap if it exceeds
    ``_COLUMN_GAP_THRESHOLD`` AND both sides hold ≥2 boxes. The size
    constraint prevents a lone marginal box (e.g. a page number) from
    creating a fake column that swaps reading order. Recurses to handle
    3+ column layouts; falls back to plain row-major for single-column
    pages and very short sequences.
    """
    if len(boxes) < 4:
        return sorted(boxes, key=lambda b: (b[1], b[0]))

    indexed = sorted(
        enumerate(boxes), key=lambda ib: (ib[1][0] + ib[1][2]) / 2
    )
    centers_sorted = [(b[0] + b[2]) / 2 for _, b in indexed]

    biggest_gap = 0.0
    biggest_gap_idx = -1
    for i in range(1, len(centers_sorted)):
        gap = centers_sorted[i] - centers_sorted[i - 1]
        if gap > biggest_gap:
            biggest_gap = gap
            biggest_gap_idx = i

    # Need both groups ≥2 boxes to call it a real column split.
    if (
        biggest_gap < _COLUMN_GAP_THRESHOLD
        or biggest_gap_idx < 2
        or biggest_gap_idx > len(centers_sorted) - 2
    ):
        return sorted(boxes, key=lambda b: (b[1], b[0]))

    left_boxes = [b for _, b in indexed[:biggest_gap_idx]]
    right_boxes = [b for _, b in indexed[biggest_gap_idx:]]
    return _reading_order_sort(left_boxes) + _reading_order_sort(right_boxes)


def _normalize_lines(llm_text) -> list[str]:
    """Accept str or list[str]; return non-empty stripped lines."""
    if not llm_text:
        return []
    if isinstance(llm_text, str):
        raw = llm_text.split("\n")
    else:
        raw = []
        for item in llm_text:
            raw.extend(str(item).split("\n"))
    return [s.strip() for s in raw if s and s.strip()]


# Cost constants — tuned to favor matching over skipping, with cheap box-skips
# (many detected boxes are decorative / rules / empty regions) and relatively
# expensive line-skips (LLM text should all land somewhere).
_SKIP_LINE_COST = 1.0
_SKIP_BOX_COST = 0.4


def _estimated_capacities(boxes: list[BBox]) -> list[float]:
    """
    Estimate each box's character capacity from its normalized area.

    We don't know the actual font size, so we calibrate relatively: compute
    area and treat the total area as proportional to total chars. What matters
    for the DP is the relative capacity across boxes, not absolute chars.
    """
    # Area proxy for "how much text fits here". Taller boxes hold wrapped
    # text; wider boxes hold long lines. Using area captures both.
    areas = [max(1e-6, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
    return areas


def _match_cost(line_chars: int, expected_chars: float) -> float:
    """
    Relative char-count mismatch cost, clipped to [0, 2].

    Normalized by the larger of actual/expected so the cost is symmetric
    (a 10-char line on a 100-char box is equally "wrong" as vice versa).
    """
    expected = max(1.0, expected_chars)
    actual = max(1, line_chars)
    denom = max(actual, expected)
    return min(2.0, abs(actual - expected) / denom)


def _dp_align(lines: list[str], boxes: list[BBox]) -> dict[int, list[str]]:
    """
    Monotonic Needleman-Wunsch alignment of lines → boxes.

    Returns: {box_index: [line_text, ...]} in alignment order.
    Unmatched lines are attached to the nearest preceding matched box
    (or the first box if no prior match exists) so no LLM text is lost.
    """
    N = len(lines)
    M = len(boxes)
    if N == 0 or M == 0:
        return {}
    total_chars = max(1, sum(len(l) for l in lines))

    # Distribute total chars across boxes by area.
    caps = _estimated_capacities(boxes)
    total_cap = sum(caps)
    expected = [c / total_cap * total_chars for c in caps]

    INF = float("inf")
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    # Back-pointer ops: 0 = match, 1 = skip_line, 2 = skip_box
    back = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    # Boundary rows/cols: cumulative gap penalties.
    for j in range(1, M + 1):
        dp[0][j] = dp[0][j - 1] + _SKIP_BOX_COST
        back[0][j] = 2
    for i in range(1, N + 1):
        dp[i][0] = dp[i - 1][0] + _SKIP_LINE_COST
        back[i][0] = 1

    for i in range(1, N + 1):
        li = lines[i - 1]
        for j in range(1, M + 1):
            m_cost = dp[i - 1][j - 1] + _match_cost(len(li), expected[j - 1])
            sl_cost = dp[i - 1][j] + _SKIP_LINE_COST
            sb_cost = dp[i][j - 1] + _SKIP_BOX_COST

            best = m_cost
            op = 0
            if sl_cost < best:
                best, op = sl_cost, 1
            if sb_cost < best:
                best, op = sb_cost, 2
            dp[i][j] = best
            back[i][j] = op

    # Backtrack to produce the ordered op list.
    mapping: dict[int, list[str]] = {}
    i, j = N, M
    ops: list[tuple[int, int, int]] = []  # (op, line_idx, box_idx)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == 0 and i > 0 and j > 0:
            ops.append((0, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op == 1 and i > 0:
            ops.append((1, i - 1, j - 1 if j > 0 else -1))
            i -= 1
        elif op == 2 and j > 0:
            ops.append((2, i - 1 if i > 0 else -1, j - 1))
            j -= 1
        else:  # safety: shouldn't happen but avoids infinite loop
            if i > 0:
                i -= 1
            elif j > 0:
                j -= 1
    ops.reverse()

    # Replay ops in reading order. Track the most recent matched box so that
    # skip_line ops can attach their text to it (no lost LLM text).
    last_matched_box: int | None = None
    for op, li, bj in ops:
        if op == 0:
            mapping.setdefault(bj, []).append(lines[li])
            last_matched_box = bj
        elif op == 1 and li >= 0:
            # Unmatched line: attach to last matched box, or to first box
            # if we haven't matched anything yet.
            target = last_matched_box if last_matched_box is not None else 0
            mapping.setdefault(target, []).append(lines[li])
        # op == 2 (skip_box): nothing to add for this box
    return mapping
