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
            # Stable row-major default. The actual reading-order choice for
            # the DP happens inside align_text, which tries both row-major
            # and column-major orderings and picks the lower-cost result.
            boxes.sort(key=lambda b: (b[1], b[0]))
            all_boxes.append(boxes)
        return all_boxes

    def align_text(
        self,
        structured_data: list[tuple[BBox, str]],
        llm_text,
    ) -> list[tuple[BBox, str]]:
        """
        Bind LLM text to detected boxes via Needleman-Wunsch line-to-box DP.

        VLMs disagree on reading order: OlmOCR-2 emits column-major on
        multi-column pages (entire left column, then right), while other
        VLMs may emit row-major (top-to-bottom across the whole page).
        We can't normalize this via prompt because OlmOCR's prompt is
        RL-locked. Instead, the DP is run twice — once with boxes in
        row-major order and once in column-major — and whichever gives
        the lower total cost wins. The DP cost itself is a robust signal
        for which ordering matches the LLM's emission, so the same code
        path works for any model without per-model branching.

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

        # Permutations to try: row-major (y, x) and column-major
        # (column groups, then y within column). For single-column pages
        # both collapse to the same order — we still run both, but the DP
        # is O(N*M) which is negligible at typical page sizes (~30 boxes).
        candidates = [
            sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0])),
            _reading_order_indices(boxes),
        ]

        best_cost = float("inf")
        best_perm: list[int] = candidates[0]
        best_mapping: dict[int, list[str]] = {}
        best_match_count = 0
        for perm in candidates:
            ordered_boxes = [boxes[i] for i in perm]
            cost, mapping, match_count = _dp_align(lines, ordered_boxes)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
                best_mapping = mapping
                best_match_count = match_count

        # Degenerate-alignment safety net. Two failure modes pack all the
        # LLM text into one box and leave the rest empty (which users see
        # as "all text packed in the top-left corner"):
        #
        #   1. Zero real matches — every line was skip_line'd and attached
        #      to box 0 by the fallback. Unlikely with default cost
        #      constants but cheap to check.
        #   2. The LLM emitted ONE giant line but Surya found many boxes
        #      (e.g. an OlmOCR variant that doesn't break lines on
        #      handwritten content). The DP can only match that single
        #      line to one box; every other box stays empty. Output is
        #      effectively a single block in whichever box won the match.
        #
        # In both cases, embedding the LLM text in a single full-page bbox
        # keeps the output searchable across the whole page instead of
        # corralled into one corner. We only consider case 2 when there
        # are several boxes — a 1-line / 1-box page is the normal trivial
        # case and should keep the existing placement.
        is_zero_match = best_match_count == 0 and len(lines) > 1
        is_single_line_many_boxes = len(lines) == 1 and len(boxes) >= 5
        if is_zero_match or is_single_line_many_boxes:
            reason = "no line→box matches" if is_zero_match else (
                "LLM emitted a single line for many boxes — likely a model "
                "variant that doesn't break visual lines"
            )
            logging.warning(
                "Degenerate hybrid alignment: %s (lines=%d, boxes=%d). "
                "Falling back to a full-page text layer so output stays "
                "searchable. Try --grounded or a different --model.",
                reason, len(lines), len(boxes),
            )
            return [([0.0, 0.0, 1.0, 1.0], "\n".join(lines))]

        # Translate the per-perm-index mapping back to per-input-index text.
        text_per_input: list[str] = ["" for _ in boxes]
        for perm_idx, texts in best_mapping.items():
            text_per_input[best_perm[perm_idx]] = " ".join(texts).strip()

        logging.debug(
            f"DEBUG: DP aligned {len(lines)} lines → {best_match_count}/{len(boxes)} "
            f"boxes (cost={best_cost:.3f})"
        )
        return [(box, text) for box, text in zip(boxes, text_per_input)]


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


def _reading_order_indices(boxes: list[BBox]) -> list[int]:
    """
    Permutation of ``boxes`` indices in column-major reading order.

    Multi-column pages: split at the largest x-center gap if it exceeds
    ``_COLUMN_GAP_THRESHOLD`` AND both sides hold ≥2 boxes. The size
    constraint prevents a lone marginal box (e.g. a page number) from
    creating a fake column that swaps reading order. Recurses to handle
    3+ column layouts; falls back to plain row-major for single-column
    pages and very short sequences.
    """
    n = len(boxes)
    if n < 4:
        return sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0]))

    sorted_idx = sorted(range(n), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
    centers_sorted = [(boxes[i][0] + boxes[i][2]) / 2 for i in sorted_idx]

    biggest_gap = 0.0
    biggest_gap_pos = -1
    for k in range(1, len(centers_sorted)):
        gap = centers_sorted[k] - centers_sorted[k - 1]
        if gap > biggest_gap:
            biggest_gap = gap
            biggest_gap_pos = k

    if (
        biggest_gap < _COLUMN_GAP_THRESHOLD
        or biggest_gap_pos < 2
        or biggest_gap_pos > len(centers_sorted) - 2
    ):
        return sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0]))

    left_indices = sorted_idx[:biggest_gap_pos]
    right_indices = sorted_idx[biggest_gap_pos:]

    # Recurse on each side, mapping sub-permutations back to global indices.
    left_subboxes = [boxes[i] for i in left_indices]
    right_subboxes = [boxes[i] for i in right_indices]
    left_perm = _reading_order_indices(left_subboxes)
    right_perm = _reading_order_indices(right_subboxes)
    return (
        [left_indices[k] for k in left_perm]
        + [right_indices[k] for k in right_perm]
    )


def _reading_order_sort(boxes: list[BBox]) -> list[BBox]:
    """
    Return ``boxes`` sorted in column-major reading order.

    Thin wrapper around :func:`_reading_order_indices`. Kept as a public-ish
    helper for tests and diagnostic scripts.
    """
    return [boxes[i] for i in _reading_order_indices(boxes)]


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


def _dp_align(
    lines: list[str], boxes: list[BBox]
) -> tuple[float, dict[int, list[str]], int]:
    """
    Monotonic Needleman-Wunsch alignment of lines → boxes.

    Returns: ``(total_cost, mapping, match_count)`` where ``mapping`` is
    ``{box_index: [line_text, ...]}``, ``total_cost`` is the DP's
    minimum alignment cost, and ``match_count`` is the number of
    ``op=match`` operations the DP made (i.e. real line→box pairings,
    excluding lines attached via the skip-line fallback). The cost
    lets callers compare the same alignment under different box
    orderings; ``match_count`` lets callers detect a degenerate
    alignment where every line was skipped and dumped onto box 0.

    Unmatched lines are attached to the nearest preceding matched box
    (or the first box if no prior match exists) so no LLM text is lost.
    """
    N = len(lines)
    M = len(boxes)
    if N == 0 or M == 0:
        return 0.0, {}, 0
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
    match_count = 0
    for op, li, bj in ops:
        if op == 0:
            mapping.setdefault(bj, []).append(lines[li])
            last_matched_box = bj
            match_count += 1
        elif op == 1 and li >= 0:
            # Unmatched line: attach to last matched box, or to first box
            # if we haven't matched anything yet.
            target = last_matched_box if last_matched_box is not None else 0
            mapping.setdefault(target, []).append(lines[li])
        # op == 2 (skip_box): nothing to add for this box
    return dp[N][M], mapping, match_count
