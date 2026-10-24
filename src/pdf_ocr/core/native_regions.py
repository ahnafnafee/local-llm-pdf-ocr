"""Source-driven region planning and placement for native OCR models.

The detector supplies geometry only. Every word still comes from the selected
vision client; neither references nor document-specific vocabulary is used.
"""

from dataclasses import replace
from collections import Counter
from statistics import median
from difflib import SequenceMatcher
import re

from pdf_ocr.core._layout import split_multi_line_bbox, searchable_text, table_cell_segments


class RegionBlocks(list):
    """Recognized text plus excluded visual regions that may contain labels."""

    def __init__(self, blocks=(), *, visual_regions=()):
        super().__init__(blocks)
        self.visual_regions = list(visual_regions)


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def enclosed(box, blocks, fraction=.35):
    return any(intersection(box, block.bbox) >= area(box) * fraction for block in blocks)


def has_ink(source, box, minimum=.004):
    crop = source.crop(tuple(round(v * source.size[i % 2]) for i, v in enumerate(box)))
    if not crop.width or not crop.height:
        return False
    histogram = crop.convert('L').histogram()
    # Pale notebook grids and background speckles are not text detections.
    return sum(histogram[:160]) / (crop.width * crop.height) >= minimum


def region_bounds(boxes, padding=.008):
    return [max(0., min(b[a] for b in boxes) - padding) if a < 2
            else min(1., max(b[a] for b in boxes) + padding) for a in range(4)]


def plan_regions(boxes, limit=12):
    """Partition at whitespace gutters, retaining nearby lines as context."""
    if not boxes:
        return []
    if len(boxes) <= limit:
        return [boxes]
    candidates = []
    for axis in (0, 1):
        ordered = sorted(boxes, key=lambda b: b[axis])
        end = ordered[0][axis + 2]
        for i in range(1, len(ordered)):
            gap = ordered[i][axis] - end
            if gap > .001:
                balance = min(i, len(boxes) - i) / len(boxes)
                candidates.append((gap * (1.4 if axis == 0 else 1) * (.5 + balance),
                                   ordered[:i], ordered[i:]))
            end = max(end, ordered[i][axis + 2])
    if candidates:
        _, first, second = max(candidates, key=lambda c: c[0])
    else:
        ordered = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        first, second = ordered[:len(boxes) // 2], ordered[len(boxes) // 2:]
    return plan_regions(first, limit) + plan_regions(second, limit)


def repair_targets(blocks, boxes, source, visual_regions):
    boxes = [b for b in boxes if area(b) > 0 and has_ink(source, b)]
    if not boxes:
        return False, []
    # Small text in many columns needs local coordinates even when the
    # full-page response is syntactically valid and appears complete.
    dense = (len(boxes) >= 45 and median(b[3] - b[1] for b in boxes) < .03
             and median(b[2] - b[0] for b in boxes) < .4)
    if dense:
        return True, boxes
    missing = [box for box in boxes if not enclosed(box, blocks)]
    # Native image descriptions must never become OCR text. Read the source
    # labels inside these regions instead, including embedded whiteboards.
    targets = [box for box in missing if any(intersection(box, v) >= area(box) * .5
                                              for v in visual_regions)]
    if len(missing) >= 3 and len(missing) / len(boxes) >= .15:
        targets = missing
    return False, targets


def anchor_local_blocks(blocks, boxes):
    """Use nearby detected lines to correct coarse local model rectangles."""
    from scipy.optimize import linear_sum_assignment

    lines = []
    for block in blocks:
        parts = table_cell_segments(block.bbox, block.text)
        parts = parts or ([(block.bbox, block.text)] if ('<table' in block.text.lower()
                                                       or '$$' in block.text or r'\[' in block.text)
                 else split_multi_line_bbox(block.bbox, block.text))
        lines.extend(replace(block, bbox=b, text=t) for b, t in parts)
    if not lines or not boxes:
        return lines
    if len(lines) == len(boxes) == 1 and '<table' not in lines[0].text.lower():
        # An isolated source line has no competing placement. Tiny padded
        # crops can shift native coordinates above the ink, so overlap alone
        # would reject the very detection that selected this crop.
        return [replace(lines[0], bbox=boxes[0])]
    scores = []
    for block in lines:
        row = []
        a = block.bbox
        for b in boxes:
            y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            yh = max(1e-8, min(a[3] - a[1], b[3] - b[1]))
            xw = max(1e-8, min(a[2] - a[0], b[2] - b[0]))
            center_distance = abs((a[1] + a[3] - b[1] - b[3]) / 2)
            center_score = max(0., 1 - center_distance / max(a[3]-a[1], b[3]-b[1], 1e-8))
            score = .45 * y_overlap / yh + .35 * center_score + .2 * x_overlap / xw
            if y_overlap / yh < .35 or x_overlap / xw < .25 or '<table' in block.text.lower():
                score = 0
            row.append(score)
        scores.append(row)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    for row, col in zip(rows, cols):
        score = scores[row][col]
        alternatives = [s for i, s in enumerate(scores[row]) if i != col]
        if score >= .55 and (not alternatives or score - max(alternatives) >= .05):
            lines[row] = replace(lines[row], bbox=boxes[col])
    return lines


def deduplicate(blocks):
    """Remove only repeated text at overlapping source positions."""
    kept = []
    for block in blocks:
        key = ' '.join(searchable_text(block.text).casefold().split())
        duplicate = None
        for i, (old, old_key) in enumerate(kept):
            overlap = intersection(block.bbox, old.bbox) / max(1e-8, min(area(block.bbox), area(old.bbox)))
            shorter = min(len(key), len(old_key))
            partial = (shorter >= 35 and overlap >= .35 and
                       SequenceMatcher(None, key, old_key).find_longest_match().size >= .8 * shorter)
            if (key == old_key and overlap >= .4) or partial:
                duplicate = i
                if len(key) > len(old_key):
                    kept[i] = (block, key)
                break
        if duplicate is not None:
            continue
        kept.append((block, key))
    return [block for block, _ in kept]


def superseded_overview(block, recovered):
    """Recognize an overview duplicate whose coordinates drifted sideways.

    Dense page grounding can invent narrow columns in an embedded text panel.
    A long, nearly identical nearby source read replaces that overview block;
    short repeated labels and repetitions on distant rows remain independent.
    """
    text = re.sub(r'\W+', ' ', searchable_text(block.text).casefold()).strip()
    if len(text) < 20:
        return False
    cy = (block.bbox[1] + block.bbox[3]) / 2
    for other in recovered:
        if abs(cy - (other.bbox[1] + other.bbox[3]) / 2) > .08:
            continue
        candidate = re.sub(r'\W+', ' ', searchable_text(other.text).casefold()).strip()
        if SequenceMatcher(None, text, candidate).ratio() >= .8:
            return True
    return False


def merge_refined(blocks, refined, boxes):
    """Replace a displaced coarse cell with its unique source-anchored reread.

    Equal-width table intervals can put a short label beside its actual ink.
    A child crop can recover that label, but overlap-only deduplication misses
    the old copy. Retain anchored, repeated and distant labels independently.
    """
    def key(block):
        return ' '.join(searchable_text(block.text).casefold().split())

    old_counts = Counter(map(key, blocks))
    new_counts = Counter(map(key, refined))
    anchors = {key(block): block for block in refined if block.bbox in boxes}
    retained = []
    for block in blocks:
        text = key(block)
        other = anchors.get(text)
        if (not text or other is None or old_counts[text] != 1 or new_counts[text] != 1
                or any(intersection(block.bbox, box) >= area(box)*.35 for box in boxes)):
            retained.append(block)
            continue
        a, b = block.bbox, other.bbox
        y_overlap = max(0, min(a[3], b[3])-max(a[1], b[1]))
        same_row = y_overlap >= .5 * min(a[3]-a[1], b[3]-b[1])
        x_gap = max(0, b[0]-a[2], a[0]-b[2])
        if not same_row or x_gap > .015:
            retained.append(block)
    return retained + refined


async def recover_regions(source, blocks, boxes, visual_regions, recognize):
    """Reread selected regions with bounded subdivision, then merge by location.

    ``recognize`` receives normalized source bounds and a token budget, and
    returns blocks already mapped back into the source frame. The caller owns
    the page deadline, client and cancellation lifecycle.
    """
    from pdf_ocr.inference import IncompleteCompletionError

    dense, targets = repair_targets(blocks, boxes, source, visual_regions)
    if not targets:
        return blocks
    recovered = []
    successful_regions = []
    successful_boxes = []

    async def read(group, depth=0):
        bounds = region_bounds(group)
        visual = []
        try:
            result = await recognize(bounds, max(512, min(2048, len(group) * 128)))
            visual = getattr(result, 'visual_regions', [])
        except (ValueError, IncompleteCompletionError):
            if depth >= 2 or len(group) == 1:
                raise
            result = []
        result = anchor_local_blocks(result, group)
        # Geometry alone is not evidence that a local transcription omitted
        # a word. Re-reading every unmatched box as an isolated crop both
        # duplicates text and loses handwriting context. Subdivide only a
        # failed read or an explicitly excluded visual region.
        omitted = [b for b in group if not enclosed(b, result) and
                   (not result or any(intersection(b, v) >= area(b) * .5 for v in visual))]
        if omitted and depth < 2:
            # A diagram can remain an image even after a contextual crop.
            # Smaller source regions expose its literal fields and labels.
            for child in plan_regions(omitted, limit=max(1, len(group) // 3)):
                result = merge_refined(result, await read(child, depth + 1), group)
        return deduplicate(result)

    for group in plan_regions(targets):
        result = await read(group)
        if result:
            successful_regions.append(region_bounds(group, padding=0))
            successful_boxes.extend(group)
            recovered.extend(result)
    if not recovered:
        return blocks
    retained = []
    for block in blocks:
        cx, cy = (block.bbox[0] + block.bbox[2]) / 2, (block.bbox[1] + block.bbox[3]) / 2
        replaced = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in successful_regions)
        if dense:
            # Retain undetected content such as an embedded printed question,
            # but discard displaced boxes in empty margins of a dense page.
            replaced = (replaced or any(intersection(block.bbox, b) > 0 for b in successful_boxes)
                        or not has_ink(source, block.bbox, minimum=.03)
                        or superseded_overview(block, recovered))
        if not replaced:
            retained.append(block)
    result = deduplicate(retained + recovered)
    return sorted(result, key=lambda b: (b.bbox[1], b.bbox[0]))
