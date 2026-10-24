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


def enhance_faint_ink(image):
    """Strengthen sparse, light colored strokes without changing geometry.

    Minimum-channel contrast makes cyan/yellow handwriting legible while
    leaving neutral pixels unchanged. Require a light background and thin,
    sparse colored marks; filled highlights and colored panels stay intact.
    Only regional inference views use this, never the document background.
    """
    from PIL import ImageChops, ImageFilter

    gray = image.convert('L')
    pixels = image.width * image.height
    if not pixels or sum(gray.histogram()[220:]) / pixels < .85:
        return image
    red, green, blue = image.convert('RGB').split()
    low = ImageChops.darker(ImageChops.darker(red, green), blue)
    high = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    mask = ImageChops.multiply(
        ImageChops.subtract(high, low).point(lambda value: 255 if value >= 60 else 0),
        gray.point(lambda value: 255 if value >= 140 else 0))
    colored = mask.histogram()[255]
    if not .0005 <= colored / pixels <= .03:
        return image
    # Filled patches retain a substantial interior after erosion; pen strokes
    # do not. Darkening a highlight would hide the black text underneath it.
    if mask.filter(ImageFilter.MinFilter(5)).histogram()[255] / colored > .2:
        return image
    return low.convert('RGB')


async def detect_missing_lines(source, boxes, detect):
    """Supplement dense-page geometry using four overlapping source views.

    Tiny embedded print can disappear when the page detector downsamples a
    whole sheet. Zoomed detection supplies only missing geometry; existing
    lines remain unchanged and overlapping tiles cannot duplicate a line.
    """
    import io

    regions = [[0, 0, .55, .55], [.45, 0, 1, .55],
               [0, .45, .55, 1], [.45, .45, 1, 1]]
    images = []
    for region in regions:
        crop = source.crop(tuple(round(v * source.size[i % 2]) for i, v in enumerate(region)))
        buffer = io.BytesIO()
        crop.save(buffer, format='PNG')
        images.append(buffer.getvalue())
    result = list(boxes)
    for region, detected in zip(regions, await detect(images)):
        for box in detected:
            mapped = [region[i % 2] + box[i] * (region[2 + i % 2] - region[i % 2])
                      for i in range(4)]
            size = area(mapped)
            # Reject rediscoveries that split or merge existing lines. A large
            # merged box must not compete with the original precise anchors.
            if (size > 0 and has_ink(source, mapped)
                    and not any(area(old) > 0 and
                                intersection(mapped, old) >= min(size, area(old)) * .5
                                for old in result)):
                result.append(mapped)
    return result


def region_bounds(boxes, padding=.008, *, neighbors=()):
    core = [min(b[a] for b in boxes) if a < 2 else max(b[a] for b in boxes)
            for a in range(4)]
    margins = [padding] * 4
    for other in neighbors:
        for axis in (0, 1):
            cross = 1 - axis
            if min(core[cross+2], other[cross+2]) <= max(core[cross], other[cross]):
                continue
            # Padding must not include a clipped fragment of the next line.
            # Such fragments can make a native recognizer invent its ending.
            if other[axis+2] <= core[axis]:
                margins[axis] = min(margins[axis], (core[axis]-other[axis+2])/2)
            if other[axis] >= core[axis+2]:
                margins[axis+2] = min(margins[axis+2], (other[axis]-core[axis+2])/2)
    return [max(0., core[a]-margins[a]) if a < 2 else min(1., core[a]+margins[a])
            for a in range(4)]


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


def _stacked_rows(rectangles):
    """Recognize one column, including short rows with a hanging indent."""
    return ((max(b[0] for b in rectangles) < min(b[2] for b in rectangles)
             or max(b[0] for b in rectangles)-min(b[0] for b in rectangles)
                <= .15 * max(b[2]-b[0] for b in rectangles))
            and all(a[3]-b[1] <= .25 * min(a[3]-a[1], b[3]-b[1])
                    for a, b in zip(rectangles, rectangles[1:])))


def _expand_paragraph_rows(lines, boxes):
    """Restore source rows when a native paragraph joins several print lines.

    Require a regular column and nearby ordered blocks. Relative block heights
    allocate the detected rows; their widths distribute the unchanged words.
    Inline math stays atomic, and display math/tables/code keep native layout.
    """
    if not 0 < len(lines) < len(boxes):
        return None
    ordered = sorted(lines, key=lambda b: b.bbox[1]+b.bbox[3])
    boxes = sorted(boxes, key=lambda b: b[1]+b[3])
    heights = [b[3]-b[1] for b in boxes]
    if (min(heights) <= 0 or max(heights) > 2 * min(heights)
            or not _stacked_rows(boxes) or not _stacked_rows([b.bbox for b in ordered])
            or any(re.search(r'<table|\$\$|\\\[|`', b.text, re.I) for b in ordered)):
        return None
    weights = [b.bbox[3]-b.bbox[1] for b in ordered]
    if min(weights) <= 0:
        return None
    ideal = [weight/sum(weights)*len(boxes) for weight in weights]
    counts = [max(1, int(value)) for value in ideal]
    if sum(counts) > len(boxes):
        return None
    while sum(counts) < len(boxes):
        index = max(range(len(counts)), key=lambda i: ideal[i]-counts[i])
        counts[index] += 1
    import fitz

    font = fitz.Font('tiro')
    result, start = [], 0
    for block, count in zip(ordered, counts):
        rows = boxes[start:start+count]
        start += count
        native = block.bbox
        target = region_bounds(rows, padding=0)
        if (abs(native[1]+native[3]-target[1]-target[3])/2
                > max(native[3]-native[1], target[3]-target[1])
                or min(native[2], target[2]) <= max(native[0], target[0])):
            return None
        words = re.findall(r'(?<!\\)\$(?!\$)[^$]+?(?<!\\)\$|\\\([^)]*?\\\)|\S+', block.text)
        if len(words) < count:
            return None
        lengths = [font.text_length(searchable_text(word)+' ', fontsize=1) for word in words]
        widths = [b[2]-b[0] for b in rows]
        total_width, total_length = sum(widths), sum(lengths)
        consumed, end_width = 0, 0
        for index, row in enumerate(rows):
            end_width += widths[index]
            if index == count-1:
                end = len(words)
            else:
                target_length = total_length * end_width / total_width
                end = min(range(consumed+1, len(words)-(count-index-1)+1),
                          key=lambda j: abs(sum(lengths[:j])-target_length))
            result.append(replace(block, bbox=row, text=' '.join(words[consumed:end])))
            consumed = end
    return result


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
    expanded = _expand_paragraph_rows(lines, boxes)
    if expanded is not None:
        return expanded
    if len(lines) == len(boxes) == 1 and '<table' not in lines[0].text.lower():
        # An isolated source line has no competing placement. Tiny padded
        # crops can shift native coordinates above the ink, so overlap alone
        # would reject the very detection that selected this crop.
        return [replace(lines[0], bbox=boxes[0])]
    if len(lines) == len(boxes) and all('<table' not in line.text.lower() for line in lines):
        ordered_lines = sorted(range(len(lines)), key=lambda i: sum(lines[i].bbox[1::2]))
        ordered_boxes = sorted(boxes, key=lambda b: b[1] + b[3])

        ordered_native = [lines[i].bbox for i in ordered_lines]
        if _stacked_rows(ordered_native) and _stacked_rows(ordered_boxes):
            # Native crop coordinates can drift upward by one line height.
            # Equal counts in one clearly ordered column identify the rows
            # even when the last native rectangle misses its own source ink.
            # Require nearby centers and horizontal overlap for every pair;
            # columns, missing rows and distant text keep spatial matching.
            nearby = all(
                abs(a[1]+a[3]-b[1]-b[3])/2 <= max(a[3]-a[1], b[3]-b[1])
                and max(0, min(a[2], b[2])-max(a[0], b[0]))
                    >= .25 * min(a[2]-a[0], b[2]-b[0])
                for a, b in zip(ordered_native, ordered_boxes))
            if nearby:
                for index, box in zip(ordered_lines, ordered_boxes):
                    lines[index] = replace(lines[index], bbox=box)
                return lines
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
        bounds = region_bounds(group, neighbors=boxes)
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
