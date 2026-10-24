"""Conservative geometry anchors for recognizers that return native boxes."""

from pdf_ocr.core._layout import split_multi_line_bbox
from difflib import SequenceMatcher
import re


_INLINE_IDENTIFIER = re.compile(r"\$([A-Za-z][A-Za-z_{}]{1,15})\$")


def has_inline_identifier(text):
    """An isolated subscript-like identifier inside prose needs a literal view."""
    for line in text.splitlines():
        matches = list(_INLINE_IDENTIFIER.finditer(line))
        if (len(matches) == 1 and "_" in matches[0][1]
                and len(re.findall(r"[A-Za-z]+", _INLINE_IDENTIFIER.sub("", line))) >= 3):
            return True
    return False


def restore_literal_identifiers(text, transcription):
    """Replace invented inline notation only with a uniquely anchored acronym.

    Both readings come from the same source image through the model's native
    modes. Require identical surrounding prose, one uppercase alternative,
    and similar letters. Never infer an acronym from a dictionary or fixture.
    Equations, numeric subscripts and conflicting page reads remain unchanged.
    """
    def prose(line):
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        return re.sub(r"^[#*+•\-\s]+", "", " ".join(line.split()))

    candidates = [prose(line) for line in transcription.splitlines()]
    result = []
    for line in text.splitlines():
        if not has_inline_identifier(line):
            result.append(line)
            continue
        match = _INLINE_IDENTIFIER.search(line)
        letters = re.sub(r"[^A-Za-z]", "", match[1])
        if not 2 <= len(letters) <= 8:
            result.append(line)
            continue
        marked = prose(line[:match.start()] + "OCRIDENTIFIER" + line[match.end():])
        pattern = re.escape(marked).replace("OCRIDENTIFIER", r"([A-Za-z]{2,8})")
        alternatives = set()
        for candidate in candidates:
            found = re.fullmatch(pattern, candidate, re.IGNORECASE)
            if found:
                word = found[1]
                if (word.isupper() and len(word) == len(letters)
                        and SequenceMatcher(None, letters.casefold(), word.casefold()).ratio() >= .5):
                    alternatives.add(word)
        if len(alternatives) == 1:
            line = line[:match.start()] + alternatives.pop() + line[match.end():]
        result.append(line)
    return "\n".join(result)


def _iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
    return overlap / union if union > 0 else 0.0


def anchor_native_blocks(blocks, detected_boxes):
    """Snap unambiguous native lines to detected ink without changing words.

    Keep native reading order and every recognized block. Detection alone
    cannot supply text for an omitted region. A detector box may anchor only
    one line; ambiguous or differently segmented regions keep model geometry.
    Tables retain their structural markup and enclosing native rectangle.
    """
    result = []
    used = set()
    for bbox, text in blocks:
        lines = ([(bbox, text)] if "<table" in text.lower()
                 else split_multi_line_bbox(bbox, text))
        for line_box, line in lines:
            ranked = sorted(((_iou(line_box, box), i) for i, box in enumerate(detected_boxes)
                             if i not in used), reverse=True)
            if (ranked and ranked[0][0] >= 0.5
                    and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.1)):
                index = ranked[0][1]
                line_box = detected_boxes[index]
                used.add(index)
            result.append((line_box, line))
    return result
