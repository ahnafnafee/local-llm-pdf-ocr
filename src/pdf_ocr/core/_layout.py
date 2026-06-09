"""
Shared bbox-layout helpers used by every output writer.

Two corner cases need special handling before any writer (PDF, HTML, …)
can position glyphs / spans:

1. **Full-page fallback** — the aligner emits a [0,0,1,1] bbox containing
   '\\n'-joined text when the LLM produced output but Surya found no
   layout boxes to attach it to. A writer must NOT render this bbox as
   a single page-spanning text element; doing so clobbers other bboxes'
   search positions and is the failure mode this helper guards against.

2. **Multi-line text in a real bbox** — grounded VLMs sometimes emit
   visually-adjacent lines joined by '\\n' as the text for one bbox.
   Search / selection only works correctly if each visual line lands at
   its own y position, so the bbox is split vertically into N sub-rects
   proportional to the line count.

Both helpers are pure: no I/O, no rendering. The PDF and HTML writers
call them and decide how to render each sub-rect themselves.
"""

from __future__ import annotations


def is_full_page_fallback(rect_coords: list[float], text: str) -> bool:
    """True if `rect_coords` is the aligner's [0,0,1,1] fallback bbox.

    The fallback is only meaningful when the text contains line breaks —
    a grounded VLM emitting a [0,0,1,1] bbox with single-line text is
    unusual and not a fallback case, so we require '\\n' in `text` too.
    """
    if "\n" not in text:
        return False
    nx0, ny0, nx1, ny1 = rect_coords
    return (
        nx0 <= 0.001 and ny0 <= 0.001
        and nx1 >= 0.999 and ny1 >= 0.999
    )


def split_multi_line_bbox(
    rect_coords: list[float],
    text: str,
) -> list[tuple[list[float], str]]:
    """Split a bbox whose text has embedded '\\n's into per-line sub-bboxes.

    Each sub-bbox gets a proportional vertical slice of the input bbox.
    Empty / whitespace-only lines are dropped.

    Returns:
        - `[]` if `text` is empty / whitespace
        - `[(rect_coords_copy, stripped_text)]` if no split is needed
          (no `'\\n'` in text, or only one non-empty line after stripping)
        - `[(sub_rect_i, line_i), ...]` with N entries when the bbox is
          split into N visual lines
    """
    text = (text or "").strip()
    if not text:
        return []

    if "\n" not in text:
        return [(list(rect_coords), text)]

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        return [(list(rect_coords), lines[0])]

    nx0, ny0, nx1, ny1 = rect_coords
    slice_h = (ny1 - ny0) / len(lines)
    return [
        (
            [nx0, ny0 + i * slice_h, nx1, ny0 + (i + 1) * slice_h],
            line,
        )
        for i, line in enumerate(lines)
    ]
