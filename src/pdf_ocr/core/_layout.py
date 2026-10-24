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

from html.parser import HTMLParser
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _math_converter():
    from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec, EnvironmentTextSpec, get_default_latex_context_db

    context = get_default_latex_context_db()
    context.add_context_category('ocr-symbols', macros=[MacroTextSpec('checkmark', '✓'),
                                                     MacroTextSpec('boxed', '%s'),
                                                     MacroTextSpec('textcircled', '%s')],
                                 environments=[EnvironmentTextSpec(name, discard=False)
                                               for name in ('aligned', 'cases')], prepend=True)
    return LatexNodes2Text(latex_context=context, math_mode='text',
                          strict_latex_spaces='based-on-source')


@lru_cache(maxsize=1)
def _math_parser_context():
    from pylatexenc.latexwalker import get_default_latex_context_db
    from pylatexenc.macrospec import MacroSpec, EnvironmentSpec

    context = get_default_latex_context_db()
    context.add_context_category('ocr-symbols', macros=[MacroSpec('boxed', '{'),
                                                      MacroSpec('textcircled', '{')],
                                 environments=[EnvironmentSpec(name, is_math_mode=True)
                                               for name in ('aligned', 'cases')], prepend=True)
    return context


_MATH = re.compile(r'(?<!\\)\$\$[\s\S]+?(?<!\\)\$\$|(?<![\\$])\$(?!\$)[^\n$]+?(?<!\\)\$(?!\$)'
                   r'|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]')
_CODE = re.compile(r'(```[\s\S]*?(?:```|$)|`[^`\n]*`)')


def _visible_math(text: str) -> str:
    def convert(match):
        source = match[0]
        # Currency pairs are not a mathematical environment.
        if source.startswith('$') and re.match(r'\$\d', source) and '\\' not in source:
            return source
        converter = _math_converter()
        # Unknown commands can carry meaningful text. Preserve the complete
        # expression instead of letting a permissive converter erase them.
        for macro in re.findall(r'\\([a-zA-Z]+)', source):
            if macro not in {'left', 'right', 'begin', 'end'} and converter.latex_context.get_macro_spec(macro) is None:
                return source
        for environment in re.findall(r'\\(?:begin|end)\{([^}]+)\}', source):
            if converter.latex_context.get_environment_spec(environment) is None:
                return source
        try:
            return converter.latex_to_text(source, latex_context=_math_parser_context()).strip()
        except (ValueError, IndexError):
            return source

    return ''.join(part if i % 2 else _MATH.sub(convert, part)
                   for i, part in enumerate(_CODE.split(text)))


def searchable_text(text: str) -> str:
    """Use visible table cells in positioned text layers, not HTML syntax.

    Markdown output retains the recognizer's original markup. Only explicit
    HTML tables are decoded here; literal code such as `#include <stdio.h>`
    and mathematical comparisons must survive unchanged.
    """
    if text.lstrip().startswith("```"):
        return text
    if not re.search(r"<table(?:\s|>)", text, re.IGNORECASE):
        return _visible_math(text)

    class TableText(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

        def handle_starttag(self, tag, attrs):
            if tag == "br":
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                self.parts.append(" ")

        def handle_endtag(self, tag):
            if tag in {"tr", "table"}:
                self.parts.append("\n")

    parser = TableText()
    # Formatting whitespace between tags is not a visual table row.
    parser.feed(re.sub(r">\s+<", "><", text))
    return _visible_math("\n".join(" ".join(row.split()) for row in "".join(parser.parts).splitlines()
                                  if row.strip()))


def table_cell_segments(bbox, text):
    """Place simple HTML table cells in their row and column intervals.

    Native table markup has no pixel coordinates per cell. These intervals
    provide initial placement; regional OCR can anchor them to detected cells.
    Complex row spans retain the enclosing table instead of guessing a grid.
    """
    if not text.lstrip().lower().startswith('<table'):
        return []

    class Cells(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.rows, self.row, self.cell = [], [], None
            self.span, self.invalid, self.depth = 1, False, 0

        def handle_starttag(self, tag, attrs):
            if tag == 'table':
                self.depth += 1
                self.invalid |= self.depth > 1
            if tag == 'tr':
                self.row = []
            if tag in {'td', 'th'}:
                attrs = dict(attrs)
                try:
                    self.span = int(attrs.get('colspan', 1))
                    self.invalid |= not 1 <= self.span <= 64 or int(attrs.get('rowspan', 1)) != 1
                except (TypeError, ValueError):
                    self.invalid = True
                self.cell = []
            if tag == 'br' and self.cell is not None:
                self.cell.append('\n')

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

        def handle_endtag(self, tag):
            if tag in {'td', 'th'} and self.cell is not None:
                self.row.append((''.join(self.cell).strip(), self.span))
                self.cell = None
            if tag == 'tr' and self.row:
                self.rows.append(self.row)
                self.row = []
            if tag == 'table':
                self.depth -= 1

    parser = Cells()
    parser.feed(text)
    if parser.invalid or not parser.rows or parser.cell is not None:
        return []
    columns = max(sum(span for _, span in row) for row in parser.rows)
    x0, y0, x1, y1 = bbox
    result = []
    for row_index, row in enumerate(parser.rows):
        column = 0
        for value, span in row:
            if value:
                result.append(([x0 + (x1-x0)*column/columns,
                                y0 + (y1-y0)*row_index/len(parser.rows),
                                x0 + (x1-x0)*(column+span-.08)/columns,
                                y0 + (y1-y0)*(row_index+1)/len(parser.rows)], value))
            column += span
    return result


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
