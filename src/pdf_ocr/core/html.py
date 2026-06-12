"""
HTMLHandler - emit OCR results as an HTML document with invisible-text overlay.

Each page is shown as a background image with OCR text overlaid as
invisible, absolutely-positioned ``<span>``s. Browsers select /
Ctrl+F-search / copy the text exactly as if it were a real text layer,
while still showing the original page image untouched.

Co-authored-by: Milan Hauth <milahu@milahu.duckdns.org>  (PR #8 prototype)
"""

from __future__ import annotations

import base64
import html as _html
import io
import os
import urllib.parse
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image, ImageSequence

from pdf_ocr.core._layout import is_full_page_fallback, split_multi_line_bbox
from pdf_ocr.core.geometry import (
    ANGLE_FLATTEN_DEG,
    quad_affine_components,
    quad_edge_lengths,
    quad_perspective_residual_px,
)
from pdf_ocr.core.pdf import _is_image_path

# Image types every modern browser renders natively. Other extensions
# (TIFF, BMP) get rasterized to sidecar JPEGs even in external mode.
_BROWSER_NATIVE_IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif",
})

MODE_LETTER_SPACING = "letter-spacing"
MODE_FULL_HEIGHT = "full-height"
MODE_SCALED = "scaled"
_VALID_MODES = frozenset({MODE_LETTER_SPACING, MODE_FULL_HEIGHT, MODE_SCALED})
DEFAULT_MODE = MODE_SCALED

# Browser monospace stacks (Courier-family) sit near 0.6 width:height.
_MONOSPACE_FONT_ASPECT = 0.6

# A quad whose bottom-right corner sits further than this (native page
# px) from the parallelogram completion of the other three corners is
# perspective-distorted: the pure-CSS affine matrix can match at most
# three of its corners, so the span also carries `data-quad` and the
# matrix3d script refines it. Surya min-area rects are parallelograms
# (residual ≈ float noise); photo-path quads mapped back through the
# inverse homography routinely exceed this.
_PERSPECTIVE_RESIDUAL_PX = 1.5

# Axis-aligned boxes at least this many times taller than wide (pixel
# terms), holding 2+ characters, render as a vertical run: a horizontal
# nowrap line would overflow the box's right edge by the same factor.
_VERTICAL_ASPECT = 2.0

# Scripts whose glyphs stay upright in vertical runs (writing-mode alone
# suffices; everything else also gets sideways orientation).
_CJK_RANGES = (
    (0x3040, 0x30FF),   # hiragana + katakana
    (0x3400, 0x4DBF),   # CJK extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xAC00, 0xD7AF),   # hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xFF00, 0xFF60),   # full-width forms
)

_JPEG_QUALITY = 80

# `.page` is sized in CSS pixels via `--page-w` so browser zoom
# (Ctrl++ / Ctrl+-) scales the whole page uniformly. An inline script
# at the end of the body shrinks any page wider than the viewport
# once at load, so the default view fits without horizontal scroll
# while subsequent zooming still grows or shrinks the page.
# `container-type: inline-size` plus `%` / `cqw` units on the spans
# keeps overlay selection extents locked to the rasterized image at
# every zoom level.
_PAGE_CSS = """\
body { margin: 0; background: #f5f5f5; }
div.page {
  position: relative;
  width: calc(var(--page-w) * 1px);
  aspect-ratio: var(--page-w) / var(--page-h);
  container-type: inline-size;
  background-repeat: no-repeat;
  background-size: 100% 100%;
  margin: 0 auto 1em auto;
  outline: solid 1px #d0d0d0;
}
span.line {
  position: absolute;
  color: transparent;
  white-space: nowrap;
  font-family: monospace;
  line-height: 1;
  transform: matrix(var(--quad, 1, 0, 0, 1, 0, 0)) scaleX(var(--scale-x, 1));
  transform-origin: 0 0;
}
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; }
  div.page {
    outline-color: #404040;
  }
}
"""

# Opt-in: invert page images in dark mode so scanned white-background
# documents appear dark. hue-rotate(180deg) re-maps colours so dark-blue
# becomes light-blue instead of yellow.
_DARK_INVERT_CSS = """\
@media (prefers-color-scheme: dark) {
  div.page {
    filter: invert() hue-rotate(180deg);
  }
}
"""

# Opt-in: reveal the otherwise-invisible text layer on hover/focus so a
# reader can inspect which text is bound to which region. The dark
# backdrop keeps the reveal readable on any page background — under the
# dark-mode inversion filter it flips to dark-on-light but keeps its
# contrast.
_HOVER_CSS = """\
span.line:hover, span.line:focus {
  color: #fff;
  background: rgba(0, 0, 0, 0.7);
}
"""

# Inline fit-to-viewport script. Runs once at page load: any page
# whose native `--page-w` exceeds the body's usable width is shrunk
# to fit. The reference is `document.body.clientWidth` rather than
# `window.innerWidth` — innerWidth includes the vertical scrollbar,
# so pages sized to it overflow the usable width and trigger a
# horizontal scrollbar. The extra 1px subtracted absorbs fractional
# CSS-pixel rounding when the OS display zoom makes
# `devicePixelRatio` non-integral. Subsequent browser zoom multiplies
# the rewritten CSS width as expected: zooming in produces horizontal
# scroll, zooming out shrinks the page below the viewport.
_FIT_SCRIPT = """\
<script>
(function(){
  var px = document.body.clientWidth - 1;
  for (const p of document.querySelectorAll('div.page')) {
    var w = parseFloat(p.style.getPropertyValue('--page-w'));
    if (!isFinite(w) || w <= 0 || w <= px) continue;
    p.style.width = px + 'px';
  }
})();
</script>
"""

# Measured width congruence (the PDF.js textLayer approach): each span's
# text is measured in the font it actually renders with, and a scaleX
# factor stretches its glyph run to exactly the detected box width.
# `offsetWidth` is the pre-transform layout width, so the rotation in
# the same transform doesn't feed back into the measurement, and re-runs
# are idempotent. Without JavaScript the spans keep their server-side
# fit, which under-fills rather than overflows. Emitted only in `scaled`
# mode: `letter-spacing` mode already stretches via spacing, and
# `full-height` mode keeps the natural glyph width by design.
_MEASURE_SCRIPT = """\
<script>
(function(){
  var ctx = document.createElement('canvas').getContext('2d');
  for (const s of document.querySelectorAll('span.line')) {
    var cs = getComputedStyle(s);
    if ((cs.writingMode || '').indexOf('vertical') === 0) continue;
    ctx.font = cs.fontSize + ' ' + cs.fontFamily;
    var m = ctx.measureText(s.textContent);
    if (!(m.width > 0) || !(s.offsetWidth > 0)) continue;
    s.style.setProperty('--scale-x', s.offsetWidth / m.width);
  }
})();
</script>
"""

# Perspective refinement for keystone-distorted quads: the stylesheet's
# affine matrix can match at most three corners of a non-parallelogram
# quad, so spans flagged with `data-quad` get an exact CSS matrix3d()
# homography here instead. A homography's perspective terms are ratios
# of *rendered* pixel lengths — unlike the affine components they are
# not scale-invariant, so they can't be precomputed server-side. Pages
# keep a fixed CSS-pixel width once the fit script has run (browser
# zoom scales the CSS px unit itself, preserving the mapping), so one
# pass after it suffices. Runs after the measure script so the
# composed scaleX is final; without JavaScript the affine
# approximation from the stylesheet still applies.
_QUAD3D_SCRIPT = """\
<script>
(function(){
  for (const s of document.querySelectorAll('span.line[data-quad]')) {
    var page = s.closest('div.page');
    if (!page) continue;
    var pw = page.clientWidth, ph = page.clientHeight;
    var q = s.dataset.quad.split(' ').map(Number);
    var w = s.offsetWidth, h = s.offsetHeight;
    if (q.length !== 8 || !(pw > 0) || !(ph > 0) || !(w > 0) || !(h > 0)) continue;
    var ox = q[0]*pw, oy = q[1]*ph;
    var x1 = q[2]*pw - ox, y1 = q[3]*ph - oy;
    var x2 = q[4]*pw - ox, y2 = q[5]*ph - oy;
    var x3 = q[6]*pw - ox, y3 = q[7]*ph - oy;
    var dx1 = x1 - x2, dy1 = y1 - y2;
    var dx2 = x3 - x2, dy2 = y3 - y2;
    var rx = x2 - x1 - x3, ry = y2 - y1 - y3;
    var den = dx1*dy2 - dx2*dy1;
    if (!den) continue;
    var g = (rx*dy2 - dx2*ry) / den;
    var k = (dx1*ry - rx*dy1) / den;
    var m = [
      (1+g)*x1/w, (1+g)*y1/w, 0, g/w,
      (1+k)*x3/h, (1+k)*y3/h, 0, k/h,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    var sx = (getComputedStyle(s).getPropertyValue('--scale-x') || '').trim() || '1';
    s.style.transform = 'matrix3d(' + m.join(',') + ') scaleX(' + sx + ')';
  }
})();
</script>
"""


class HTMLHandler:
    """Output writer mirroring ``PDFHandler.embed_structured_text``.

    The HTML overlay references the page image as either an external
    file (default — keeps the HTML small and reuses the input file when
    it is browser-renderable) or an inline base64 ``data:`` URL
    (``inline_images=True`` — produces a single self-contained file at
    the cost of ~35% size inflation).

    External-mode dispatch:

    * Browser-renderable single-frame inputs (JPEG/PNG/WebP/AVIF/GIF):
      the input file is referenced directly via a relative URL from the
      output HTML. No new files are written.
    * Everything else (PDF, multi-frame TIFF, BMP, multi-frame WebP/GIF):
      each page is rasterized to a sidecar JPEG named
      ``<output_stem>_p<N>.jpg`` (1-indexed, zero-padded to the
      page-count width so listings sort in page order) next to the
      output HTML and referenced via a relative URL. Single-page inputs
      get plain ``<output_stem>.jpg`` with no page suffix.
    """

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        inline_images: bool = False,
        invert_dark: bool = False,
        hover_text: bool = False,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"unknown HTML mode {mode!r}; "
                f"expected one of {sorted(_VALID_MODES)}"
            )
        self.mode = mode
        self.inline_images = inline_images
        self.invert_dark = invert_dark
        self.hover_text = hover_text

    def embed_structured_text(
        self,
        input_path: str,
        output_path: str,
        pages_data: dict,
        dpi: int = 200,
    ) -> None:
        """Render the OCR'd `pages_data` as HTML next to `input_path` images.

        Parameters mirror ``PDFHandler.embed_structured_text`` so this
        class can drop into ``OCRPipeline(output_writer=...)``.
        """
        title = Path(input_path).stem or "OCR output"
        pages = list(self._iter_page_renderings(input_path, output_path, dpi))
        html = self._render_html(title, pages, pages_data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    # --- page-image sourcing ---------------------------------------------

    def _iter_page_renderings(
        self, input_path: str, output_path: str, dpi: int,
    ) -> Iterator[tuple[int, str, float, float]]:
        """Yield (page_index, image_url, width_px, height_px) per page.

        ``image_url`` is suitable for direct embedding in a CSS
        ``background-image: url(...)`` value: either a ``data:`` URL
        (inline mode) or a relative URL to a real file (external mode).
        """
        if self.inline_images:
            for page_idx, img_bytes, w, h in self._rasterize_pages(input_path, dpi):
                yield page_idx, _data_url_jpeg(img_bytes), w, h
            return

        out_parent = Path(output_path).resolve().parent
        out_stem = Path(output_path).stem
        direct = self._direct_reference_url(input_path, out_parent)
        if direct is not None:
            url, width, height = direct
            yield 0, url, float(width), float(height)
            return

        rasterized = list(self._rasterize_pages(input_path, dpi))
        # Page numbers are zero-padded to the page-count width
        # (`_p01`..`_p12` for a 12-page input) so lexicographic file
        # listings sort in page order.
        pad = len(str(len(rasterized)))
        for page_idx, img_bytes, w, h in rasterized:
            if len(rasterized) == 1:
                sidecar_name = f"{out_stem}.jpg"
            else:
                sidecar_name = f"{out_stem}_p{page_idx + 1:0{pad}d}.jpg"
            (out_parent / sidecar_name).write_bytes(img_bytes)
            yield page_idx, _url_quote_segment(sidecar_name), w, h

    @staticmethod
    def _direct_reference_url(
        input_path: str, output_parent: Path,
    ) -> tuple[str, int, int] | None:
        """Return (relative_url, width, height) if ``input_path`` is a
        single-frame browser-native image, else None."""
        suffix = Path(input_path).suffix.lower()
        if suffix not in _BROWSER_NATIVE_IMAGE_EXTS:
            return None
        try:
            with Image.open(input_path) as src:
                if getattr(src, "n_frames", 1) != 1:
                    return None
                width, height = src.size
        except (OSError, ValueError):
            return None
        rel = _relative_url(Path(input_path).resolve(), output_parent)
        return rel, width, height

    def _rasterize_pages(
        self, input_path: str, dpi: int,
    ) -> Iterator[tuple[int, bytes, float, float]]:
        """Yield (page_index, jpeg_bytes, width_px, height_px) for every
        page of the input, regardless of source type."""
        if _is_image_path(input_path):
            yield from self._pages_from_image(input_path)
        else:
            yield from self._pages_from_pdf(input_path, dpi)

    @staticmethod
    def _pages_from_pdf(
        input_path: str, dpi: int,
    ) -> Iterator[tuple[int, bytes, float, float]]:
        """Rasterize each PDF page at `dpi` and emit JPEG bytes + dims."""
        doc = fitz.open(input_path)
        try:
            for page_idx, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("jpg", jpg_quality=_JPEG_QUALITY)
                # Pixel dimensions so absolute-positioned spans align
                # with the rasterized background image.
                yield page_idx, img_bytes, float(pix.width), float(pix.height)
        finally:
            doc.close()

    @staticmethod
    def _pages_from_image(
        input_path: str,
    ) -> Iterator[tuple[int, bytes, float, float]]:
        """Iterate frames of a single- or multi-frame image (TIFF, etc.)."""
        with Image.open(input_path) as src:
            for page_idx, frame in enumerate(ImageSequence.Iterator(src)):
                img = frame.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
                yield page_idx, buf.getvalue(), float(img.width), float(img.height)

    # --- HTML rendering ---------------------------------------------------

    def _render_html(self, title: str, pages, pages_data: dict) -> str:
        out = io.StringIO()
        out.write("<!doctype html>\n<html>\n<head>\n")
        out.write('<meta charset="utf-8" />\n')
        out.write(f"<title>{_html.escape(title)}</title>\n")
        out.write("<style>\n")
        out.write(_PAGE_CSS)
        if self.hover_text:
            out.write(_HOVER_CSS)
        if self.invert_dark:
            out.write(_DARK_INVERT_CSS)
        out.write("</style>\n</head>\n<body>\n")
        # Span emission records which optional machinery the document
        # actually needs, so pages without (e.g.) perspective quads
        # don't carry the script that serves them.
        features: set[str] = set()
        for page_idx, image_url, width, height in pages:
            self._render_page(
                out, page_idx, image_url, width, height,
                pages_data.get(page_idx, []), features,
            )
        out.write(_FIT_SCRIPT)
        if self.mode == MODE_SCALED:
            # Must run after the fit script: fitting rewrites page widths,
            # and the measurement reads post-fit span layout widths.
            out.write(_MEASURE_SCRIPT)
        if "perspective" in features:
            # Must run after the measure script: it bakes the final
            # --scale-x into the composed matrix3d transform.
            out.write(_QUAD3D_SCRIPT)
        out.write("</body>\n</html>\n")
        return out.getvalue()

    def _render_page(
        self, out, page_idx, image_url, width, height, items, features,
    ):
        out.write(
            f'<div class="page" data-page="{page_idx + 1}" '
            f'style="--page-w:{_num(width)};--page-h:{_num(height)};'
            f"background-image:url('{image_url}')\">\n"
        )
        for rect_coords, text in items:
            # Passed through unchanged — converting to a plain list here
            # would strip OrientedBox's quad/angle and silently disable
            # rotated rendering.
            self._render_box(
                out, rect_coords, text or "", width, height, features,
            )
        out.write("</div>\n")

    def _render_box(self, out, rect_coords, text, page_width, page_height, features):
        text = text.strip()
        if not text:
            return

        # The aligner's [0,0,1,1] full-page fallback bbox: render in a
        # small inset rect, with each '\n'-separated line on its own row.
        if is_full_page_fallback(rect_coords, text):
            inset = [
                10.0 / page_width,
                10.0 / page_height,
                (page_width - 10.0) / page_width,
                (page_height - 10.0) / page_height,
            ]
            for sub_rect, line in split_multi_line_bbox(inset, text):
                self._emit_span(out, sub_rect, line, page_width, page_height, features)
            return

        # Real bbox with multi-line content (grounded VLM joined lines):
        # split vertically, emit one span per line.
        if "\n" in text:
            sub = split_multi_line_bbox(rect_coords, text)
            if len(sub) > 1:
                for sub_rect, line in sub:
                    self._emit_span(out, sub_rect, line, page_width, page_height, features)
                return
            if sub:
                text = sub[0][1]

        self._emit_span(out, rect_coords, text, page_width, page_height, features)

    def _emit_span(self, out, rect_coords, text, page_width, page_height, features):
        # Tilted detector quads anchor the span at the quad's top-left
        # corner with the quad's own edge lengths; the stylesheet's
        # matrix(var(--quad)) — whose columns are the quad's edge unit
        # vectors — then rotates/shears it into place exactly. The
        # affine components are dimensionless, so this stays correct at
        # any page scale with no script. Axis-aligned boxes keep the
        # envelope geometry and the variable's identity default.
        # Near-horizontal tilt is flattened — min-area rects wobble a
        # degree or two on straight text, and rotating the overlay by
        # that noise hurts selection congruence.
        angle = getattr(rect_coords, "angle", 0.0)
        quad = getattr(rect_coords, "quad", None)
        quad_css = ""
        quad_attr = ""
        if quad is not None and abs(angle) >= ANGLE_FLATTEN_DEG:
            run_px, height_px = quad_edge_lengths(quad, page_width, page_height)
            left, top = quad[0], quad[1]
            w_norm = run_px / page_width
            h_norm = height_px / page_height
            a, b, c, d = quad_affine_components(quad, page_width, page_height)
            quad_css = f"--quad:{_num(a)},{_num(b)},{_num(c)},{_num(d)};"
            # A keystone-distorted quad (photo path) can't be matched on
            # all four corners by any affine: ship the corners and let
            # the matrix3d script compute the exact homography.
            residual = quad_perspective_residual_px(quad, page_width, page_height)
            if residual > _PERSPECTIVE_RESIDUAL_PX:
                quad_attr = ' data-quad="' + " ".join(_num(v) for v in quad) + '"'
                features.add("perspective")
        else:
            nx0, ny0, nx1, ny1 = rect_coords
            left, top = nx0, ny0
            w_norm = nx1 - nx0
            h_norm = ny1 - ny0
        if w_norm <= 0 or h_norm <= 0 or not text:
            return

        # Tall narrow axis-aligned boxes hold vertical runs (CJK columns,
        # sideways labels): flow the text down the box instead of
        # overflowing its right edge. Honored quads never take this
        # branch — a sideways line arrives as a ±90° quad and the matrix
        # path renders it exactly.
        vertical = False
        if not quad_css and self.mode == MODE_SCALED:
            w_px = w_norm * page_width
            h_px = h_norm * page_height
            vertical = (
                len(text) >= 2 and w_px > 0
                and h_px >= _VERTICAL_ASPECT * w_px
            )
        vertical_css = ""
        cjk = False
        if vertical:
            cjk = _is_mostly_cjk(text)
            vertical_css = "writing-mode:vertical-rl;"
            if not cjk:
                vertical_css += "text-orientation:sideways;"

        conf = getattr(rect_coords, "confidence", None)
        conf_attr = (
            f' data-conf="{float(conf):.3f}"' if conf is not None else ""
        )

        sizing = self._span_sizing_style(
            text, w_norm, h_norm, page_width, page_height,
            vertical=vertical, cjk=cjk,
        )
        # Escape the minimum HTML special chars so the text is parsed
        # correctly even though it's rendered transparent.
        safe_text = _html.escape(text, quote=False)
        out.write(
            f'<span class="line"{conf_attr}{quad_attr} '
            f'style="left:{_num(left * 100)}%;'
            f'top:{_num(top * 100)}%;'
            f'width:{_num(w_norm * 100)}%;{quad_css}{vertical_css}{sizing}">'
            f'{safe_text}</span>\n'
        )

    def _span_sizing_style(
        self, text: str, w_norm: float, h_norm: float,
        page_width: float, page_height: float,
        vertical: bool = False, cjk: bool = False,
    ) -> str:
        """Choose font-size + letter-spacing for one span per `self.mode`.

        Positions are emitted in `%` and font sizes in `cqw` (1cqw = 1% of
        the page container's width) so that the entire overlay scales
        with the page when the viewport is narrower than the rasterized
        image. The math is the same as the pixel version — we just
        express every length in container-relative units before writing.
        """
        n_chars = max(1, len(text))
        # bbox dimensions expressed in cqw (1cqw = 1% of page width).
        w_cqw = w_norm * 100
        h_cqw = h_norm * (page_height / page_width) * 100
        h_pct = h_norm * 100

        if vertical:
            # Glyphs stack down the box: the per-glyph advance along the
            # column is ~1 em for upright CJK and ~one monospace advance
            # for sideways Latin; the cross-axis glyph size is the font
            # size itself, capped by the box width. The measure script
            # skips vertical spans (measureText is horizontal-advance
            # only), so the server-side fit is final.
            per_char = h_cqw / n_chars
            if cjk:
                font_size_cqw = min(w_cqw, per_char)
            else:
                font_size_cqw = min(w_cqw, per_char / _MONOSPACE_FONT_ASPECT)
            return f"font-size:{_num(font_size_cqw)}cqw;height:{_num(h_pct)}%;"

        if self.mode == MODE_LETTER_SPACING:
            # Font sized to bbox height; spread characters to fill width
            # via letter-spacing. Negative spacing is allowed — characters
            # overlap visually but selection extents still span the box.
            font_size_cqw = h_cqw
            natural_width_cqw = n_chars * font_size_cqw * _MONOSPACE_FONT_ASPECT
            letter_spacing_cqw = (w_cqw - natural_width_cqw) / n_chars
            return (
                f"font-size:{_num(font_size_cqw)}cqw;"
                f"letter-spacing:{_num(letter_spacing_cqw)}cqw;"
                f"height:{_num(h_pct)}%;"
            )

        if self.mode == MODE_FULL_HEIGHT:
            # Font sized to bbox height regardless of resulting width.
            return f"font-size:{_num(h_cqw)}cqw;height:{_num(h_pct)}%;"

        # MODE_SCALED: pick the smaller of width-fit and height-fit so
        # neither dimension overflows.
        char_width_for_height = h_cqw * _MONOSPACE_FONT_ASPECT
        char_width_for_width = w_cqw / n_chars
        char_width = min(char_width_for_height, char_width_for_width)
        font_size_cqw = char_width / _MONOSPACE_FONT_ASPECT
        return f"font-size:{_num(font_size_cqw)}cqw;height:{_num(h_pct)}%;"


def _is_mostly_cjk(text: str) -> bool:
    """True when at least half the non-space glyphs are CJK — those stay
    upright in a vertical run, everything else reads better sideways."""
    glyphs = [ch for ch in text if not ch.isspace()]
    if not glyphs:
        return False
    cjk = sum(
        1 for ch in glyphs
        if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES)
    )
    return cjk * 2 >= len(glyphs)


def _num(n: float) -> str:
    """Render `n` as an int if it has no fractional part, else 4 decimals."""
    if isinstance(n, int):
        return str(n)
    if n == int(n):
        return str(int(n))
    return f"{n:.4f}".rstrip("0").rstrip(".")


def _url_quote_segment(name: str) -> str:
    """Percent-encode a single filename segment for safe use in a URL."""
    return urllib.parse.quote(name, safe="")


def _relative_url(target: Path, base_dir: Path) -> str:
    """Return a URL-encoded relative path from `base_dir` to `target`.

    Falls back to an absolute file:// URL when the paths are on
    different drives (Windows) so the HTML still loads.
    """
    try:
        rel = os.path.relpath(target, start=base_dir)
    except ValueError:
        return target.as_uri()
    parts = rel.replace(os.sep, "/").split("/")
    encoded = [
        p if p in (".", "..") else urllib.parse.quote(p, safe="")
        for p in parts
    ]
    return "/".join(encoded)


def _data_url_jpeg(img_bytes: bytes) -> str:
    """Build a `data:image/jpeg;base64,...` URL from raw JPEG bytes."""
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
