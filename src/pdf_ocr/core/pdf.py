"""
PDFHandler - PDF processing utilities.

Handles PDF to image conversion and text embedding for creating searchable
PDFs. Also accepts raw image inputs (JPEG/PNG/TIFF/BMP/WebP/AVIF) —
including multi-page TIFF — so single-scan-per-file workflows don't need
a PDF wrap step first. AVIF support is provided natively by Pillow ≥
11.3 (the `pyproject.toml` constraint enforces that floor).
"""

import base64
import io
from pathlib import Path
import os

import fitz  # PyMuPDF
from PIL import Image, ImageSequence


IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".avif",
})


def _is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


class PDFHandler:
    """
    PDF processing handler for OCR workflows.

    Handles:
    - Converting PDF pages to base64 PNG images
    - Embedding an invisible text layer to produce a "sandwich" PDF
      (image background + selectable/searchable text overlay)
    """

    def convert_to_images(
        self,
        pdf_path,
        dpi: int = 150,
        max_image_dim: int = 1024,
    ) -> dict[int, str]:
        """
        Render every page to a base64-encoded JPEG, capped at `max_image_dim`
        pixels on the longest edge so the image fits the VLM's context window.

        Accepts either a PDF or a raw image file
        (JPEG/PNG/TIFF/BMP/WebP/AVIF). Multi-page TIFFs are expanded to one
        page per frame. For images the `dpi` argument is ignored — the file
        is used at its native resolution, capped by `max_image_dim`.

        Smaller caps are required by some local VLMs:
          - OlmOCR-2 (Qwen2.5-VL base): 1024 is fine (default)
          - GLM-OCR:1.1B (Ollama): ~640 — larger images crash the runner
          - Florence-2 / MinerU: see their docs

        Returns a dict of {page_num: base64_str}.
        """
        if _is_image_path(pdf_path):
            return self._images_from_image_file(pdf_path, max_image_dim)

        images: dict[int, str] = {}
        doc = fitz.open(pdf_path)
        try:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("jpg", jpg_quality=50)))
                img.thumbnail((max_image_dim, max_image_dim))

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=50)
                images[page_num] = base64.b64encode(buffer.getvalue()).decode("utf-8")
        finally:
            doc.close()
        return images

    @staticmethod
    def _images_from_image_file(path, max_image_dim: int) -> dict[int, str]:
        """Load a JPEG/PNG/TIFF/BMP; multi-frame TIFFs become multiple pages."""
        images: dict[int, str] = {}
        with Image.open(path) as src:
            for page_num, frame in enumerate(ImageSequence.Iterator(src)):
                img = frame.convert("RGB").copy()
                img.thumbnail((max_image_dim, max_image_dim))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                images[page_num] = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return images

    def embed_structured_text(
        self,
        input_pdf_path: str,
        output_pdf_path: str,
        pages_data: dict[int, list[tuple[list[float], str]]],
        dpi: int = 200,
    ) -> None:
        """
        Build a searchable "sandwich" PDF: rasterize each page as a background
        image and overlay invisible text positioned to match the source layout.

        Accepts either a PDF or a raw image (JPEG/PNG/TIFF/BMP/WebP/AVIF)
        as input. Image inputs are converted to a 1-page-per-frame PDF —
        no rasterization-to-PDF-to-rasterization round trip required.

        Args:
            input_pdf_path: Path to the source PDF or image file.
            output_pdf_path: Where to write the searchable PDF.
            pages_data: {page_num: [([nx0, ny0, nx1, ny1], text), ...]} with
                normalized (0..1) box coordinates.
            dpi: Rasterization DPI for PDF-sourced backgrounds (ignored for
                image inputs — they're used at native resolution).
        """
        output_ext = os.path.splitext(output_pdf_path)[1].lower()
        print(f"output_ext: {output_ext!r}")
        if output_ext == ".html":
            return self._embed_structured_text_html(
                input_pdf_path,
                output_pdf_path,
                pages_data,
                dpi,
            )
        if _is_image_path(input_pdf_path):
            self._embed_from_image_input(input_pdf_path, output_pdf_path, pages_data)
            return

        doc = fitz.open(input_pdf_path)
        new_doc = fitz.open()

        try:
            for page_num in range(len(doc)):
                old_page = doc[page_num]
                width = old_page.rect.width
                height = old_page.rect.height

                pix = old_page.get_pixmap(dpi=dpi)
                img_data = pix.tobytes("jpg", jpg_quality=80)

                new_page = new_doc.new_page(width=width, height=height)
                new_page.insert_image(new_page.rect, stream=img_data)

                for rect_coords, text in pages_data.get(page_num, []):
                    self._draw_invisible_text(new_page, rect_coords, text, width, height)

            new_doc.save(output_pdf_path)
        finally:
            new_doc.close()
            doc.close()

    def _embed_from_image_input(
        self,
        image_path: str,
        output_pdf_path: str,
        pages_data: dict,
    ) -> None:
        """Build a sandwich PDF directly from an image (single- or multi-frame)."""
        new_doc = fitz.open()
        try:
            with Image.open(image_path) as src:
                for page_num, frame in enumerate(ImageSequence.Iterator(src)):
                    img = frame.convert("RGB")
                    # One PDF point = 1/72 inch. Assume image is 72 DPI so
                    # pixel count equals page size in points. Concrete value
                    # doesn't matter — all coords are normalized.
                    width, height = float(img.width), float(img.height)

                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_data = buf.getvalue()

                    new_page = new_doc.new_page(width=width, height=height)
                    new_page.insert_image(new_page.rect, stream=img_data)

                    for rect_coords, text in pages_data.get(page_num, []):
                        self._draw_invisible_text(
                            new_page, rect_coords, text, width, height
                        )
            new_doc.save(output_pdf_path)
        finally:
            new_doc.close()

    @staticmethod
    def _draw_invisible_text(page, rect_coords, text, page_width, page_height):
        """
        Embed invisible `text` so its glyph bboxes span the *full width* of
        the source bbox — selecting anywhere inside the bbox in a PDF viewer
        returns the text.

        Strategy: size the font by box height (glyphs never exceed the box
        vertically → no bleeding into neighbouring rows), then apply a
        horizontal-scale matrix via the `morph` parameter so rendered glyph
        bboxes span the box width. `render_mode=3` keeps the layer invisible;
        the geometric distortion only affects selection/search extents, not
        Unicode codepoints (copy, Ctrl+F, accessibility tools still return
        the original text verbatim).

        Why not `min(width_based, height_based)` fontsize like before?
        Whichever constraint is tighter wins, and when height is tighter
        (short wide boxes: headings, form labels, fields) the text ends
        partway across the box — selection on the right side returns
        nothing. Sizing to fill width instead causes vertical overflow
        that bleeds into neighbouring rows. Horizontal scaling decouples
        the two axes: height fits, width fills, neither constraint is
        violated.
        """
        text = (text or "").strip()
        if not text:
            return  # empty box — nothing to embed

        nx0, ny0, nx1, ny1 = rect_coords

        # Detect the aligner's full-page fallback by bbox (covers the
        # whole normalized page, [0,0,1,1]) rather than by the presence
        # of "\n" in text. A grounded VLM that emits multi-line content
        # for a real bbox must NOT be redirected to the full-page
        # fallback rect — that would shift the text to the page top and
        # clobber other bboxes' search positions, surfacing as
        # "following lines moved up" in the rendered output.
        is_full_page_fallback = (
            nx0 <= 0.001 and ny0 <= 0.001
            and nx1 >= 0.999 and ny1 >= 0.999
            and "\n" in text
        )
        if is_full_page_fallback:
            fallback_rect = fitz.Rect(10, 10, page_width - 10, page_height - 10)
            page.insert_textbox(
                fallback_rect, text,
                fontsize=6, fontname="helv",
                render_mode=3, color=(0, 0, 0), align=0,
            )
            return

        # A real bbox with multi-line content: split by line and recurse
        # to place each line at its own vertical sub-slice of the bbox.
        # Handles grounded-VLM outputs that joined visual lines into one
        # element with an embedded "\n" — search/selection still land at
        # the right y position instead of being shifted off-page.
        if "\n" in text:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if len(lines) > 1:
                slice_h = (ny1 - ny0) / len(lines)
                for i, line in enumerate(lines):
                    PDFHandler._draw_invisible_text(
                        page,
                        [nx0, ny0 + i * slice_h, nx1, ny0 + (i + 1) * slice_h],
                        line, page_width, page_height,
                    )
                return
            text = lines[0] if lines else text  # only one non-empty line

        pdf_rect = fitz.Rect(
            nx0 * page_width,
            ny0 * page_height,
            nx1 * page_width,
            ny1 * page_height,
        )

        box_width = pdf_rect.width
        box_height = pdf_rect.height
        if box_width <= 0 or box_height <= 0:
            return

        font = fitz.Font("helv")

        # Size so the full glyph extent (ascender - descender in em-units)
        # fits exactly inside the box height. For Helvetica this works out
        # to ~box_height/1.374 ≈ box_height * 0.728 — tighter than the old
        # 0.85 constant, and eliminates the ascender overshoot we used to
        # tolerate at the top edge.
        ascender = getattr(font, "ascender", 1.075)  # Helvetica fallback
        descender = getattr(font, "descender", -0.299)
        extent_em = max(0.01, ascender - descender)  # descender is negative
        fontsize = max(3.0, min(72.0, box_height / extent_em))

        # Multi-line bbox detection: Surya occasionally groups visually
        # adjacent handwritten lines into one detection (e.g. "schwache
        # Grenzen / im Kopf"). The DP then matches the LLM's joined
        # string to that one tall bbox, and the embed would render the
        # whole phrase at one y (the bottom of the bbox), leaving the
        # upper visual line empty in the searchable text layer.
        #
        # Heuristic: distinguish a multi-line region from a single tall-
        # but-still-one-line bbox. Aspect alone confuses handwritten
        # "Typen 23" (one line, aspect ≈ 0.28 due to padding around the
        # glyphs) with a real two-line region (aspect ≈ 0.27-0.35).
        # Combine signals — the bbox must be tall in the absolute sense
        # (norm height > 7% of page) AND have multi-line aspect ratio.
        # Single visual lines on Letter-size pages take ~4-6% of page
        # height even with generous padding, so the combined gate
        # catches the 2-line case without over-splitting padded single
        # lines.
        words = text.split()
        norm_height = ny1 - ny0
        aspect = box_height / max(0.01, box_width)
        if norm_height > 0.07 and aspect > 0.20 and len(words) >= 2:
            n_lines = 3 if norm_height > 0.13 else 2
            n_lines = min(n_lines, len(words))
            slice_h = (ny1 - ny0) / n_lines
            for i in range(n_lines):
                start = round(i * len(words) / n_lines)
                end = round((i + 1) * len(words) / n_lines)
                line_text = " ".join(words[start:end])
                if not line_text:
                    continue
                PDFHandler._draw_invisible_text(
                    page,
                    [nx0, ny0 + i * slice_h, nx1, ny0 + (i + 1) * slice_h],
                    line_text, page_width, page_height,
                )
            return

        natural_width = font.text_length(text, fontsize=fontsize)
        if natural_width <= 0:
            return  # nothing measurable to draw (e.g. all-whitespace)

        # Horizontal scale so extracted word bbox spans the full box width
        # (minus a hairline margin so we don't butt up against neighbours).
        # scale_x > 1 stretches; scale_x < 1 compresses — both correctly
        # size the word's bounding box, which is what selection uses.
        target_width = max(1.0, box_width * 0.98)
        scale_x = target_width / natural_width

        # Place baseline at box bottom, shifted up by the descender so tails
        # of "g"/"p"/"y" sit inside the box and the glyph tops land on the
        # box top (since ascender*fontsize + |descender|*fontsize = box_height
        # by construction).
        baseline = fitz.Point(pdf_rect.x0, pdf_rect.y1 + descender * fontsize)

        # Morph pivot = baseline. Matrix scales x around that pivot so
        # the text's starting x stays at pdf_rect.x0 and the end x lands
        # at pdf_rect.x0 + target_width.
        morph = (baseline, fitz.Matrix(scale_x, 1.0))
        page.insert_text(
            baseline, text,
            fontsize=fontsize, fontname="helv",
            render_mode=3, color=(0, 0, 0),
            morph=morph,
        )


    def _embed_structured_text_html(
        self,
        input_pdf_path: str,
        output_html_path: str,
        pages_data: dict[int, list[tuple[list[float], str]]],
        dpi: int = 200,
    ) -> None:
        """
        Build an HTML document: rasterize each page as a background
        image and overlay invisible text positioned to match the source layout.

        Accepts either a PDF or a raw image (JPEG/PNG/TIFF/BMP/WebP/AVIF)
        as input. Image inputs are converted to a 1-page-per-frame PDF —
        no rasterization-to-PDF-to-rasterization round trip required.

        Args:
            input_pdf_path: Path to the source PDF or image file.
            output_html_path: Where to write the HTML document.
            pages_data: {page_num: [([nx0, ny0, nx1, ny1], text), ...]} with
                normalized (0..1) box coordinates.
            dpi: Rasterization DPI for PDF-sourced backgrounds (ignored for
                image inputs — they're used at native resolution).
        """
        if _is_image_path(input_pdf_path):
            self._embed_from_image_input_html(input_pdf_path, output_html_path, pages_data)
            return

        raise NotImplementedError("_is_image_path(input_pdf_path) == False")


    def _embed_from_image_input_html(
        self,
        image_path: str,
        output_html_path: str,
        pages_data: dict,
    ) -> None:
        """Build an HTML document directly from an image (single- or multi-frame)."""
        new_doc = open(output_html_path, "w", encoding="utf8")

        new_doc.write('<!doctype html>\n')
        new_doc.write('<html>\n')
        new_doc.write('<head>\n')
        new_doc.write('<style>\n')
        new_doc.write('div.page {\n')
        new_doc.write('  outline: solid 1px gray;\n')
        new_doc.write('}\n')
        new_doc.write('span.line {\n')
        new_doc.write('  position: absolute;\n')
        new_doc.write('  color: transparent;\n')
        new_doc.write('  white-space: nowrap;\n')
        # font aspect: 0.6
        new_doc.write('  font-family: monospace;\n')
        new_doc.write('}\n')
        new_doc.write('</style>\n')
        new_doc.write('</head>\n')
        new_doc.write('<body>\n')

        try:
            with Image.open(image_path) as src:
                if src.n_frames == 0:
                    raise ValueError(f"image has no frames: {image_path!r}")
                # if src.n_frames > 1:
                #     raise NotImplementedError("src.n_frames > 1")
                for page_num, frame in enumerate(ImageSequence.Iterator(src)):
                    # one image, one page
                    new_page = new_doc
                    # width, height = float(frame.width), float(frame.height) # why float?
                    width, height = frame.width, frame.height
                    if src.n_frames == 1:
                        # single-frame image
                        # use path to image file
                        img_url = os.path.relpath(image_path, os.path.dirname(output_html_path))
                    else:
                        # multi-frame image
                        # inline frame as data URL
                        img = frame.convert("RGB") # TODO why not keep grayscale
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        img_url = (
                            "data:base64," +
                            base64.b64encode(buf.getvalue()).decode("ascii")
                        )
                        del img
                    page_style = (
                        f"background-image:url({img_url!r});" +
                        f"background-repeat:no-repeat;" +
                        f"width:{width}px;" +
                        f"height:{height}px;" +
                        ""
                    )
                    del img_url
                    new_doc.write(f'<div class="page" style="{page_style}">\n')
                    del page_style
                    for rect_coords, text in pages_data.get(page_num, []):
                        self._draw_invisible_text_html(
                            new_page, rect_coords, text, width, height
                        )
                    new_doc.write(f'</div>\n')
                return
                # TODO remove
                for page_num, frame in enumerate(ImageSequence.Iterator(src)):
                    img = frame.convert("RGB")
                    # One PDF point = 1/72 inch. Assume image is 72 DPI so
                    # pixel count equals page size in points. Concrete value
                    # doesn't matter — all coords are normalized.
                    width, height = float(img.width), float(img.height)

                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_data = buf.getvalue()

                    new_page = new_doc.new_page(width=width, height=height)
                    new_page.insert_image(new_page.rect, stream=img_data)

                    for rect_coords, text in pages_data.get(page_num, []):
                        self._draw_invisible_text(
                            new_page, rect_coords, text, width, height
                        )
            # new_doc.save(output_pdf_path)
            new_doc.write(f'</body>\n')
            new_doc.write(f'</html>\n')
        finally:
            new_doc.close()


    @staticmethod
    def _draw_invisible_text_html(page, rect_coords, text, page_width, page_height):
        text = (text or "").strip()
        if not text:
            return  # empty box — nothing to embed

        nx0, ny0, nx1, ny1 = rect_coords

        # Detect the aligner's full-page fallback by bbox (covers the
        # whole normalized page, [0,0,1,1]) rather than by the presence
        # of "\n" in text. A grounded VLM that emits multi-line content
        # for a real bbox must NOT be redirected to the full-page
        # fallback rect — that would shift the text to the page top and
        # clobber other bboxes' search positions, surfacing as
        # "following lines moved up" in the rendered output.
        is_full_page_fallback = (
            nx0 <= 0.001 and ny0 <= 0.001
            and nx1 >= 0.999 and ny1 >= 0.999
            and "\n" in text
        )
        if is_full_page_fallback:
            fallback_rect = (
                10,
                10,
                page_width - 10,
                page_height - 10
            )
            _page_insert_textbox_html(
                page,
                fallback_rect,
                text,
                # fontsize=6,
                # fontname="helv",
                # render_mode=3,
                # color=(0, 0, 0),
                # align=0,
            )
            return

        # A real bbox with multi-line content: split by line and recurse
        # to place each line at its own vertical sub-slice of the bbox.
        # Handles grounded-VLM outputs that joined visual lines into one
        # element with an embedded "\n" — search/selection still land at
        # the right y position instead of being shifted off-page.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        slice_h = (ny1 - ny0) / len(lines)
        for i, line in enumerate(lines):
            line_rect = (
                nx0,
                ny0 + i * slice_h,
                nx1,
                ny0 + (i + 1) * slice_h
            )
            line_rect = (
                page_width * (nx0),
                page_height * (ny0 + i * slice_h),
                page_width * (nx1),
                page_height * (ny0 + (i + 1) * slice_h)
            )
            _page_insert_textbox_html(
                page,
                line_rect,
                line,
            )
            # PDFHandler._draw_invisible_text(
            #     page,
            #     line_rect,
            #     line,
            #     page_width,
            #     page_height,
            # )
        return
        # TODO remove
        text = lines[0] if lines else text  # only one non-empty line

        pdf_rect = fitz.Rect(
            nx0 * page_width,
            ny0 * page_height,
            nx1 * page_width,
            ny1 * page_height,
        )

        box_width = pdf_rect.width
        box_height = pdf_rect.height
        if box_width <= 0 or box_height <= 0:
            return

        font = fitz.Font("helv")

        # Size so the full glyph extent (ascender - descender in em-units)
        # fits exactly inside the box height. For Helvetica this works out
        # to ~box_height/1.374 ≈ box_height * 0.728 — tighter than the old
        # 0.85 constant, and eliminates the ascender overshoot we used to
        # tolerate at the top edge.
        ascender = getattr(font, "ascender", 1.075)  # Helvetica fallback
        descender = getattr(font, "descender", -0.299)
        extent_em = max(0.01, ascender - descender)  # descender is negative
        fontsize = max(3.0, min(72.0, box_height / extent_em))

        # Multi-line bbox detection: Surya occasionally groups visually
        # adjacent handwritten lines into one detection (e.g. "schwache
        # Grenzen / im Kopf"). The DP then matches the LLM's joined
        # string to that one tall bbox, and the embed would render the
        # whole phrase at one y (the bottom of the bbox), leaving the
        # upper visual line empty in the searchable text layer.
        #
        # Heuristic: distinguish a multi-line region from a single tall-
        # but-still-one-line bbox. Aspect alone confuses handwritten
        # "Typen 23" (one line, aspect ≈ 0.28 due to padding around the
        # glyphs) with a real two-line region (aspect ≈ 0.27-0.35).
        # Combine signals — the bbox must be tall in the absolute sense
        # (norm height > 7% of page) AND have multi-line aspect ratio.
        # Single visual lines on Letter-size pages take ~4-6% of page
        # height even with generous padding, so the combined gate
        # catches the 2-line case without over-splitting padded single
        # lines.
        words = text.split()
        norm_height = ny1 - ny0
        aspect = box_height / max(0.01, box_width)
        if norm_height > 0.07 and aspect > 0.20 and len(words) >= 2:
            n_lines = 3 if norm_height > 0.13 else 2
            n_lines = min(n_lines, len(words))
            slice_h = (ny1 - ny0) / n_lines
            for i in range(n_lines):
                start = round(i * len(words) / n_lines)
                end = round((i + 1) * len(words) / n_lines)
                line_text = " ".join(words[start:end])
                if not line_text:
                    continue
                PDFHandler._draw_invisible_text(
                    page,
                    [nx0, ny0 + i * slice_h, nx1, ny0 + (i + 1) * slice_h],
                    line_text, page_width, page_height,
                )
            return

        natural_width = font.text_length(text, fontsize=fontsize)
        if natural_width <= 0:
            return  # nothing measurable to draw (e.g. all-whitespace)

        # Horizontal scale so extracted word bbox spans the full box width
        # (minus a hairline margin so we don't butt up against neighbours).
        # scale_x > 1 stretches; scale_x < 1 compresses — both correctly
        # size the word's bounding box, which is what selection uses.
        target_width = max(1.0, box_width * 0.98)
        scale_x = target_width / natural_width

        # Place baseline at box bottom, shifted up by the descender so tails
        # of "g"/"p"/"y" sit inside the box and the glyph tops land on the
        # box top (since ascender*fontsize + |descender|*fontsize = box_height
        # by construction).
        baseline = fitz.Point(pdf_rect.x0, pdf_rect.y1 + descender * fontsize)

        # Morph pivot = baseline. Matrix scales x around that pivot so
        # the text's starting x stays at pdf_rect.x0 and the end x lands
        # at pdf_rect.x0 + target_width.
        morph = (baseline, fitz.Matrix(scale_x, 1.0))
        page.insert_text(
            baseline, text,
            fontsize=fontsize, fontname="helv",
            render_mode=3, color=(0, 0, 0),
            morph=morph,
        )




def _page_insert_textbox_html(
        page,
        rect,
        text,
        # fontsize=6,
        # fontname="helv",
        # render_mode=3,
        # color=(0, 0, 0),
        # align=0,
    ):
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    font_rel_width = 0.6 # monospace # width / height
    char_width = width / len(text)
    char_height = char_width / font_rel_width
    font_size = min(height, char_height)
    line_style = (
        f"left:{x0}px;" +
        f"top:{y0}px;" +
        f"width:{width}px;" +
        f"height:{height}px;" +
        f"font-size:{font_size}px;" +
        ""
    )
    page.write(f'<span class="line" style="{line_style}">{text}</span>\n')
