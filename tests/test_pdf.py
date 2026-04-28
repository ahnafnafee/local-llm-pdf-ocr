"""
PDFHandler tests using the on-disk example PDFs.

These validate the conversion → embed round trip without hitting Surya or
the LLM. No model loads, so they run in well under a second.
"""

from __future__ import annotations

import base64
from pathlib import Path

import fitz
import pytest

from src.pdf_ocr.core.pdf import PDFHandler


@pytest.fixture
def pdf_handler() -> PDFHandler:
    return PDFHandler()


@pytest.mark.parametrize("name", ["digital.pdf", "hybrid.pdf", "handwritten.pdf"])
def test_convert_to_images_produces_base64_per_page(
    pdf_handler: PDFHandler, example_pdfs: dict[str, Path], name: str
):
    images = pdf_handler.convert_to_images(str(example_pdfs[name]))

    # At least one page, indexed from 0, contiguous.
    assert images, f"{name}: no images returned"
    assert min(images) == 0
    assert list(images.keys()) == sorted(images.keys())

    # Each value should be decodable base64 of a non-empty image.
    for page_num, b64 in images.items():
        raw = base64.b64decode(b64)
        assert len(raw) > 256, f"{name} page {page_num}: suspiciously small ({len(raw)} bytes)"


def test_embed_structured_text_produces_searchable_pdf(
    pdf_handler: PDFHandler, example_pdfs: dict[str, Path], tmp_path: Path
):
    """Embed a known text marker and verify it's recoverable via get_text."""
    input_pdf = str(example_pdfs["digital.pdf"])
    output_pdf = str(tmp_path / "out.pdf")

    marker = "ZZUNIQUEMARKERZZ"
    pages_data = {0: [([0.1, 0.1, 0.6, 0.15], marker)]}

    pdf_handler.embed_structured_text(input_pdf, output_pdf, pages_data, dpi=150)

    doc = fitz.open(output_pdf)
    try:
        text = doc[0].get_text("text")
    finally:
        doc.close()
    assert marker in text, f"expected marker {marker!r} in embedded text; got {text[:200]!r}"


def test_embed_handles_empty_text_gracefully(
    pdf_handler: PDFHandler, example_pdfs: dict[str, Path], tmp_path: Path
):
    input_pdf = str(example_pdfs["digital.pdf"])
    output_pdf = str(tmp_path / "out_empty.pdf")
    # A real pipeline output: one box with text, one empty (DP-skipped).
    pages_data = {
        0: [
            ([0.1, 0.1, 0.6, 0.15], "hello world"),
            ([0.1, 0.2, 0.6, 0.25], ""),
        ]
    }
    pdf_handler.embed_structured_text(input_pdf, output_pdf, pages_data, dpi=150)

    doc = fitz.open(output_pdf)
    try:
        text = doc[0].get_text("text")
    finally:
        doc.close()
    assert "hello world" in text


@pytest.mark.parametrize("name", ["digital.pdf", "hybrid.pdf", "handwritten.pdf"])
def test_output_preserves_page_count(
    pdf_handler: PDFHandler, example_pdfs: dict[str, Path], tmp_path: Path, name: str
):
    input_pdf = str(example_pdfs[name])
    output_pdf = str(tmp_path / f"out_{name}")

    with fitz.open(input_pdf) as src:
        src_pages = len(src)

    # No text to embed — still a valid sandwich PDF of the same length.
    pdf_handler.embed_structured_text(input_pdf, output_pdf, {}, dpi=100)

    with fitz.open(output_pdf) as out:
        assert len(out) == src_pages


# --- raw image inputs (JPEG / PNG / multi-page TIFF) -----------------------


def _make_image_file(path: Path, size=(800, 1000), mode="RGB") -> Path:
    """Write a simple JPEG/PNG/TIFF with some content to `path`."""
    from PIL import Image, ImageDraw
    img = Image.new(mode, size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 200, 700, 300], fill="lightgray")
    fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
           "tif": "TIFF", "tiff": "TIFF"}[path.suffix.lower().lstrip(".")]
    img.save(path, format=fmt)
    return path


def _make_multiframe_tiff(path: Path, n_frames: int = 3, size=(600, 800)) -> Path:
    from PIL import Image, ImageDraw
    frames = []
    for i in range(n_frames):
        img = Image.new("RGB", size, "white")
        ImageDraw.Draw(img).text((50, 50), f"Page {i+1}", fill="black")
        frames.append(img)
    frames[0].save(path, format="TIFF", save_all=True, append_images=frames[1:])
    return path


@pytest.mark.parametrize("ext", ["jpg", "png", "tif"])
def test_convert_image_file_to_single_page(
    pdf_handler: PDFHandler, tmp_path: Path, ext: str
):
    src = _make_image_file(tmp_path / f"scan.{ext}")
    images = pdf_handler.convert_to_images(str(src))
    assert list(images.keys()) == [0]

    import base64
    from PIL import Image as PILImage
    import io
    raw = base64.b64decode(images[0])
    img = PILImage.open(io.BytesIO(raw))
    assert img.size[0] > 0 and img.size[1] > 0


def test_convert_multiframe_tiff(pdf_handler: PDFHandler, tmp_path: Path):
    src = _make_multiframe_tiff(tmp_path / "pages.tif", n_frames=4)
    images = pdf_handler.convert_to_images(str(src))
    assert sorted(images.keys()) == [0, 1, 2, 3]


def test_embed_into_image_input_produces_searchable_pdf(
    pdf_handler: PDFHandler, tmp_path: Path
):
    """Image → sandwich PDF round-trip: marker must be extractable."""
    src = _make_image_file(tmp_path / "scan.png")
    output_pdf = str(tmp_path / "out.pdf")

    marker = "IMAGEMARKERXYZ"
    pages_data = {0: [([0.15, 0.35, 0.55, 0.40], marker)]}
    pdf_handler.embed_structured_text(str(src), output_pdf, pages_data)

    with fitz.open(output_pdf) as doc:
        assert len(doc) == 1
        assert marker in doc[0].get_text("text")


def test_embed_multiframe_tiff_produces_multipage_pdf(
    pdf_handler: PDFHandler, tmp_path: Path
):
    src = _make_multiframe_tiff(tmp_path / "multi.tiff", n_frames=3)
    output_pdf = str(tmp_path / "multi_out.pdf")

    pages_data = {
        0: [([0.1, 0.1, 0.4, 0.15], "MARKERPAGEONE")],
        2: [([0.1, 0.1, 0.4, 0.15], "MARKERPAGETHREE")],
    }
    pdf_handler.embed_structured_text(str(src), output_pdf, pages_data)

    with fitz.open(output_pdf) as doc:
        assert len(doc) == 3
        assert "MARKERPAGEONE" in doc[0].get_text("text")
        assert doc[1].get_text("text").strip() == ""  # no text embedded on page 2
        assert "MARKERPAGETHREE" in doc[2].get_text("text")


def test_image_extension_detection():
    from src.pdf_ocr.core.pdf import _is_image_path
    assert _is_image_path("scan.jpg")
    assert _is_image_path("SCAN.JPEG")
    assert _is_image_path("pages.tiff")
    assert _is_image_path("page.png")
    assert _is_image_path("photo.webp")
    assert _is_image_path("photo.avif")
    assert _is_image_path("PHOTO.AVIF")
    assert not _is_image_path("doc.pdf")
    assert not _is_image_path("doc.PDF")
    assert not _is_image_path("notes.txt")


def test_is_blank_crop_distinguishes_blank_from_text(tmp_path: Path):
    """Stddev-based blank detector: blank crops short-circuit refine
    before paying for an LLM call (OlmOCR hallucinates canned text on
    blank input)."""
    from PIL import Image, ImageDraw
    import base64
    import io
    from src.pdf_ocr.utils.image import is_blank_crop

    # Build a page-like image: top half blank-white, bottom half has text.
    img = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(img)
    # Sprinkle a few light dots like a notebook grid (low variance).
    for x in range(0, 800, 40):
        for y in range(0, 500, 40):
            draw.point((x, y), fill="lightgray")
    # Bold text in bottom half (high variance vs white background).
    draw.rectangle([100, 700, 700, 800], fill="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    blank_bbox = [0.05, 0.05, 0.95, 0.45]   # top half — dots only
    text_bbox = [0.10, 0.70, 0.90, 0.85]    # bottom half — solid text
    assert is_blank_crop(b64, blank_bbox), "dotted background must be blank"
    assert not is_blank_crop(b64, text_bbox), "solid text region must not be blank"


def test_avif_input_round_trip(pdf_handler: PDFHandler, tmp_path: Path):
    """AVIF input must decode and embed end-to-end (pillow-avif-plugin)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (800, 1000), "white")
    ImageDraw.Draw(img).rectangle([100, 200, 700, 300], fill="lightgray")
    src = tmp_path / "scan.avif"
    img.save(src, format="AVIF", quality=80)

    images = pdf_handler.convert_to_images(str(src))
    assert list(images.keys()) == [0]

    output_pdf = str(tmp_path / "out.pdf")
    marker = "AVIFMARKERZZ"
    pdf_handler.embed_structured_text(
        str(src), output_pdf, {0: [([0.15, 0.35, 0.55, 0.40], marker)]}
    )
    with fitz.open(output_pdf) as doc:
        assert len(doc) == 1
        assert marker in doc[0].get_text("text")
