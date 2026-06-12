"""Unit tests for the crop utilities used by the refine and dense stages."""

from __future__ import annotations

import base64
import io
import math

from PIL import Image, ImageDraw

from pdf_ocr.core.geometry import OrientedBox
from pdf_ocr.utils.image import crop_box_to_base64, crop_for_ocr


def _make_image_b64(size=(800, 1000)) -> str:
    img = Image.new("RGB", size, "white")
    # Paint a visible patch so we can verify the crop roughly captured it.
    for y in range(400, 600):
        for x in range(100, 300):
            img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_b64_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def test_crop_returns_valid_image():
    page = _make_image_b64()
    bbox = [0.1, 0.4, 0.4, 0.6]
    crop_b64 = crop_box_to_base64(page, bbox)
    out = _decode_b64_image(crop_b64)
    assert out.width > 0 and out.height > 0


def test_crop_upscales_tiny_regions():
    page = _make_image_b64(size=(1000, 1000))
    tiny_bbox = [0.0, 0.0, 0.02, 0.02]  # 20x20 pixels raw
    crop_b64 = crop_box_to_base64(page, tiny_bbox, min_dim=256)
    out = _decode_b64_image(crop_b64)
    # The helper should upscale so the VLM can read glyphs.
    assert out.width >= 256 or out.height >= 256


def test_crop_captures_painted_region():
    page = _make_image_b64()  # red patch in (100..300, 400..600)
    bbox = [0.1, 0.4, 0.4, 0.6]  # same region normalized
    crop_b64 = crop_box_to_base64(page, bbox, padding=0.0)
    out = _decode_b64_image(crop_b64)
    # Center pixel should be red-ish (JPEG compression is forgiving).
    cx, cy = out.width // 2, out.height // 2
    r, g, b = out.getpixel((cx, cy))
    assert r > 150 and g < 100 and b < 100, f"expected red-ish, got {(r,g,b)}"


def test_crop_clamps_out_of_range_bbox():
    page = _make_image_b64()
    # Negative + >1 coords: helper must clamp without crashing.
    crop_b64 = crop_box_to_base64(page, [-0.1, -0.1, 1.2, 1.2])
    out = _decode_b64_image(crop_b64)
    assert out.width > 0 and out.height > 0


# --- neighbor masking ---------------------------------------------------------


def test_padding_is_page_relative():
    # Padding must stay page-relative — NOT shrink with the box.
    # Handwriting strokes routinely extend past their detected box, and
    # a margin capped to the box's own size measurably degrades reads;
    # dense-print neighbour bleed is handled by mask_boxes instead.
    page = _make_image_b64(size=(1000, 1000))
    short = _decode_b64_image(
        crop_box_to_base64(page, [0.1, 0.5, 0.9, 0.51], min_dim=1)  # 10px tall
    )
    roomy = _decode_b64_image(
        crop_box_to_base64(page, [0.4, 0.4, 0.6, 0.6], min_dim=1)  # 200px tall
    )
    # Both get the same ~5px pad per side.
    assert 18 <= short.height <= 22
    assert 205 <= roomy.height <= 215


def _page_with_bands(bands, size=(1000, 1000)) -> str:
    """White page with dark horizontal bands at the given normalized
    (y0, y1) ranges."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for y0, y1 in bands:
        draw.rectangle(
            (0, int(y0 * size[1]), size[0] - 1, int(y1 * size[1]) - 1),
            fill=(20, 20, 20),
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_mask_boxes_hides_neighbor_ink():
    # Two stacked text lines; the target's padded crop reaches into the
    # neighbour's band. With the neighbour declared via mask_boxes, its
    # ink must come back paper-white; without, it stays dark.
    line_a = [0.1, 0.40, 0.9, 0.45]
    line_b = [0.1, 0.452, 0.9, 0.50]
    page = _page_with_bands([(0.40, 0.45), (0.452, 0.50)])

    masked = _decode_b64_image(
        crop_for_ocr(page, line_a, padding=0.01, min_dim=1, mask_boxes=[line_b])
    )
    unmasked = _decode_b64_image(crop_for_ocr(page, line_a, padding=0.01, min_dim=1))
    assert masked.size == unmasked.size
    # padding 0.01 -> 10px: crop spans page y [390, 460); the neighbour's
    # rows start at 452 -> crop row 62.
    nb_row = 452 - 390 + 2
    assert unmasked.getpixel((unmasked.width // 2, nb_row))[0] < 100
    assert masked.getpixel((masked.width // 2, nb_row))[0] > 200
    # The target's own band is untouched.
    own_row = 425 - 390
    assert masked.getpixel((masked.width // 2, own_row))[0] < 100


def test_mask_never_covers_target_region():
    # A neighbour overlapping the target: only the part of the neighbour
    # OUTSIDE the target rect is masked — the shared band keeps its ink.
    target = [0.1, 0.40, 0.9, 0.50]
    overlapping_neighbor = [0.1, 0.48, 0.9, 0.58]
    page = _page_with_bands([(0.40, 0.58)])

    out = _decode_b64_image(
        crop_for_ocr(
            page, target, padding=0.005, min_dim=1,
            mask_boxes=[overlapping_neighbor],
        )
    )
    # pad = 5px -> crop spans page y [395, 505). Inside the target
    # (page y 490): dark, even though the neighbour also covers it.
    assert out.getpixel((out.width // 2, 490 - 395))[0] < 100
    # Below the target (page y 502): the neighbour's exclusive zone is
    # masked white.
    assert out.getpixel((out.width // 2, 502 - 395))[0] > 200


def test_mask_boxes_ignored_on_rectified_quads():
    # The rectified branch warps along the line's own quad; neighbour
    # masking is an axis-aligned concept and must not break the warp.
    quad_px = _rot_quad_px(25.0)
    page = _page_with_bar(quad_px)
    crop_b64 = crop_for_ocr(
        page, _oriented_from_quad(quad_px, 25.0),
        mask_boxes=[[0.0, 0.0, 0.2, 0.1]],
    )
    assert crop_b64 is not None


def test_blank_check_runs_after_masking():
    # A target whose padded crop is non-blank ONLY because of neighbour
    # ink must be treated as blank once the neighbour is masked. The
    # neighbour box spans the full ink band (the page's bands are drawn
    # full-width).
    target = [0.1, 0.40, 0.9, 0.45]          # blank band
    neighbor = [0.0, 0.452, 1.0, 0.50]       # inked band
    page = _page_with_bands([(0.452, 0.50)])
    assert crop_for_ocr(page, target, padding=0.01, min_dim=1) is not None
    assert (
        crop_for_ocr(page, target, padding=0.01, min_dim=1, mask_boxes=[neighbor])
        is None
    )


# --- perspective-rectified quad crops ---------------------------------------

_PAGE_W, _PAGE_H = 800, 600


def _rot_quad_px(angle_deg, p0=(150.0, 200.0), run=300.0, h=40.0):
    """Rotated-rect corners, clockwise from the text's top-left."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -math.sin(a), math.cos(a)
    return [
        p0,
        (p0[0] + run * dx, p0[1] + run * dy),
        (p0[0] + run * dx + h * px, p0[1] + run * dy + h * py),
        (p0[0] + h * px, p0[1] + h * py),
    ]


def _page_with_bar(quad_px) -> str:
    """White page with a dark bar filling exactly the quad."""
    img = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
    ImageDraw.Draw(img).polygon([(x, y) for x, y in quad_px], fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _oriented_from_quad(quad_px, angle: float) -> OrientedBox:
    xs = [p[0] for p in quad_px]
    ys = [p[1] for p in quad_px]
    quad = []
    for x, y in quad_px:
        quad.extend((x / _PAGE_W, y / _PAGE_H))
    return OrientedBox(
        [min(xs) / _PAGE_W, min(ys) / _PAGE_H, max(xs) / _PAGE_W, max(ys) / _PAGE_H],
        quad=quad, angle=angle,
    )


def test_rectified_crop_is_flat_strip():
    # A 25°-tilted bar must come back as a flat, run-long strip: dark
    # across the middle rows, white padding above and below.
    quad_px = _rot_quad_px(25.0)
    page = _page_with_bar(quad_px)
    crop_b64 = crop_for_ocr(page, _oriented_from_quad(quad_px, 25.0))
    assert crop_b64 is not None
    out = _decode_b64_image(crop_b64).convert("L")

    aspect = out.width / out.height
    assert 5.5 < aspect < 7.5, f"expected ~300/40 strip, got aspect {aspect:.2f}"

    def row_mean(y: int) -> float:
        row = [out.getpixel((x, y)) for x in range(0, out.width, 16)]
        return sum(row) / len(row)

    mid = out.height // 2
    assert row_mean(mid) < 90, "bar must fill the strip's middle rows"
    assert row_mean(2) > 180, "top padding must stay paper-white"
    assert row_mean(out.height - 3) > 180, "bottom padding must stay paper-white"


def test_rectified_crop_flatter_than_envelope():
    # The same region cropped through the plain-list path keeps the
    # inflated axis-aligned envelope; the quad path must be much flatter.
    quad_px = _rot_quad_px(25.0)
    page = _page_with_bar(quad_px)

    oriented = _oriented_from_quad(quad_px, 25.0)
    rect = _decode_b64_image(crop_for_ocr(page, oriented))
    envelope = _decode_b64_image(crop_for_ocr(page, list(oriented)))

    rect_aspect = rect.width / rect.height
    env_aspect = envelope.width / envelope.height
    assert rect_aspect > 3 * env_aspect


def test_sideways_quad_rectifies_to_wide_strip():
    # ±90°-ish runs: the corner order encodes the reading direction, so
    # rectification alone straightens sideways text — no extra rotate
    # pass needed.
    quad_px = _rot_quad_px(85.0, p0=(500.0, 100.0))
    page = _page_with_bar(quad_px)
    crop_b64 = crop_for_ocr(page, _oriented_from_quad(quad_px, 85.0))
    assert crop_b64 is not None
    out = _decode_b64_image(crop_b64)
    assert out.width / out.height > 4


def test_blank_quad_region_returns_none():
    quad_px = _rot_quad_px(25.0)
    page = _page_with_bar(quad_px)
    # Same shape, but over the empty bottom-right of the page.
    blank_quad = _rot_quad_px(20.0, p0=(420.0, 420.0), run=250.0, h=30.0)
    assert crop_for_ocr(page, _oriented_from_quad(blank_quad, 20.0)) is None


def test_degenerate_quad_falls_back_to_envelope():
    quad_px = _rot_quad_px(25.0)
    page = _page_with_bar(quad_px)
    box = _oriented_from_quad(quad_px, 45.0)
    box.quad = [0.4, 0.4] * 4  # all corners coincide
    # Falls back to the (non-blank) axis-aligned envelope crop.
    assert crop_for_ocr(page, box) is not None


def test_sub_threshold_angle_keeps_envelope_crop():
    # Below the flatten threshold the quad is detector noise; the crop
    # must stay the plain axis-aligned envelope.
    quad_px = _rot_quad_px(1.5)
    page = _page_with_bar(quad_px)
    oriented = _oriented_from_quad(quad_px, 1.5)
    a = _decode_b64_image(crop_for_ocr(page, oriented))
    b = _decode_b64_image(crop_for_ocr(page, list(oriented)))
    assert (a.width, a.height) == (b.width, b.height)


def test_crop_box_to_base64_also_rectifies():
    quad_px = _rot_quad_px(25.0)
    page = _page_with_bar(quad_px)
    out = _decode_b64_image(
        crop_box_to_base64(page, _oriented_from_quad(quad_px, 25.0))
    )
    assert out.width / out.height > 5


def test_thin_quad_keeps_page_relative_padding():
    # A 10px-tall tilted line keeps the page-relative pad (3.5px/side on
    # this 800x600 page -> ~17px strip): tilted lines are handwriting
    # territory, where stroke extents exceed the detected quad and a
    # tighter margin degrades transcription.
    quad_px = _rot_quad_px(25.0, h=10.0)
    page = _page_with_bar(quad_px)
    out = _decode_b64_image(
        crop_box_to_base64(page, _oriented_from_quad(quad_px, 25.0), min_dim=1)
    )
    assert 15 <= out.height <= 19
