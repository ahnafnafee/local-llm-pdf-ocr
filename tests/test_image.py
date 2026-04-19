"""Unit tests for the crop utility used by the refine stage."""

from __future__ import annotations

import base64
import io

from PIL import Image

from src.pdf_ocr.utils.image import crop_box_to_base64


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
