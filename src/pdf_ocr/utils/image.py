"""Image utilities for cropping page regions by normalized bounding box."""

import base64
import io

from PIL import Image


def crop_box_to_base64(
    image_base64: str,
    bbox: list[float],
    *,
    padding: float = 0.005,
    min_dim: int = 256,
    quality: int = 85,
) -> str:
    """
    Crop a normalized bbox region out of a base64 image and return the crop
    as a new base64 JPEG string, upscaled if needed so the VLM doesn't see
    a postage stamp.

    Args:
        image_base64: Base64-encoded image (full page).
        bbox: [nx0, ny0, nx1, ny1] in 0..1 space.
        padding: Normalized padding added around the bbox before cropping.
        min_dim: Minimum dimension (px) to upscale the crop to.
        quality: JPEG quality for the returned image.
    """
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    w, h = img.size

    nx0, ny0, nx1, ny1 = bbox
    nx0 = max(0.0, nx0 - padding)
    ny0 = max(0.0, ny0 - padding)
    nx1 = min(1.0, nx1 + padding)
    ny1 = min(1.0, ny1 + padding)

    crop = img.crop((int(nx0 * w), int(ny0 * h), int(nx1 * w), int(ny1 * h)))

    # Upscale small crops so the VLM can read the glyphs.
    cw, ch = crop.size
    if cw < min_dim or ch < min_dim:
        scale = max(min_dim / max(1, cw), min_dim / max(1, ch))
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
