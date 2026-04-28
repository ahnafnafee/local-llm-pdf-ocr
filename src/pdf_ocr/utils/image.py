"""Image utilities for cropping page regions by normalized bounding box."""

import base64
import io

from PIL import Image, ImageStat


def is_blank_crop(
    image_base64: str,
    bbox: list[float],
    *,
    std_threshold: float = 12.0,
) -> bool:
    """
    Heuristic: True if the bbox region of the image is mostly uniform.

    Local VLMs hallucinate canned content (OlmOCR-2 falls back to the
    "The quick brown fox..." pangram) when shown a blank or near-blank
    crop. The refine stage feeds many such crops to the LLM whenever
    Surya detected a region that turned out not to contain text — empty
    notebook grid cells, blank margins between sections, etc. We short-
    circuit the OCR call for low-variance crops to keep their hallucinated
    output from polluting the final text layer.

    Threshold tuned for notebook backgrounds with light dot grids:
    a dot-only region has stddev ~7-8, a region with even a single
    handwritten character has stddev ≥20.
    """
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("L")
    w, h = img.size
    nx0, ny0, nx1, ny1 = bbox
    crop = img.crop(
        (int(nx0 * w), int(ny0 * h), int(nx1 * w), int(ny1 * h))
    )
    if crop.size[0] == 0 or crop.size[1] == 0:
        return True
    return ImageStat.Stat(crop).stddev[0] < std_threshold


def crop_for_ocr(
    image_base64: str,
    bbox: list[float],
    *,
    padding: float = 0.005,
    min_dim: int = 256,
    quality: int = 85,
    std_threshold: float = 12.0,
) -> str | None:
    """
    Decode the page image once, crop the padded bbox region, run the
    blank-region check on that *same padded crop*, and return the
    encoded JPEG — or ``None`` if the region is mostly uniform (so the
    caller can skip the LLM round-trip without polluting the output
    layer with hallucinated fallback content).

    Combining the two operations matters in dense-mode: a 150-box page
    that called :func:`is_blank_crop` and :func:`crop_box_to_base64`
    separately would decode the full-page image 300 times. Here we
    decode it once per box, and the blank check sees exactly the pixels
    the LLM would see (including the ``padding`` margin) so it can't
    short-circuit when the padded region picks up text just outside the
    raw bbox.
    """
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    w, h = img.size
    nx0, ny0, nx1, ny1 = bbox
    nx0 = max(0.0, nx0 - padding)
    ny0 = max(0.0, ny0 - padding)
    nx1 = min(1.0, nx1 + padding)
    ny1 = min(1.0, ny1 + padding)
    crop = img.crop((int(nx0 * w), int(ny0 * h), int(nx1 * w), int(ny1 * h)))
    if crop.size[0] == 0 or crop.size[1] == 0:
        return None
    if ImageStat.Stat(crop.convert("L")).stddev[0] < std_threshold:
        return None

    cw, ch = crop.size
    if cw < min_dim or ch < min_dim:
        scale = max(min_dim / max(1, cw), min_dim / max(1, ch))
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
