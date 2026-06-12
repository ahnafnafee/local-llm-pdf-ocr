"""Image utilities for cropping page regions by normalized bounding box.

Boxes that carry an honored tilted quad (``OrientedBox`` with tilt at or
above ``ANGLE_FLATTEN_DEG``) are perspective-rectified flat before being
handed to the VLM: the axis-aligned envelope of a tilted line swallows
the neighbouring rows above and below it, so an envelope crop makes the
model transcribe the neighbours instead of the target line. The warp
also presents glyphs upright — including fully sideways (±90°) runs,
whose reading-direction corner order the detector preserves — so no
separate "rotate tall crops" pass is needed.
"""

import base64
import io
import math

from PIL import Image, ImageDraw, ImageStat

from pdf_ocr.core.geometry import ANGLE_FLATTEN_DEG

# Upper bound on a rectified crop's edge, as a multiple of the page's
# longest edge. A line's quad can never legitimately exceed the page;
# anything bigger means corrupt detector geometry, and warping it would
# allocate an absurd image.
#
# Crop padding stays page-relative and is NOT capped to the box's own
# size: handwriting strokes routinely extend well past their detected
# box, and shrinking the margin measurably degrades transcription
# (capped crops disagree with stable uncapped reads far beyond LLM
# sampling noise). Neighbour-content bleed on dense print is handled by
# ``mask_boxes`` instead, which erases the neighbours without
# tightening the margin around the target's own ink.
_MAX_RECTIFIED_DIM_FACTOR = 2


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
    mask_boxes: list[list[float]] | None = None,
) -> str | None:
    """
    Decode the page image once, extract the padded bbox region —
    perspective-rectified flat when the box carries an honored tilted
    quad, axis-aligned otherwise — run the blank-region check on that
    *same* region, and return the encoded JPEG. Returns ``None`` if the
    region is mostly uniform (so the caller can skip the LLM round-trip
    without polluting the output layer with hallucinated fallback
    content).

    ``mask_boxes`` (axis-aligned crops only): normalized boxes of the
    page's OTHER detected regions. Any part of them visible inside the
    crop — but outside the target box itself — is painted paper-white
    before the blank check. On dense pages adjacent lines sit so close
    that even a tightly padded crop shows readable slivers of the
    neighbouring lines, and the VLM transcribes those too, duplicating
    their content across boxes. The target's own rect is never painted,
    so overlapping detections can't erase the content they share.
    Rectified (tilted-quad) crops ignore the masks — the warp follows
    the line's own quad and the masks' axis-aligned geometry doesn't
    map into that frame.

    Combining the operations matters in dense-mode: a 150-box page that
    called :func:`is_blank_crop` and :func:`crop_box_to_base64`
    separately would decode the full-page image 300 times. Here we
    decode it once per box, and the blank check sees exactly the pixels
    the LLM would see (including the ``padding`` margin, any masking,
    and any rectification) so it can't short-circuit on different
    content than the model receives.
    """
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    crop = _extract_region(img, bbox, padding, mask_boxes=mask_boxes)
    if crop.size[0] == 0 or crop.size[1] == 0:
        return None
    if ImageStat.Stat(crop.convert("L")).stddev[0] < std_threshold:
        return None
    return _encode_upscaled(crop, min_dim, quality)


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
    a postage stamp. Honored tilted quads are perspective-rectified flat,
    like :func:`crop_for_ocr` (but with no blank-region check).

    Args:
        image_base64: Base64-encoded image (full page).
        bbox: [nx0, ny0, nx1, ny1] in 0..1 space (optionally an
            ``OrientedBox`` carrying quad geometry).
        padding: Normalized padding added around the region before cropping.
        min_dim: Minimum dimension (px) to upscale the crop to.
        quality: JPEG quality for the returned image.
    """
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    crop = _extract_region(img, bbox, padding)
    return _encode_upscaled(crop, min_dim, quality)


# --- internals ---------------------------------------------------------------


def _extract_region(
    img: Image.Image, bbox, padding: float, mask_boxes=None,
) -> Image.Image:
    """Return the region for `bbox`: a flat perspective-rectified strip
    when the box carries an honored tilted quad, else the padded
    axis-aligned crop (with `mask_boxes` regions painted paper-white).
    Degenerate quads fall back to the envelope."""
    quad = getattr(bbox, "quad", None)
    angle = getattr(bbox, "angle", 0.0)
    if quad is not None and abs(angle) >= ANGLE_FLATTEN_DEG:
        crop = _rectified_crop(img, quad, padding)
        if crop is not None:
            return crop

    w, h = img.size
    nx0, ny0, nx1, ny1 = bbox
    window = (
        int(max(0.0, nx0 - padding) * w),
        int(max(0.0, ny0 - padding) * h),
        int(min(1.0, nx1 + padding) * w),
        int(min(1.0, ny1 + padding) * h),
    )
    crop = img.crop(window)
    if mask_boxes:
        target_px = (int(nx0 * w), int(ny0 * h), int(nx1 * w), int(ny1 * h))
        neighbors_px = [
            (int(m[0] * w), int(m[1] * h), int(m[2] * w), int(m[3] * h))
            for m in mask_boxes
        ]
        _mask_neighbor_boxes(crop, window, target_px, neighbors_px)
    return crop


def _mask_neighbor_boxes(crop, window, target, neighbors) -> None:
    """Paint paper-white every part of `neighbors` that is visible in
    the crop `window` but lies outside the `target` rect (all in page
    pixel coordinates). The target rect itself is never painted, so a
    neighbour overlapping the target can't erase shared content — at
    worst some of the neighbour's ink survives inside the target rect,
    which is the pre-masking status quo."""
    wx0, wy0, wx1, wy1 = window
    tx0, ty0, tx1, ty1 = target
    draw = ImageDraw.Draw(crop)
    for nx0, ny0, nx1, ny1 in neighbors:
        ix0, iy0 = max(nx0, wx0), max(ny0, wy0)
        ix1, iy1 = min(nx1, wx1), min(ny1, wy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        pieces = []
        if iy0 < ty0:  # strip above the target
            pieces.append((ix0, iy0, ix1, min(iy1, ty0)))
        if iy1 > ty1:  # strip below the target
            pieces.append((ix0, max(iy0, ty1), ix1, iy1))
        band0, band1 = max(iy0, ty0), min(iy1, ty1)
        if band0 < band1:  # strips beside the target, within its y-band
            if ix0 < tx0:
                pieces.append((ix0, band0, min(ix1, tx0), band1))
            if ix1 > tx1:
                pieces.append((max(ix0, tx1), band0, ix1, band1))
        for px0, py0, px1, py1 in pieces:
            if px1 > px0 and py1 > py0:
                # PIL rectangles are corner-inclusive; stop one pixel
                # short so a piece abutting the target never paints the
                # target's first row/column.
                draw.rectangle(
                    (px0 - wx0, py0 - wy0, px1 - wx0 - 1, py1 - wy0 - 1),
                    fill=(255, 255, 255),
                )


def _rectified_crop(
    img: Image.Image, quad, padding: float,
) -> Image.Image | None:
    """Warp the padded quad to a flat (top-edge × left-edge) strip.

    Output size comes from the quad's own edge lengths (longer of each
    opposing pair), so glyphs keep their on-page scale. Regions the
    expanded quad covers outside the image fill paper-white rather than
    black, which would otherwise read as ink to the VLM and skew the
    blank-region variance check.
    """
    pts = _quad_points_px(quad, img.width, img.height)
    pad_px = padding * (img.width + img.height) / 2.0
    if pad_px > 0:
        pts = _expand_quad(pts, pad_px)

    out_w = int(round(max(math.dist(pts[0], pts[1]), math.dist(pts[3], pts[2]))))
    out_h = int(round(max(math.dist(pts[0], pts[3]), math.dist(pts[1], pts[2]))))
    cap = _MAX_RECTIFIED_DIM_FACTOR * max(img.width, img.height)
    if out_w < 1 or out_h < 1 or out_w > cap or out_h > cap:
        return None

    coeffs = _perspective_coeffs(pts, out_w, out_h)
    if coeffs is None:
        return None
    return img.transform(
        (out_w, out_h),
        Image.Transform.PERSPECTIVE,
        data=coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def _quad_points_px(quad, width: int, height: int) -> list[tuple[float, float]]:
    """Normalized quad (8 floats or 4 pairs, clockwise from top-left) →
    pixel-space corner list."""
    if len(quad) == 8:
        pairs = [(quad[i], quad[i + 1]) for i in range(0, 8, 2)]
    else:
        pairs = [(p[0], p[1]) for p in quad]
    return [(x * width, y * height) for x, y in pairs]


def _expand_quad(
    pts: list[tuple[float, float]], pad: float,
) -> list[tuple[float, float]]:
    """Push each corner outward by `pad` px along its two edge directions,
    so the padding margin follows the quad's orientation instead of the
    page axes."""
    def unit(p, q):
        d = math.dist(p, q)
        return ((q[0] - p[0]) / d, (q[1] - p[1]) / d) if d > 0 else (0.0, 0.0)

    top = unit(pts[0], pts[1])
    bottom = unit(pts[3], pts[2])
    left = unit(pts[0], pts[3])
    right = unit(pts[1], pts[2])
    return [
        (pts[0][0] - pad * (top[0] + left[0]), pts[0][1] - pad * (top[1] + left[1])),
        (pts[1][0] + pad * (top[0] - right[0]), pts[1][1] + pad * (top[1] - right[1])),
        (pts[2][0] + pad * (bottom[0] + right[0]), pts[2][1] + pad * (bottom[1] + right[1])),
        (pts[3][0] - pad * (bottom[0] - left[0]), pts[3][1] - pad * (bottom[1] - left[1])),
    ]


def _perspective_coeffs(
    pts: list[tuple[float, float]], out_w: int, out_h: int,
) -> tuple[float, ...] | None:
    """PIL ``Image.transform(PERSPECTIVE)`` coefficients mapping every
    OUTPUT pixel (x, y) in [0, out_w]×[0, out_h] onto the INPUT quad
    `pts` (TL, TR, BR, BL pixel corners): the closed-form unit-square
    homography, rescaled to the output size. Returns ``None`` for
    degenerate quads.

    PIL's convention: ``data=(a, b, c, d, e, f, g, h)`` samples input
    ``((a·x + b·y + c) / (g·x + h·y + 1), (d·x + e·y + f) / (…))``.
    """
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
    rx = x0 - x1 + x2 - x3
    ry = y0 - y1 + y2 - y3
    dx1, dy1 = x1 - x2, y1 - y2
    dx2, dy2 = x3 - x2, y3 - y2
    den = dx1 * dy2 - dx2 * dy1
    if abs(den) < 1e-9:
        return None
    g = (rx * dy2 - dx2 * ry) / den
    h = (dx1 * ry - rx * dy1) / den
    a = x1 - x0 + g * x1
    b = x3 - x0 + h * x3
    d = y1 - y0 + g * y1
    e = y3 - y0 + h * y3
    return (
        a / out_w, b / out_h, x0,
        d / out_w, e / out_h, y0,
        g / out_w, h / out_h,
    )


def _encode_upscaled(crop: Image.Image, min_dim: int, quality: int) -> str:
    """JPEG-encode `crop`, upscaling first so the VLM can read the glyphs."""
    cw, ch = crop.size
    if cw < min_dim or ch < min_dim:
        scale = max(min_dim / max(1, cw), min_dim / max(1, ch))
        crop = crop.resize(
            (max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS
        )
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
