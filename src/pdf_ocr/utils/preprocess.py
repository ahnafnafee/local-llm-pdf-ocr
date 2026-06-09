"""Photo preprocessing: rectify a photographed page for detection/OCR
while keeping every output coordinate on the ORIGINAL image.

Camera captures of documents arrive tilted, perspective-distorted, and
unevenly lit (a sheet photographed on a desk). Detectors and vision
models read a rectified, flattened version far better — but the
deliverable must overlay the photo the user actually has. The page
quad found here therefore yields BOTH directions: the forward warp
(original → rectified) that feeds detection and OCR, and the inverse
homography used to bring every detected box back into original pixel
space before the writers run. The writers always rasterize from the
original input file, so only box coordinates need mapping.

Engagement is conservative by design: a page only rectifies when a
convex 4-corner page outline is confidently found AND it deviates from
an axis-aligned rectangle enough to matter. Flat scans and rasterized
PDF pages (white page filling the frame, no contrasting border) find
no quad and pass through untouched, so the existing pipeline behavior
is unchanged for non-photo inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# A candidate page outline must cover at least this fraction of the
# frame — smaller quads are objects on the desk, not the page.
_MIN_AREA_FRAC = 0.18
# ... and at most this fraction: a quad hugging the full frame is the
# frame itself (flat scan), where rectification is a no-op with risk.
_MAX_AREA_FRAC = 0.97
# Rectify only when the quad meaningfully disagrees with its own
# axis-aligned bounding box; below this the capture is already flat.
_MIN_PERSPECTIVE_DEVIATION = 0.015


@dataclass
class PageRectification:
    """One page's forward warp result + the inverse mapping home."""

    rectified_bgr: np.ndarray   # warped (and illumination-flattened) image
    h_inv: np.ndarray           # rectified px -> original px homography
    original_size: tuple[int, int]   # (width, height) of the source image
    rectified_size: tuple[int, int]  # (width, height) of the warp target


def find_page_quad(img_bgr: np.ndarray) -> np.ndarray | None:
    """Locate the photographed page's outline as a convex 4-point quad.

    Returns corners ordered clockwise from top-left in pixel space, or
    None when no plausible page outline exists (typical for flat scans:
    a white page filling the frame has no contrasting border contour).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu separates a bright page from a darker desk without tuning;
    # Canny alone fragments on textured backgrounds (carpet, bedding).
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8),
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    frame_area = float(w * h)
    best: np.ndarray | None = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (_MIN_AREA_FRAC * frame_area <= area <= _MAX_AREA_FRAC * frame_area):
            continue
        approx = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True,
        )
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)
    if best is None:
        return None
    return _order_corners(best)


def perspective_deviation(quad: np.ndarray) -> float:
    """How far the quad strays from its own axis-aligned bounding box,
    as a fraction of the bounding box diagonal. ~0 for flat captures."""
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    rect = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32,
    )
    diag = math.hypot(x1 - x0, y1 - y0)
    if diag <= 0:
        return 0.0
    return float(np.linalg.norm(quad - rect, axis=1).max() / diag)


def should_rectify(quad: np.ndarray | None) -> bool:
    return quad is not None and perspective_deviation(quad) >= _MIN_PERSPECTIVE_DEVIATION


def rectify_page(img_bgr: np.ndarray, quad: np.ndarray) -> PageRectification:
    """Warp the page quad to a flat rectangle and flatten illumination.

    The target size comes from the quad's own edge lengths, so the
    rectified image keeps the page's physical aspect ratio.
    """
    (tl, tr, br, bl) = quad
    target_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    target_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    target_w = max(32, target_w)
    target_h = max(32, target_h)

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    h_fwd = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(img_bgr, h_fwd, (target_w, target_h))
    warped = _flatten_illumination(warped)

    h, w = img_bgr.shape[:2]
    return PageRectification(
        rectified_bgr=warped,
        h_inv=np.linalg.inv(h_fwd),
        original_size=(w, h),
        rectified_size=(target_w, target_h),
    )


def map_points_to_original(
    points_px: np.ndarray, rect: PageRectification,
) -> np.ndarray:
    """Map (N,2) rectified-space pixel points back onto the original."""
    pts = points_px.reshape(-1, 1, 2).astype(np.float64)
    mapped = cv2.perspectiveTransform(pts, rect.h_inv)
    return mapped.reshape(-1, 2)


def _flatten_illumination(img_bgr: np.ndarray) -> np.ndarray:
    """Divide out low-frequency lighting so shadows and gradients from
    the photo don't bias detection thresholds or the vision model."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    flattened = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return cv2.cvtColor(flattened, cv2.COLOR_GRAY2BGR)


def _order_corners(quad: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary corners clockwise from top-left."""
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).reshape(-1)
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(d)]
    bl = quad[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# --- pipeline-facing helpers -------------------------------------------------


def rectify_page_images(
    images_b64: dict[int, str], page_nums: list[int], mode: str,
) -> tuple[dict[int, str], dict[int, PageRectification]]:
    """Rectify the pages that warrant it; leave the rest untouched.

    Returns the (possibly partially replaced) base64 image dict plus a
    per-page :class:`PageRectification` for every page that was warped.
    ``mode``: "auto" engages on a confident page quad with meaningful
    perspective deviation; "always" engages whenever a quad is found.
    """
    import base64

    out = dict(images_b64)
    rect_maps: dict[int, PageRectification] = {}
    for p in page_nums:
        raw = base64.b64decode(images_b64[p])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        quad = find_page_quad(img)
        engage = (
            quad is not None if mode == "always" else should_rectify(quad)
        )
        if not engage:
            continue
        rect = rectify_page(img, quad)
        ok, buf = cv2.imencode(
            ".jpg", rect.rectified_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not ok:
            continue
        out[p] = base64.b64encode(buf.tobytes()).decode()
        rect_maps[p] = rect
    return out, rect_maps


def map_structured_to_original(
    pages_structured: dict[int, list],
    rect_maps: dict[int, PageRectification],
) -> dict[int, list]:
    """Bring every (box, text) pair on rectified pages back onto the
    original image, as OrientedBoxes whose quads carry the perspective.

    Rectangles in rectified space become arbitrary convex quads on the
    original photo; the rotation-aware writers place text along those
    quads, so the overlay follows the page exactly as photographed.
    """
    from pdf_ocr.core.geometry import OrientedBox, quad_angle_deg

    out: dict[int, list] = {}
    for p, items in pages_structured.items():
        rect = rect_maps.get(p)
        if rect is None:
            out[p] = items
            continue
        rw, rh = rect.rectified_size
        ow, oh = rect.original_size
        mapped_items = []
        for box, text in items:
            quad_norm = getattr(box, "quad", None)
            if quad_norm is not None:
                pts = np.array(
                    [[quad_norm[i] * rw, quad_norm[i + 1] * rh]
                     for i in range(0, 8, 2)],
                    dtype=np.float64,
                )
            else:
                x0, y0, x1, y1 = box
                pts = np.array(
                    [[x0 * rw, y0 * rh], [x1 * rw, y0 * rh],
                     [x1 * rw, y1 * rh], [x0 * rw, y1 * rh]],
                    dtype=np.float64,
                )
            mapped = map_points_to_original(pts, rect)
            flat = [c for pt in mapped for c in pt]
            angle = quad_angle_deg(flat)
            xs, ys = mapped[:, 0], mapped[:, 1]
            aabb = [
                _clamp01(float(xs.min()) / ow), _clamp01(float(ys.min()) / oh),
                _clamp01(float(xs.max()) / ow), _clamp01(float(ys.max()) / oh),
            ]
            quad_out = []
            for px, py in mapped:
                quad_out.extend(
                    (_clamp01(float(px) / ow), _clamp01(float(py) / oh)),
                )
            mapped_items.append((
                OrientedBox(
                    aabb, quad=quad_out, angle=angle,
                    confidence=getattr(box, "confidence", None),
                ),
                text,
            ))
        out[p] = mapped_items
    return out


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))
