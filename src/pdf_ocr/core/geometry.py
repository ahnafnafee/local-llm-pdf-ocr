"""Oriented box geometry shared across detector, pipeline, and writers.

The pipeline's interchange shape for a detected region is a normalized
axis-aligned bbox ``[nx0, ny0, nx1, ny1]`` in 0..1. Detectors that know
more than that — Surya's min-area-rect polygons carry rotation and a
confidence — attach the extra geometry via :class:`OrientedBox`, a
``list`` subclass, so every existing consumer of plain 4-float bboxes
(DP aligner, output writers, JSON serialization, tests) keeps working
unchanged while rotation-aware writers read the extra attributes.
"""

from __future__ import annotations

import math

# Below this tilt, writers treat a line as horizontal. Min-area rects
# fitted to segmentation heatmaps wobble a degree or two on perfectly
# straight printed text; rotating the overlay by that noise would make
# selection geometry worse, not better. Genuine skew that hurts
# selection congruence starts well past this. Mirrors Tesseract's
# ClipBaseline policy of projecting near-horizontal baselines flat.
ANGLE_FLATTEN_DEG = 3.0


class OrientedBox(list):
    """Axis-aligned bbox that also carries rotated-quad geometry.

    Behaves exactly like the plain ``[nx0, ny0, nx1, ny1]`` list it
    subclasses; the bbox is the envelope of ``quad`` when one is set.

    Attributes:
        quad: 8 normalized floats ``(x0, y0, x1, y1, x2, y2, x3, y3)``,
            corners clockwise from top-left, or ``None`` when the
            source only had an axis-aligned box.
        angle: rotation of the text line in degrees, screen/image
            coordinates (y down) — positive tilts the line's right end
            downward, matching CSS ``rotate()``. ``0.0`` for
            axis-aligned sources.
        confidence: detector confidence, or ``None``. Surya's value is
            normalized per page (the strongest box on a page is 1.0),
            so it ranks boxes within a page rather than across pages.
    """

    def __init__(self, aabb, quad=None, angle: float = 0.0, confidence=None):
        super().__init__(aabb)
        self.quad = list(quad) if quad is not None else None
        self.angle = float(angle)
        self.confidence = confidence

    @property
    def is_rotated(self) -> bool:
        """True when the tilt is large enough for writers to honor."""
        return self.quad is not None and abs(self.angle) >= ANGLE_FLATTEN_DEG


def quad_angle_deg(quad_px) -> float:
    """Rotation of a quad's top edge, degrees, screen coordinates.

    Must be computed from PIXEL-space corners — normalizing x and y by
    different page dimensions first would distort the angle. Input is
    8 floats or 4 ``(x, y)`` pairs, clockwise from top-left.
    """
    pts = _as_points(quad_px)
    (x0, y0), (x1, y1) = pts[0], pts[1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def quad_edge_lengths(quad, page_width: float, page_height: float) -> tuple[float, float]:
    """(top-edge length, left-edge length) of a normalized quad, in pixels.

    Edge lengths must be measured in pixel space for the same reason as
    :func:`quad_angle_deg`; pass the page dimensions the quad was
    normalized against.
    """
    pts = [(x * page_width, y * page_height) for x, y in _as_points(quad)]
    top = math.dist(pts[0], pts[1])
    left = math.dist(pts[0], pts[3])
    return top, left


def quad_affine_components(
    quad, page_width: float, page_height: float,
) -> tuple[float, float, float, float]:
    """CSS ``matrix()`` linear components ``(a, b, c, d)`` mapping a span
    laid out with the quad's edge lengths onto the quad's orientation.

    The span's layout box is (top-edge length × left-edge length); the
    returned columns are the pixel-space unit vectors of those two edges,
    so the transform rotates/shears the box onto the quad without scaling
    glyphs. For a rotated rectangle this is exactly ``rotate(angle)``;
    sheared parallelograms come out exact as well. The values are
    dimensionless ratios of pixel lengths, so they stay correct at any
    uniform page rendering scale — pure CSS, no script needed.
    """
    pts = [(x * page_width, y * page_height) for x, y in _as_points(quad)]
    top = math.dist(pts[0], pts[1])
    left = math.dist(pts[0], pts[3])
    if top <= 0 or left <= 0:
        return (1.0, 0.0, 0.0, 1.0)
    return (
        (pts[1][0] - pts[0][0]) / top,
        (pts[1][1] - pts[0][1]) / top,
        (pts[3][0] - pts[0][0]) / left,
        (pts[3][1] - pts[0][1]) / left,
    )


def quad_perspective_residual_px(
    quad, page_width: float, page_height: float,
) -> float:
    """Distance in pixels between the quad's bottom-right corner and the
    parallelogram completion of the other three (``TR + BL - TL``).

    Zero for rectangles and parallelograms — Surya min-area rects land
    here — and grows with keystone distortion, e.g. boxes mapped back
    through the photo path's inverse homography. Writers use it to decide
    whether an affine transform reproduces the quad exactly or a full
    perspective mapping is required.
    """
    pts = [(x * page_width, y * page_height) for x, y in _as_points(quad)]
    completion = (
        pts[1][0] + pts[3][0] - pts[0][0],
        pts[1][1] + pts[3][1] - pts[0][1],
    )
    return math.dist(pts[2], completion)


def _as_points(quad) -> list[tuple[float, float]]:
    if len(quad) == 8:
        return [(quad[i], quad[i + 1]) for i in range(0, 8, 2)]
    return [(p[0], p[1]) for p in quad]
