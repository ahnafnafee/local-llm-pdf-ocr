"""Unit tests for `pdf_ocr.core.geometry` — the OrientedBox interchange
type and the quad math used by rotation-aware writers."""

from __future__ import annotations

import json
import math

import pytest

from pdf_ocr.core.geometry import (
    ANGLE_FLATTEN_DEG,
    OrientedBox,
    quad_affine_components,
    quad_angle_deg,
    quad_edge_lengths,
    quad_perspective_residual_px,
)


class TestOrientedBoxListCompat:
    """The box must be indistinguishable from a plain 4-float list for
    every existing consumer (DP aligner, writers, serialization)."""

    def test_unpacks_like_a_list(self):
        b = OrientedBox([0.1, 0.2, 0.6, 0.3])
        x0, y0, x1, y1 = b
        assert (x0, y0, x1, y1) == (0.1, 0.2, 0.6, 0.3)

    def test_indexing_and_len(self):
        b = OrientedBox([0.1, 0.2, 0.6, 0.3])
        assert b[2] - b[0] == pytest.approx(0.5)
        assert len(b) == 4

    def test_json_serializes_as_plain_array(self):
        b = OrientedBox([0.1, 0.2, 0.6, 0.3], quad=[0] * 8, angle=5.0)
        assert json.dumps(b) == "[0.1, 0.2, 0.6, 0.3]"

    def test_sorts_among_plain_lists(self):
        items = [[0.0, 0.5, 1.0, 0.6], OrientedBox([0.0, 0.1, 1.0, 0.2])]
        items.sort(key=lambda b: (b[1], b[0]))
        assert isinstance(items[0], OrientedBox)

    def test_defaults(self):
        b = OrientedBox([0, 0, 1, 1])
        assert b.quad is None
        assert b.angle == 0.0
        assert b.confidence is None
        assert not b.is_rotated


class TestIsRotated:
    def test_below_flatten_threshold_not_rotated(self):
        b = OrientedBox([0, 0, 1, 1], quad=[0] * 8, angle=ANGLE_FLATTEN_DEG - 0.1)
        assert not b.is_rotated

    def test_at_threshold_rotated(self):
        b = OrientedBox([0, 0, 1, 1], quad=[0] * 8, angle=ANGLE_FLATTEN_DEG)
        assert b.is_rotated

    def test_negative_angle_symmetric(self):
        b = OrientedBox([0, 0, 1, 1], quad=[0] * 8, angle=-ANGLE_FLATTEN_DEG)
        assert b.is_rotated

    def test_angle_without_quad_never_rotated(self):
        # Writers need the quad geometry to place rotated text; an angle
        # alone cannot be honored.
        b = OrientedBox([0, 0, 1, 1], quad=None, angle=45.0)
        assert not b.is_rotated


class TestQuadMath:
    def _quad_px(self, angle_deg: float, run: float = 400.0, h: float = 40.0):
        """Build a rotated-rect quad in pixel space, clockwise from TL."""
        a = math.radians(angle_deg)
        dx, dy = math.cos(a), math.sin(a)        # run direction
        px, py = -math.sin(a), math.cos(a)       # down-perpendicular
        p0 = (100.0, 200.0)
        p1 = (p0[0] + run * dx, p0[1] + run * dy)
        p3 = (p0[0] + h * px, p0[1] + h * py)
        p2 = (p1[0] + h * px, p1[1] + h * py)
        return [p0, p1, p2, p3]

    def test_horizontal_quad_angle_zero(self):
        assert quad_angle_deg(self._quad_px(0.0)) == pytest.approx(0.0)

    def test_positive_angle_round_trip(self):
        assert quad_angle_deg(self._quad_px(10.0)) == pytest.approx(10.0)

    def test_negative_angle_round_trip(self):
        assert quad_angle_deg(self._quad_px(-7.5)) == pytest.approx(-7.5)

    def test_accepts_flat_coordinate_list(self):
        pts = self._quad_px(10.0)
        flat = [c for p in pts for c in p]
        assert quad_angle_deg(flat) == pytest.approx(10.0)

    def test_degenerate_quad_angle_zero(self):
        assert quad_angle_deg([(5, 5)] * 4) == 0.0

    def test_edge_lengths_measured_in_pixel_space(self):
        # Normalized on a non-square page: naive normalized-space edge
        # lengths would distort; pixel-space lengths must round-trip.
        page_w, page_h = 800.0, 1000.0
        pts = self._quad_px(10.0, run=400.0, h=40.0)
        quad_norm = [c / (page_w if i % 2 == 0 else page_h)
                     for p in pts for i, c in enumerate(p)]
        run, height = quad_edge_lengths(quad_norm, page_w, page_h)
        assert run == pytest.approx(400.0, rel=1e-6)
        assert height == pytest.approx(40.0, rel=1e-6)


def _normalize(pts, page_w: float, page_h: float) -> list[float]:
    """Pixel corners → flat normalized quad on a (page_w × page_h) page."""
    return [c / (page_w if i % 2 == 0 else page_h)
            for p in pts for i, c in enumerate(p)]


class TestQuadAffineComponents:
    """The CSS matrix() columns must be the quad's edge unit vectors,
    recovered exactly through non-square-page normalization."""

    def _quad_px(self, angle_deg: float, run: float = 400.0, h: float = 40.0):
        a = math.radians(angle_deg)
        dx, dy = math.cos(a), math.sin(a)
        px, py = -math.sin(a), math.cos(a)
        p0 = (100.0, 200.0)
        p1 = (p0[0] + run * dx, p0[1] + run * dy)
        p3 = (p0[0] + h * px, p0[1] + h * py)
        p2 = (p1[0] + h * px, p1[1] + h * py)
        return [p0, p1, p2, p3]

    @pytest.mark.parametrize("page", [(800.0, 1000.0), (1600.0, 500.0)])
    def test_rotated_rect_equals_rotation_matrix(self, page):
        page_w, page_h = page
        quad = _normalize(self._quad_px(10.0), page_w, page_h)
        a, b, c, d = quad_affine_components(quad, page_w, page_h)
        ca, sa = math.cos(math.radians(10)), math.sin(math.radians(10))
        assert a == pytest.approx(ca, abs=1e-9)
        assert b == pytest.approx(sa, abs=1e-9)
        assert c == pytest.approx(-sa, abs=1e-9)
        assert d == pytest.approx(ca, abs=1e-9)

    def test_horizontal_rect_is_identity(self):
        quad = _normalize(self._quad_px(0.0), 800.0, 1000.0)
        assert quad_affine_components(quad, 800.0, 1000.0) == pytest.approx(
            (1.0, 0.0, 0.0, 1.0)
        )

    def test_sheared_parallelogram_columns_follow_edges(self):
        # Top edge along +x; left edge leaning (50, 100): the second
        # column must be that edge's unit vector, not a pure rotation.
        pts = [(100.0, 200.0), (300.0, 200.0), (350.0, 300.0), (150.0, 300.0)]
        quad = _normalize(pts, 800.0, 1000.0)
        a, b, c, d = quad_affine_components(quad, 800.0, 1000.0)
        n = math.hypot(50.0, 100.0)
        assert (a, b) == pytest.approx((1.0, 0.0))
        assert c == pytest.approx(50.0 / n)
        assert d == pytest.approx(100.0 / n)

    def test_degenerate_quad_falls_back_to_identity(self):
        quad = [0.5, 0.5] * 4
        assert quad_affine_components(quad, 800.0, 1000.0) == (1.0, 0.0, 0.0, 1.0)


class TestQuadPerspectiveResidual:
    def test_rotated_rect_residual_is_zero(self):
        a = math.radians(17.0)
        dx, dy = math.cos(a), math.sin(a)
        px, py = -math.sin(a), math.cos(a)
        p0 = (100.0, 200.0)
        pts = [
            p0,
            (p0[0] + 300 * dx, p0[1] + 300 * dy),
            (p0[0] + 300 * dx + 40 * px, p0[1] + 300 * dy + 40 * py),
            (p0[0] + 40 * px, p0[1] + 40 * py),
        ]
        quad = _normalize(pts, 800.0, 1000.0)
        assert quad_perspective_residual_px(quad, 800.0, 1000.0) == pytest.approx(
            0.0, abs=1e-9
        )

    @pytest.mark.parametrize("page", [(800.0, 1000.0), (1600.0, 500.0)])
    def test_keystone_residual_measured_in_pixels(self, page):
        # BR sits 10 px right of the parallelogram completion
        # (TR + BL - TL = (495, 255)); the residual must come back in
        # pixels regardless of how the page normalization distorts.
        page_w, page_h = page
        pts = [(100.0, 200.0), (500.0, 210.0), (505.0, 255.0), (95.0, 245.0)]
        quad = _normalize(pts, page_w, page_h)
        assert quad_perspective_residual_px(quad, page_w, page_h) == pytest.approx(
            10.0, abs=1e-9
        )
