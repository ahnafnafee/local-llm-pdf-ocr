"""Unit tests for `pdf_ocr.core.geometry` — the OrientedBox interchange
type and the quad math used by rotation-aware writers."""

from __future__ import annotations

import json
import math

import pytest

from pdf_ocr.core.geometry import (
    ANGLE_FLATTEN_DEG,
    OrientedBox,
    quad_angle_deg,
    quad_edge_lengths,
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
