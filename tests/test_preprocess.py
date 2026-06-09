"""Photo-rectification preprocessing: quad detection, homography
round-trips, and the pipeline-facing box mapping back onto originals."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import pytest

from pdf_ocr.core.geometry import OrientedBox
from pdf_ocr.utils.preprocess import (
    PageRectification,
    _order_corners,
    find_page_quad,
    map_points_to_original,
    map_structured_to_original,
    perspective_deviation,
    rectify_page,
    rectify_page_images,
    should_rectify,
)

# A photographed page: white quad, visibly perspective-distorted, on a
# dark desk. Corners chosen so deviation is far above the engage gate.
PHOTO_W, PHOTO_H = 800, 1000
PAGE_QUAD = np.array(
    [[180, 120], [660, 180], [620, 870], [120, 800]], dtype=np.float32,
)


def _synth_photo() -> np.ndarray:
    img = np.full((PHOTO_H, PHOTO_W, 3), 60, dtype=np.uint8)  # dark desk
    cv2.fillConvexPoly(img, PAGE_QUAD.astype(np.int32), (245, 245, 245))
    # Some "text" so illumination flattening has structure to keep.
    for y in range(220, 700, 60):
        cv2.line(img, (240, y), (560, y + 20), (40, 40, 40), 6)
    return img


def _b64_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class TestQuadDetection:
    def test_finds_the_page_on_a_photo(self):
        quad = find_page_quad(_synth_photo())
        assert quad is not None
        # Corner-by-corner within a few pixels of ground truth.
        assert np.abs(quad - PAGE_QUAD).max() < 12

    def test_flat_scan_finds_nothing(self):
        # A white page filling the frame has no contrasting border —
        # the no-op path for every existing scanned input.
        img = np.full((600, 500, 3), 250, dtype=np.uint8)
        assert find_page_quad(img) is None

    def test_small_objects_ignored(self):
        img = np.full((600, 500, 3), 60, dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (120, 120), (250, 250, 250), -1)
        assert find_page_quad(img) is None  # 2.6% of frame — not a page

    def test_corner_ordering(self):
        shuffled = np.array(
            [[620, 870], [180, 120], [120, 800], [660, 180]], dtype=np.float32,
        )
        ordered = _order_corners(shuffled)
        assert np.array_equal(ordered, PAGE_QUAD)


class TestEngagement:
    def test_tilted_quad_engages(self):
        assert should_rectify(PAGE_QUAD)

    def test_axis_aligned_quad_does_not_engage(self):
        rect = np.array(
            [[100, 100], [700, 100], [700, 900], [100, 900]], dtype=np.float32,
        )
        assert perspective_deviation(rect) == pytest.approx(0.0)
        assert not should_rectify(rect)

    def test_none_never_engages(self):
        assert not should_rectify(None)


class TestHomographyRoundTrip:
    def test_rectified_corners_map_back_to_page_corners(self):
        rect = rectify_page(_synth_photo(), PAGE_QUAD)
        rw, rh = rect.rectified_size
        corners = np.array(
            [[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]],
            dtype=np.float64,
        )
        mapped = map_points_to_original(corners, rect)
        assert np.abs(mapped - PAGE_QUAD).max() < 2.0

    def test_rectified_size_follows_page_aspect(self):
        rect = rectify_page(_synth_photo(), PAGE_QUAD)
        rw, rh = rect.rectified_size
        assert rh > rw  # the synthetic page is portrait


class TestPipelineFacingHelpers:
    def test_rectify_page_images_replaces_only_photo_pages(self):
        images = {0: _b64_jpeg(_synth_photo()),
                  1: _b64_jpeg(np.full((400, 300, 3), 250, dtype=np.uint8))}
        out, rect_maps = rectify_page_images(images, [0, 1], "auto")
        assert set(rect_maps) == {0}
        assert out[0] != images[0]      # photo page replaced
        assert out[1] == images[1]      # flat scan untouched

    def test_never_seen_pages_pass_through(self):
        items = {0: [([0.1, 0.1, 0.5, 0.2], "hello")]}
        assert map_structured_to_original(items, {}) == items

    def test_boxes_map_into_the_original_page_region(self):
        rect = rectify_page(_synth_photo(), PAGE_QUAD)
        # A line across the top of the RECTIFIED page.
        box = OrientedBox([0.1, 0.05, 0.9, 0.10], confidence=0.7)
        mapped = map_structured_to_original({0: [(box, "title line")]},
                                            {0: rect})
        (mbox, text), = mapped[0]
        assert text == "title line"
        assert isinstance(mbox, OrientedBox)
        assert mbox.quad is not None
        assert mbox.confidence == 0.7
        # Every mapped corner sits inside the photographed page outline.
        pts = np.array(
            [[mbox.quad[i] * PHOTO_W, mbox.quad[i + 1] * PHOTO_H]
             for i in range(0, 8, 2)],
            dtype=np.float32,
        )
        hull = cv2.convexHull(PAGE_QUAD)
        for pt in pts:
            assert cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), False) >= 0
        # The page tilts, so the mapped line must carry an angle.
        assert abs(mbox.angle) > 1.0


@pytest.mark.asyncio
async def test_pipeline_embeds_on_original_coordinates(stub_ocr):
    """End-to-end through OCRPipeline with preprocess=always: the writer
    must receive boxes mapped back into original-photo space."""
    from pdf_ocr.pipeline import OCRPipeline

    photo_b64 = _b64_jpeg(_synth_photo())

    class _PhotoPDF:
        def convert_to_images(self, path, dpi=150, max_image_dim=1024):
            return {0: photo_b64}

        def embed_structured_text(self, inp, out, pages, dpi):
            self.last_pages = dict(pages)

    class _RectSpaceAligner:
        """Claims one full-width line near the rectified page's top."""

        def get_detected_boxes_batch(self, images):
            return [[[0.1, 0.05, 0.9, 0.10]] for _ in images]

        def align_text(self, structured, lines):
            return [(box, "mapped line") for box, _ in structured]

    pdf = _PhotoPDF()
    pipe = OCRPipeline(_RectSpaceAligner(), stub_ocr, pdf)
    await pipe.run("photo.jpg", "out.pdf", refine=False, preprocess="always")

    (box, text), = pdf.last_pages[0]
    assert text == "mapped line"
    # In rectified space the line starts at x=0.1 of the PAGE; on the
    # original photo the page occupies roughly x in [0.15, 0.83], so the
    # mapped box must start well inside that, not at 0.1 of the frame.
    assert box[0] > 0.15
    assert getattr(box, "quad", None) is not None
