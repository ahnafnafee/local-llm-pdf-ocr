"""Generate examples/photo.jpg — a realistic 'document photographed on
a desk' input for the photo-preprocessing path.

Derived from the repo's own examples/digital.pdf (license-clean, and
its ground-truth fixture doubles as the content reference): the page is
rendered, perspective-warped onto a dark desk, given a soft cast
shadow, an uneven lighting gradient, and sensor noise. The committed
example exercises rectify → OCR → inverse-map → embed-on-original.

Usage:
    uv run scripts/make_photo_example.py            # writes examples/photo.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PDF = ROOT / "examples" / "digital.pdf"
OUT = ROOT / "examples" / "photo.jpg"

CANVAS_W, CANVAS_H = 1100, 1400
# Page corners on the desk: tilted and perspective-foreshortened, with
# deviation far above the preprocessor's engage threshold.
DEST_QUAD = np.array(
    [[265, 170], [905, 235], [840, 1240], [165, 1135]], dtype=np.float32,
)
RNG_SEED = 20260609  # fixed so the committed JPEG is reproducible


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)

    doc = fitz.open(str(SRC_PDF))
    pix = doc[0].get_pixmap(dpi=150)
    page = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n,
    )[:, :, :3][:, :, ::-1].copy()  # RGB -> BGR
    doc.close()

    # Wood-toned desk with mild texture.
    desk = np.full((CANVAS_H, CANVAS_W, 3), (38, 52, 74), dtype=np.uint8)
    texture = rng.normal(0, 6, size=(CANVAS_H, CANVAS_W, 1))
    desk = np.clip(desk.astype(np.int16) + texture.astype(np.int16), 0, 255).astype(np.uint8)

    # Cast shadow: the page quad dilated and shifted, heavily blurred.
    shadow = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    cv2.fillConvexPoly(shadow, (DEST_QUAD + [18, 26]).astype(np.int32), 90)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 25)
    desk = np.clip(desk.astype(np.int16) - shadow[..., None], 0, 255).astype(np.uint8)

    # Warp the page onto the desk.
    src_quad = np.array(
        [[0, 0], [page.shape[1] - 1, 0],
         [page.shape[1] - 1, page.shape[0] - 1], [0, page.shape[0] - 1]],
        dtype=np.float32,
    )
    h_fwd = cv2.getPerspectiveTransform(src_quad, DEST_QUAD)
    warped = cv2.warpPerspective(page, h_fwd, (CANVAS_W, CANVAS_H))
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, DEST_QUAD.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    m = (mask[..., None].astype(np.float32)) / 255.0
    composed = (warped.astype(np.float32) * m + desk.astype(np.float32) * (1 - m))

    # Uneven lighting: bright upper-left falling off to the lower-right,
    # the typical single-lamp capture.
    yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W].astype(np.float32)
    light = 1.08 - 0.28 * ((xx / CANVAS_W) * 0.6 + (yy / CANVAS_H) * 0.4)
    composed = np.clip(composed * light[..., None], 0, 255)

    # Sensor noise.
    composed = np.clip(
        composed + rng.normal(0, 2.2, size=composed.shape), 0, 255,
    ).astype(np.uint8)

    cv2.imwrite(str(OUT), composed, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
