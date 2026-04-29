"""
Shared fixtures.

- `examples_dir` / `example_pdfs` — on-disk sample documents under examples/.
- `surya_aligner` — a real HybridAligner instance shared across the session
  (Surya model load is ~5s, so we only want to pay it once).
- `stub_ocr` — an OCRProcessor replacement that returns canned text without
  hitting LM Studio, so tests can run offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Silence Surya's internal tqdm before any module loads it.
os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    d = ROOT / "examples"
    assert d.is_dir(), f"examples directory missing: {d}"
    return d


@pytest.fixture(scope="session")
def example_pdfs(examples_dir: Path) -> dict[str, Path]:
    names = ["digital.pdf", "hybrid.pdf", "handwritten.pdf"]
    paths = {n: examples_dir / n for n in names}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        pytest.skip(f"example PDFs not found: {missing}")
    return paths


@pytest.fixture(scope="session")
def surya_aligner():
    """Load HybridAligner once per session — Surya init is expensive."""
    from pdf_ocr.core.aligner import HybridAligner
    return HybridAligner()


class _StubOCR:
    """
    Drop-in replacement for OCRProcessor.

    Returns a fixed list of lines for full-page OCR and a fixed string for
    crops. Also records every call so tests can assert on behaviour.
    """

    def __init__(self, page_lines: list[str] | None = None, crop_text: str = "recovered"):
        self.page_lines = page_lines or [
            "Section heading",
            "First paragraph of body text with several words.",
            "Second paragraph with more content to align.",
            "Closing line.",
        ]
        self.crop_text = crop_text
        self.page_calls = 0
        self.crop_calls = 0

    async def perform_ocr(self, image_base64: str) -> list[str]:
        self.page_calls += 1
        return list(self.page_lines)

    async def perform_ocr_on_crop(self, image_base64: str) -> str:
        self.crop_calls += 1
        return self.crop_text


@pytest.fixture
def stub_ocr():
    return _StubOCR()


@pytest.fixture
def make_stub_ocr():
    """Factory fixture for tests that need a customised stub."""
    return _StubOCR
