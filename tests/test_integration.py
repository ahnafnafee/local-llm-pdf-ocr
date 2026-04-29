"""
Integration tests: real Surya detection + real PDF I/O + stubbed LLM.

These validate that the full pipeline executes end-to-end against the on-disk
example PDFs without requiring LM Studio. Surya load is amortized via the
session-scoped `surya_aligner` fixture.

Run just these with:   uv run pytest -m slow
Skip them during quick iteration:  uv run pytest -m "not slow"
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import fitz
import pytest

from pdf_ocr.core.ocr import OCRProcessor
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.pipeline import OCRPipeline

pytestmark = pytest.mark.slow


EXAMPLE_NAMES = ["digital.pdf", "hybrid.pdf", "handwritten.pdf"]


# --- detection sanity -------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_surya_detects_boxes_on_examples(
    surya_aligner, example_pdfs: dict[str, Path], name: str
):
    """Surya should find at least one text region in each sample PDF."""
    pdf_handler = PDFHandler()
    images = pdf_handler.convert_to_images(str(example_pdfs[name]))

    import base64
    image_bytes = [base64.b64decode(images[p]) for p in sorted(images)]
    batch = surya_aligner.get_detected_boxes_batch(image_bytes)

    total_boxes = sum(len(pg) for pg in batch)
    assert total_boxes > 0, f"{name}: expected at least one detected box"

    # Every returned box should be normalized and non-degenerate.
    for page_boxes in batch:
        for bbox in page_boxes:
            assert len(bbox) == 4
            nx0, ny0, nx1, ny1 = bbox
            assert 0.0 <= nx0 < nx1 <= 1.0, f"{name}: invalid x range {bbox}"
            assert 0.0 <= ny0 < ny1 <= 1.0, f"{name}: invalid y range {bbox}"


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_detected_boxes_are_in_reading_order(
    surya_aligner, example_pdfs: dict[str, Path], name: str
):
    """get_detected_boxes_batch returns boxes in stable row-major order
    (top-to-bottom, left-to-right). The DP itself is order-agnostic
    (auto-detects column-major vs row-major emission), but the public
    contract of the detection step is row-major as a deterministic
    default for visualization and downstream tools."""
    pdf_handler = PDFHandler()
    images = pdf_handler.convert_to_images(str(example_pdfs[name]))
    import base64
    image_bytes = [base64.b64decode(images[p]) for p in sorted(images)]
    batch = surya_aligner.get_detected_boxes_batch(image_bytes)

    for page_idx, page_boxes in enumerate(batch):
        prev_y, prev_x = -1.0, -1.0
        for bbox in page_boxes:
            y, x = bbox[1], bbox[0]
            assert (y > prev_y) or (y == prev_y and x >= prev_x), (
                f"{name} page {page_idx}: boxes out of row-major order at {bbox}"
            )
            prev_y, prev_x = y, x


# --- end-to-end against real PDFs (LLM stubbed) ----------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_end_to_end_pipeline_produces_searchable_pdf(
    surya_aligner, example_pdfs: dict[str, Path], tmp_path: Path, name: str
):
    """
    Full pipeline: convert → detect (real Surya) → align (real DP) → embed.
    LLM is stubbed to return a known marker; we assert the marker shows up in
    the output PDF's extractable text layer.
    """
    marker_lines = [
        f"ZZMARKER_{name}_HEADING",
        "Body paragraph one with enough content to place into a box.",
        f"ZZMARKER_{name}_MIDDLE",
        "Body paragraph two with a reasonable amount of text as well.",
        f"ZZMARKER_{name}_TAIL",
    ]

    class _Stub(OCRProcessor):
        def __init__(self):
            # Skip real init (no LLM client needed).
            pass

        async def perform_ocr(self, image_base64):
            return list(marker_lines)

        async def perform_ocr_on_crop(self, image_base64):
            return "ZZCROPMARKER"

    input_pdf = str(example_pdfs[name])
    output_pdf = str(tmp_path / f"out_{name}")

    pipe = OCRPipeline(
        aligner=surya_aligner,
        ocr_processor=_Stub(),
        pdf_handler=PDFHandler(),
    )
    asyncio.run(pipe.run(input_pdf, output_pdf, concurrency=2, refine=True))

    # Output PDF exists, has the same page count, and has *selectable* text.
    with fitz.open(input_pdf) as src, fitz.open(output_pdf) as out:
        assert len(out) == len(src), f"{name}: page count mismatch"
        full_text = " ".join(page.get_text("text") for page in out)

    # At least one marker must survive into the output text layer — proves the
    # DP alignment placed stubbed LLM content into the sandwich PDF.
    assert "ZZMARKER" in full_text or "ZZCROPMARKER" in full_text, (
        f"{name}: no markers recovered from output text layer; "
        f"got {full_text[:300]!r}"
    )


def test_end_to_end_with_page_filter(
    surya_aligner, example_pdfs: dict[str, Path], tmp_path: Path
):
    """--pages 1 should produce an output whose page 0 is searchable."""
    class _Stub(OCRProcessor):
        def __init__(self): pass
        async def perform_ocr(self, image_base64):
            return ["ZZONLYPAGEONE marker", "second line"]
        async def perform_ocr_on_crop(self, image_base64):
            return "ZZCROP"

    input_pdf = str(example_pdfs["digital.pdf"])
    output_pdf = str(tmp_path / "page1_only.pdf")

    pipe = OCRPipeline(surya_aligner, _Stub(), PDFHandler())
    asyncio.run(pipe.run(input_pdf, output_pdf, pages="1", refine=True))

    with fitz.open(output_pdf) as out:
        page0_text = out[0].get_text("text")

    assert "ZZONLYPAGEONE" in page0_text or "ZZCROP" in page0_text


# --- DP algorithmic properties on real Surya output -------------------------


def _get_page_boxes(
    surya_aligner,
    pdf_handler: PDFHandler,
    pdf_path: Path,
    page: int = 0,
    dpi: int = 200,
):
    """Run the real detection path and return boxes for a single page.

    NOTE: `dpi` must match the pipeline's dpi when used to validate pipeline
    output — Surya detection is slightly resolution-sensitive, so running at
    a different dpi yields differently-positioned boxes.
    """
    import base64
    images = pdf_handler.convert_to_images(str(pdf_path), dpi=dpi)
    image_bytes = [base64.b64decode(images[p]) for p in sorted(images)]
    all_boxes = surya_aligner.get_detected_boxes_batch(image_bytes)
    return all_boxes[page]


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_dp_conserves_all_lines_on_real_boxes(
    surya_aligner, example_pdfs: dict[str, Path], name: str
):
    """Every LLM line must be present somewhere in the output (no drops)."""
    boxes = _get_page_boxes(surya_aligner, PDFHandler(), example_pdfs[name])
    if len(boxes) == 0:
        pytest.skip(f"{name}: no boxes detected")

    lines = [
        "ALPHALINE first distinct line",
        "BETALINE second distinct line",
        "GAMMALINE third distinct line",
        "DELTALINE fourth distinct line",
    ]
    structured = [(b, "") for b in boxes]
    aligned = surya_aligner.align_text(structured, lines)

    joined = " ".join(t for _, t in aligned)
    for tag in ("ALPHALINE", "BETALINE", "GAMMALINE", "DELTALINE"):
        assert tag in joined, f"{name}: line tag {tag} was dropped by DP"


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_dp_placement_is_monotonic_on_real_boxes(
    surya_aligner, example_pdfs: dict[str, Path], name: str
):
    """For any two matched lines, the earlier-input one must land in an
    earlier-or-equal box position. This is the monotonicity guarantee that
    preserves reading order."""
    boxes = _get_page_boxes(surya_aligner, PDFHandler(), example_pdfs[name])
    if len(boxes) < 3:
        pytest.skip(f"{name}: need at least 3 boxes")

    # Use tagged lines so we can track original order into the output.
    lines = [f"TAG{i:02d}line with some body text here" for i in range(6)]
    structured = [(b, "") for b in boxes]
    aligned = surya_aligner.align_text(structured, lines)

    # For each tag, find the first box index it appears in.
    first_box = {}
    for box_idx, (_, text) in enumerate(aligned):
        for i, line in enumerate(lines):
            tag = f"TAG{i:02d}"
            if tag in text and i not in first_box:
                first_box[i] = box_idx

    placed = sorted(first_box.keys())
    for a, b in zip(placed, placed[1:]):
        assert first_box[a] <= first_box[b], (
            f"{name}: reading-order violation — TAG{a:02d} landed in box "
            f"{first_box[a]} but TAG{b:02d} in earlier box {first_box[b]}"
        )


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_embedded_text_positionally_matches_aligned_boxes(
    surya_aligner, example_pdfs: dict[str, Path], tmp_path: Path, name: str
):
    """
    End-to-end position correspondence on real Surya boxes.

    Run the full pipeline, then for every aligned (box, text) pair assert
    that `text` is extractable at the coordinates of `box` in the output PDF.
    This is the strongest confidence check: it verifies the alignment decision
    *and* the embedding geometry, against real detected layouts.
    """
    pdf_handler = PDFHandler()
    DPI = 200  # keep pipeline DPI and reconstruction DPI identical
    boxes = _get_page_boxes(surya_aligner, pdf_handler, example_pdfs[name], dpi=DPI)
    if len(boxes) < 3:
        pytest.skip(f"{name}: need at least 3 boxes")

    # Tagged, distinct lines — one per detectable box (padded with extras so
    # DP has freedom and some boxes may remain empty; we only check the ones
    # that got populated).
    n = min(len(boxes), 6)
    lines = [f"UNIQUE{i:02d}tag and some body content to place" for i in range(n)]

    class _Stub(OCRProcessor):
        def __init__(self): pass
        async def perform_ocr(self, image_base64):
            return list(lines)
        async def perform_ocr_on_crop(self, image_base64):
            return "REFINED"

    input_pdf = str(example_pdfs[name])
    output_pdf = str(tmp_path / f"posmatch_{name}")

    pipe = OCRPipeline(surya_aligner, _Stub(), pdf_handler)
    asyncio.run(pipe.run(input_pdf, output_pdf, concurrency=2, refine=False, dpi=DPI))

    # Inspect the output: for every box that got one of our UNIQUE tags, the
    # tag's word coordinates must fall inside the box.
    with fitz.open(output_pdf) as out:
        page = out[0]
        pw, ph = page.rect.width, page.rect.height
        words = page.get_text("words")

        # Re-run the alignment in-process to know which box got which line.
        structured = [(b, "") for b in boxes]
        aligned = surya_aligner.align_text(structured, lines)

        checked = 0
        for box, text in aligned:
            if not text:
                continue
            for i in range(n):
                tag = f"UNIQUE{i:02d}"
                if tag not in text:
                    continue
                box_rect = fitz.Rect(
                    box[0] * pw, box[1] * ph, box[2] * pw, box[3] * ph,
                )
                tag_words = [w for w in words if tag in w[4]]
                assert tag_words, f"{name}: {tag} placed in box but not found in output"

                # Physical question: if a user drags a selection rect over
                # this bbox, do they get the intended text? Require that
                # each of the tag's words overlaps the bbox by >= 50% of the
                # word's own area. That's strict enough to catch wrong-box
                # placement but tolerant of normal ascender overshoot
                # (~fontsize * 0.2pt above the bbox's top edge).
                for w in tag_words:
                    wr = fitz.Rect(w[0], w[1], w[2], w[3])
                    inter = wr & box_rect
                    if inter.is_empty:
                        overlap = 0.0
                    else:
                        overlap = inter.get_area() / max(1e-6, wr.get_area())
                    assert overlap >= 0.5, (
                        f"{name}: {tag} poorly placed — word at {list(wr)}, "
                        f"expected inside {list(box_rect)} (overlap={overlap:.2f})"
                    )
                checked += 1

        assert checked > 0, f"{name}: no tags landed in any box — DP produced empty alignment"


def test_hybrid_form_no_consecutive_duplicate_lines(
    surya_aligner, example_pdfs: dict[str, Path], tmp_path: Path
):
    """
    Regression for the bug where examples/hybrid.pdf produced an output
    PDF with two consecutive identical lines ("Name: Sally Walker DOB: ..."
    appeared twice; "HEALTH INTAKE FORM" appeared twice in another run).

    Root cause was a DP misalignment: the symmetric ``_match_cost`` let
    a long line slide into a smaller box just because the char counts
    happened to align, leaving the visually-correct wide box empty. The
    refine stage then cropped the empty box and re-OCR'd the same
    handwritten line, producing a second copy in the output text layer.

    The fix combines two changes:
      1. ``_match_cost`` is now asymmetric — overfill (line longer than
         box capacity) costs more than equivalent underfill, so the DP
         skips a too-narrow box rather than packing a long line into it.
      2. After refine, ``_drop_refined_duplicates`` clears any refined
         box whose text is a substring of (or equal to) text already in
         a vertically-nearby matched box.

    This test stubs the LLM with the exact lines OlmOCR-2 produces on
    examples/hybrid.pdf — including the joined two-line paragraph
    that triggers the misalignment — and asserts no two consecutive
    extracted lines are identical.
    """
    # The actual sequence OlmOCR-2 emits for examples/hybrid.pdf, with
    # the paragraph joined into one line (the trigger condition) and
    # missing both "FakeDoc M.D." (top-left, small) and the second
    # medication entry (faint handwriting). Refine recovers those.
    stub_lines = [
        "HEALTH INTAKE FORM",
        "Please fill out the questionnaire carefully. The information you provide will be used to complete your health profile and will be kept confidential.",
        "Date: 9/14/19",
        "Name: Sally Walker DOB: 09/04/1986",
        "Address: 24 Barney Lane City: Towaco State: NJ Zip: 07082",
        "Email: sally.walker@cmail.com Phone #: (906) 917-3486",
        "Gender: F Marital Status: Single Occupation: Software Engineer",
        "Referred By: NONE",
        "Emergency Contact: Eva Walker Emergency Contact Phone: (906) 334-8926",
        "Describe your medical concerns (symptoms, diagnoses, etc):",
        "Runny nose, mucas in throat, weakness, aches, chills, tired",
        "Are you currently taking any medication? (If yes, please describe):",
        "Vyvanse (25mg) daily for attention",
    ]

    class _Stub(OCRProcessor):
        def __init__(self):
            self._refine_counter = 0

        async def perform_ocr(self, image_base64):
            return list(stub_lines)

        async def perform_ocr_on_crop(self, image_base64):
            # Each refine call returns a unique token so two refined
            # boxes can't accidentally produce identical adjacent lines
            # in the test (which would be a false positive — the bug
            # under test is matched/refined collisions, not refined/
            # refined ones).
            self._refine_counter += 1
            return f"REFINED_{self._refine_counter:02d}"

    input_pdf = str(example_pdfs["hybrid.pdf"])
    output_pdf = str(tmp_path / "hybrid_no_dups.pdf")

    pipe = OCRPipeline(surya_aligner, _Stub(), PDFHandler())
    asyncio.run(pipe.run(input_pdf, output_pdf, concurrency=2, refine=True))

    with fitz.open(output_pdf) as out:
        text = out[0].get_text("text")

    extracted_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    consecutive_dups = [
        (i, line)
        for i, line in enumerate(extracted_lines[1:], start=1)
        if line == extracted_lines[i - 1] and len(line) > 5
    ]
    assert not consecutive_dups, (
        f"hybrid.pdf produced consecutive duplicate lines in OCR layer: "
        f"{consecutive_dups!r}"
    )


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_alignment_quality_metrics(
    surya_aligner, example_pdfs: dict[str, Path], name: str
):
    """
    Summarise quality numbers for the alignment on each example PDF.

    We don't have per-box ground truth, but we can assert sanity: most
    boxes either get text or are small enough to not matter, and no line
    is silently dropped.
    """
    pdf_handler = PDFHandler()
    boxes = _get_page_boxes(surya_aligner, pdf_handler, example_pdfs[name])
    if not boxes:
        pytest.skip(f"{name}: no boxes")

    # Realistic line count: ~ one line per box.
    lines = [f"content line number {i} with some words" for i in range(len(boxes))]
    structured = [(b, "") for b in boxes]
    aligned = surya_aligner.align_text(structured, lines)

    filled = sum(1 for _, t in aligned if t.strip())
    fill_rate = filled / len(boxes)

    # Conservation: every input line must appear in output.
    joined = " ".join(t for _, t in aligned)
    for i in range(len(lines)):
        assert f"number {i} " in joined + " ", (
            f"{name}: line {i} dropped; fill_rate={fill_rate:.2f}"
        )

    # Sanity floor: with one line per box and char-aware cost, at least half
    # the boxes should receive text directly (the rest may be absorbed via
    # the skip_line attach rule, which is fine).
    assert fill_rate >= 0.5, (
        f"{name}: unexpectedly low fill rate {fill_rate:.2f} "
        f"({filled}/{len(boxes)} boxes populated)"
    )
