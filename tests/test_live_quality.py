"""Opt-in recognition regressions using actual models and source documents.

Set OCR_LIVE_API_BASE and OCR_LIVE_MODE (hybrid or grounded), then run
pytest -m live_llm tests/test_live_quality.py. OCR_LIVE_MODEL can override
the normal mode default. These tests never run on an implicit endpoint.
"""

import os
from pathlib import Path

import pytest
import fitz

from pdf_ocr import HybridAligner, OCRPipeline, OCRProcessor, PDFHandler, PromptedGroundedOCR
from pdf_ocr.evaluation import load_doc_checks, run_doc_checks
from pdf_ocr.output import resolve_output_writer
from pdf_ocr.inference import DEFAULT_MODEL, DEFAULT_GROUNDED_MODEL

pytestmark = pytest.mark.live_llm
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,document", [
    ("hybrid", "notes"), ("hybrid", "hybrid"), ("hybrid", "handwritten"), ("grounded", "hybrid"),
    ("grounded", "notes"), ("grounded", "handwritten"),
    ("hybrid", "dense"), ("grounded", "dense"),
])
@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_live_recognition_is_stable(mode, document, seed, tmp_path):
    endpoint = os.environ.get("OCR_LIVE_API_BASE")
    if not endpoint or os.environ.get("OCR_LIVE_MODE") != mode:
        pytest.skip("Set OCR_LIVE_API_BASE and OCR_LIVE_MODE to opt into real inference")
    model = os.environ.get("OCR_LIVE_MODEL") or (
        DEFAULT_MODEL if mode == "hybrid" else DEFAULT_GROUNDED_MODEL
    )
    max_dim = int(os.environ.get("OCR_LIVE_MAX_IMAGE_DIM", "2048"))
    component = (OCRProcessor if mode == "hybrid" else PromptedGroundedOCR)(
        api_base=endpoint, model=model, api_key=os.environ.get("OCR_LIVE_API_KEY"),
        **({"max_image_dim": max_dim} if mode == "grounded" else {}),
    )
    component.client.generation_options["seed"] = seed
    handler = PDFHandler()
    output = tmp_path / ("recognized.pdf" if document == "hybrid" else "recognized.txt")
    pipe = OCRPipeline(
        aligner=HybridAligner() if mode == "hybrid" else None,
        ocr_processor=component if mode == "hybrid" else None,
        grounded_backend=component if mode == "grounded" else None,
        pdf_handler=handler,
        output_writer=resolve_output_writer(str(output)),
    )
    try:
        await pipe.run(str(ROOT / f"examples/{document}.pdf"), str(output), pages="1",
                       concurrency=1, max_image_dim=max_dim)
    finally:
        await component.aclose()
    if output.suffix == ".pdf":
        with fitz.open(output) as pdf:
            text = "\n".join(page.get_text() for page in pdf)
            if mode == "hybrid":
                # Source form: question at 76% page height, answer at 80%.
                # The former DP shift put this question over the symptoms
                # and the medication answer over its own question.
                question = pdf[0].search_for("Are you currently")
                answer = pdf[0].search_for("attention")
                assert question and all(0.74 < r.y0 / pdf[0].rect.height < 0.79 for r in question)
                assert answer and all(0.78 < r.y0 / pdf[0].rect.height < 0.85 for r in answer)
    else:
        text = output.read_text(encoding="utf-8")
    checks = run_doc_checks(text,
                            load_doc_checks(ROOT / f"evals/checks/{document}.jsonl"))
    assert checks[0] == checks[1], checks[2]
