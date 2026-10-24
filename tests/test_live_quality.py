"""Opt-in recognition regressions using actual models and source documents.

Set OCR_LIVE_API_BASE and OCR_LIVE_MODE (hybrid or grounded), then run
pytest -m live_llm tests/test_live_quality.py. OCR_LIVE_MODEL can override
the normal mode default. These tests never run on an implicit endpoint.
"""

import os
import re
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


@pytest.mark.parametrize('mode', ['hybrid', 'grounded'])
@pytest.mark.parametrize('document,pages', [('dense', '2-3'), ('notes', '6,13,17,21')])
async def test_live_complex_regions_remain_searchable_at_source_positions(mode, document, pages, tmp_path):
    """Exercise later pages, diagram fields, embedded photos and math export.

    These assertions were checked against source pixels, independently of
    model outputs. The former whole-page implementation omits the diagram and
    photo text and displaces the dense-page lines.
    """
    endpoint = os.environ.get('OCR_LIVE_API_BASE')
    if not endpoint or os.environ.get('OCR_LIVE_MODE') != mode:
        pytest.skip('Set OCR_LIVE_API_BASE and OCR_LIVE_MODE to opt into real inference')
    model = os.environ.get('OCR_LIVE_MODEL') or (
        DEFAULT_MODEL if mode == 'hybrid' else DEFAULT_GROUNDED_MODEL)
    max_dim = int(os.environ.get('OCR_LIVE_MAX_IMAGE_DIM', '2048'))
    component = (OCRProcessor if mode == 'hybrid' else PromptedGroundedOCR)(
        api_base=endpoint, model=model, api_key=os.environ.get('OCR_LIVE_API_KEY'),
        **({'max_image_dim': max_dim} if mode == 'grounded' else {}))
    component.client.generation_options['seed'] = 0
    output = tmp_path / f'{document}-{mode}.pdf'
    pipe = OCRPipeline(
        aligner=HybridAligner() if mode == 'hybrid' else None,
        ocr_processor=component if mode == 'hybrid' else None,
        grounded_backend=component if mode == 'grounded' else None,
        pdf_handler=PDFHandler(),
    )
    try:
        await pipe.run(str(ROOT / f'examples/{document}.pdf'), str(output), pages=pages,
                       concurrency=1, max_image_dim=max_dim)
    finally:
        await component.aclose()

    def in_region(page, word, region):
        x0, y0, x1, y1 = region
        return [r for r in page.search_for(word)
                if x0 <= (r.x0+r.x1)/2/page.rect.width <= x1
                and y0 <= (r.y0+r.y1)/2/page.rect.height <= y1]

    with fitz.open(output) as pdf:
        if document == 'dense':
            assert in_region(pdf[1], 'Snowball', [.15,.06,.33,.13])
            assert in_region(pdf[1], 'reachable', [.07,.18,.32,.26])
            assert in_region(pdf[2], 'BCNF', [.02,.28,.32,.34])
            assert in_region(pdf[2], 'candidate keys', [.70,.01,.98,.14])
        else:
            words = set(re.findall(r'\w+', pdf[5].get_text().casefold()))
            printed = ('EMPLOYEE FNAME MINIT LNAME SSN BDATE ADDRESS SEX SALARY SUPERSSN DNO '
                       'DEPARTMENT DNAME DNUMBER MGRSSN MGRSTARTDATE DEPT_LOCATIONS DLOCATION '
                       'PROJECT PNAME PNUMBER PLOCATION DNUM WORKS_ON ESSN PNO HOURS '
                       'DEPENDENT DEPENDENT_NAME RELATIONSHIP')
            assert set(printed.casefold().split()) <= words
            for word, region in [
                ('FNAME', [.15,.10,.24,.145]), ('MINIT', [.22,.10,.30,.145]),
                ('LNAME', [.28,.10,.36,.145]), ('DLOCATION', [.42,.22,.55,.27]),
                ('SEX', [.52,.39,.56,.43]),
                ('Name', [.32,.49,.49,.57]), ('Amount', [.31,.57,.48,.64]),
            ]:
                assert in_region(pdf[5], word, region), (word, pdf[5].get_text())
            assert len(pdf[5].search_for('SEX')) == 2
            assert len(pdf[12].get_text()) > 500
            assert in_region(pdf[12], 'Customer', [.13,.13,.5,.18])
            assert in_region(pdf[12], 'Fairfax', [.17,.49,.85,.56])
            for word in ['Sailors', 'Reserves', 'Boats']:
                assert in_region(pdf[16], word, [.17,.20,.85,.43]), word
            assert len(in_region(pdf[16], 'Sailors', [.17,.20,.85,.43])) >= 4
            assert pdf[20].get_text().strip() == 'Project 1-3'
