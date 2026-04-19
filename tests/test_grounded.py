"""
Tests for the grounded-OCR pipeline path (no Surya / no DP / no refine).

Validates:
    - Z.AI hosted response parser against a real captured fixture.
    - layout_details parser (GLM-OCR via vLLM / self-hosted).
    - OCRPipeline routes to the grounded path when a backend is supplied.
    - Grounded pipeline produces a searchable PDF whose extracted text is
      recoverable at the expected positions.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fitz
import pytest

from src.pdf_ocr.core.grounded import (
    GroundedBlock,
    GroundedResponse,
    _parse_grounded_json,
    parse_glm_layout_details,
    parse_zai_response,
)
from src.pdf_ocr.core.pdf import PDFHandler
from src.pdf_ocr.pipeline import OCRPipeline


FIXTURES = Path(__file__).parent / "fixtures"


# --- parsers ----------------------------------------------------------------


class TestParseZAIResponse:
    @pytest.fixture
    def fixture(self) -> dict:
        # Both tests/fixtures/zai_handwritten.json and
        # tests/fixtures/ground_truth_handwritten.json carry the same Z.AI
        # layout — we use the ground-truth copy as the single source.
        with (FIXTURES / "ground_truth_handwritten.json").open() as f:
            return json.load(f)

    def test_parses_captured_response(self, fixture):
        response = parse_zai_response(fixture)
        assert isinstance(response, GroundedResponse)
        assert response.page_sizes == [(1654, 2170)]
        # 15 layout entries, 1 `list_marker` skipped → 14 content blocks.
        assert len(response.blocks) == 14

    def test_coordinates_are_normalized(self, fixture):
        response = parse_zai_response(fixture)
        for b in response.blocks:
            assert 0.0 <= b.bbox[0] < b.bbox[2] <= 1.0
            assert 0.0 <= b.bbox[1] < b.bbox[3] <= 1.0

    def test_non_content_labels_filtered_out(self, fixture):
        response = parse_zai_response(fixture)
        labels = {b.label for b in response.blocks}
        assert not labels & {"image", "empty_line", "signature_line", "list_marker"}

    def test_accepts_inner_data_directly(self, fixture):
        # Passing the unwrapped `data` object should also work.
        inner_only = fixture["data"]
        response = parse_zai_response(inner_only)
        assert len(response.blocks) == 14

    def test_known_text_survives_parsing(self, fixture):
        response = parse_zai_response(fixture)
        joined = " | ".join(b.text for b in response.blocks)
        assert "computational procedure" in joined
        assert "CORRECT ALGORITHM" in joined
        assert "induction proofs" in joined

    def test_page_index_preserved(self, fixture):
        response = parse_zai_response(fixture)
        assert all(b.page_index == 0 for b in response.blocks)

    def test_empty_content_filtered(self, fixture):
        # Inject an empty block and confirm it's dropped.
        f = dict(fixture)
        f["data"] = dict(f["data"])
        f["data"]["layout"] = list(f["data"]["layout"]) + [{
            "block_content": "   ", "bbox": [0, 0, 100, 100],
            "block_id": 999, "page_index": 0, "block_label": "text", "score": 0,
        }]
        response = parse_zai_response(f)
        # Still 14 content blocks (whitespace-only block discarded).
        assert len(response.blocks) == 14


class TestParseGLMLayoutDetails:
    def test_parses_flat_list(self):
        payload = {
            "data_info": {"pages": [{"width": 1000, "height": 2000}]},
            "layout_details": [
                {"label": "text", "content": "Hello world", "bbox_2d": [100, 200, 500, 260]},
                {"label": "image", "content": "...", "bbox_2d": [0, 0, 100, 100]},
            ],
        }
        response = parse_glm_layout_details(payload)
        assert len(response.blocks) == 1
        assert response.blocks[0].text == "Hello world"
        assert response.blocks[0].bbox == [0.1, 0.1, 0.5, 0.13]

    def test_parses_nested_per_page_list(self):
        payload = {
            "data_info": {"pages": [{"width": 1000, "height": 2000}]},
            "layout_details": [
                [{"label": "text", "content": "On page 0", "bbox_2d": [0, 0, 500, 100]}],
            ],
        }
        response = parse_glm_layout_details(payload, page_index=0)
        assert response.blocks[0].text == "On page 0"

    def test_accepts_json_string(self):
        payload = {
            "data_info": {"pages": [{"width": 100, "height": 100}]},
            "layout_details": [
                {"label": "text", "content": "x", "bbox_2d": [0, 0, 50, 50]},
            ],
        }
        response = parse_glm_layout_details(json.dumps(payload))
        assert len(response.blocks) == 1


# --- pipeline routing -------------------------------------------------------


class _StubGroundedBackend:
    """Canned backend — returns fixed blocks, records invocation."""

    def __init__(self, blocks: list[GroundedBlock], page_sizes):
        self.response = GroundedResponse(blocks=blocks, page_sizes=page_sizes)
        self.called_with: list[str] = []

    async def ocr_document(self, pdf_path: str) -> GroundedResponse:
        self.called_with.append(pdf_path)
        return self.response


def test_pipeline_routes_to_grounded_when_backend_provided(
    example_pdfs: dict[str, Path], tmp_path: Path
):
    """Grounded path skips Surya entirely — no aligner or ocr_processor needed."""
    marker_blocks = [
        GroundedBlock(bbox=[0.10, 0.10, 0.60, 0.14], text="GROUNDED_ALPHA",  page_index=0),
        GroundedBlock(bbox=[0.10, 0.30, 0.60, 0.34], text="GROUNDED_BETA",   page_index=0),
        GroundedBlock(bbox=[0.10, 0.60, 0.60, 0.64], text="GROUNDED_GAMMA",  page_index=0),
    ]
    backend = _StubGroundedBackend(marker_blocks, page_sizes=[(1000, 1300)])

    input_pdf = str(example_pdfs["digital.pdf"])
    output_pdf = str(tmp_path / "grounded_out.pdf")

    pipe = OCRPipeline(pdf_handler=PDFHandler(), grounded_backend=backend)
    pages_text = asyncio.run(pipe.run(input_pdf, output_pdf))

    # Backend was called with the input path.
    assert backend.called_with == [input_pdf]
    # All three markers ended up in the output's searchable layer.
    assert pages_text[0] == ["GROUNDED_ALPHA", "GROUNDED_BETA", "GROUNDED_GAMMA"]
    with fitz.open(output_pdf) as doc:
        text = doc[0].get_text("text")
    assert "GROUNDED_ALPHA" in text
    assert "GROUNDED_BETA" in text
    assert "GROUNDED_GAMMA" in text


def test_grounded_path_preserves_bbox_position(
    example_pdfs: dict[str, Path], tmp_path: Path
):
    """Text emitted by the grounded backend must land *inside* its bbox —
    this is the same positional-correspondence guarantee as the hybrid path."""
    block = GroundedBlock(
        bbox=[0.20, 0.30, 0.60, 0.34],
        text="POSMARKER_ZETA",
        page_index=0,
    )
    backend = _StubGroundedBackend([block], page_sizes=[(1000, 1300)])

    input_pdf = str(example_pdfs["digital.pdf"])
    output_pdf = str(tmp_path / "grounded_pos.pdf")

    pipe = OCRPipeline(pdf_handler=PDFHandler(), grounded_backend=backend)
    asyncio.run(pipe.run(input_pdf, output_pdf))

    with fitz.open(output_pdf) as doc:
        page = doc[0]
        pw, ph = page.rect.width, page.rect.height
        words = page.get_text("words")

    expected_rect = fitz.Rect(0.20 * pw, 0.30 * ph, 0.60 * pw, 0.34 * ph)
    hits = [w for w in words if "POSMARKER_ZETA" in w[4]]
    assert hits, "marker missing from grounded output"
    for w in hits:
        wr = fitz.Rect(w[0], w[1], w[2], w[3])
        inter = wr & expected_rect
        assert not inter.is_empty, f"grounded marker at {list(wr)} outside {list(expected_rect)}"
        overlap = inter.get_area() / max(1e-6, wr.get_area())
        assert overlap >= 0.5, f"overlap too low: {overlap:.2f}"


def test_pipeline_rejects_construction_without_pdf_handler():
    with pytest.raises(ValueError, match="pdf_handler is required"):
        OCRPipeline(grounded_backend=_StubGroundedBackend([], [(100, 100)]))


# --- prompted grounded JSON parser (Qwen2.5-VL / Qwen3-VL response shapes) --


class TestPromptedGroundedParser:
    def test_bare_json_array_qwen3_vl_style(self):
        # Qwen3-VL returns a bare JSON array with no fence wrapper.
        raw = (
            '[{"bbox_2d": [40, 38, 175, 96], "content": "Algorithms"}, '
            '{"bbox_2d": [50, 107, 487, 153], "content": "- computational procedure"}]'
        )
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=800, img_h=1000)
        assert len(blocks) == 2
        assert blocks[0].text == "Algorithms"
        # Coordinates should be normalized.
        assert blocks[0].bbox == [40 / 800, 38 / 1000, 175 / 800, 96 / 1000]

    def test_fenced_json_qwen25_vl_style(self):
        # Qwen2.5-VL wraps in ```json ... ```.
        raw = (
            '```json\n'
            '[\n'
            '    {"bbox_2d": [40, 38, 175, 96], "content": "Algorithms,"},\n'
            '    {"bbox_2d": [68, 117, 226, 150], "content": "- computational"}\n'
            ']\n'
            '```'
        )
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=800, img_h=1000)
        assert len(blocks) == 2
        assert blocks[0].text == "Algorithms,"

    def test_object_wrapping_array(self):
        # Some models wrap the array in {"results": [...]}.
        raw = '{"results": [{"bbox_2d":[0,0,100,50],"content":"hi"}]}'
        blocks = _parse_grounded_json(raw, page_idx=2, img_w=100, img_h=100)
        assert len(blocks) == 1
        assert blocks[0].page_index == 2
        assert blocks[0].text == "hi"

    def test_invalid_bbox_filtered(self):
        raw = (
            '[{"bbox_2d":[0,0,100,50],"content":"keep"},'
            '{"bbox_2d":[100,0,0,50],"content":"drop-x-reversed"},'
            '{"bbox_2d":[0,100,100,50],"content":"drop-y-reversed"},'
            '{"content":"drop-missing-bbox"},'
            '{"bbox_2d":[0,0,10],"content":"drop-wrong-length"}]'
        )
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=100, img_h=100)
        assert [b.text for b in blocks] == ["keep"]

    def test_empty_content_filtered(self):
        raw = '[{"bbox_2d":[0,0,10,10],"content":"  "},{"bbox_2d":[0,10,10,20],"content":"real"}]'
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=10, img_h=20)
        assert [b.text for b in blocks] == ["real"]

    def test_empty_input_returns_empty(self):
        assert _parse_grounded_json("", 0, 100, 100) == []

    def test_garbage_input_returns_empty(self):
        assert _parse_grounded_json("not json at all", 0, 100, 100) == []

    def test_alternate_field_names(self):
        # Accept `bbox` + `text` as aliases.
        raw = '[{"bbox":[0,0,50,50],"text":"alt-named"}]'
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=100, img_h=100)
        assert blocks[0].text == "alt-named"

    def test_preamble_prose_tolerated(self):
        raw = 'Here is the result:\n[{"bbox_2d":[0,0,10,10],"content":"x"}]'
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=10, img_h=10)
        assert len(blocks) == 1
