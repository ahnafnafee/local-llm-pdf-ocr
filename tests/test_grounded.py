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
from unittest.mock import AsyncMock, Mock

import fitz
import pytest

from types import SimpleNamespace

from pdf_ocr.core.grounded import (
    GroundedBlock,
    GroundedResponse,
    PromptedGroundedOCR,
    _parse_grounded_json,
    parse_glm_layout_details,
    parse_zai_response,
)
from pdf_ocr.core.ocr import LLMCallError, ModelNotLoadedError
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.pipeline import OCRPipeline


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
    """Canned backend — returns fixed blocks, records invocation + progress."""

    def __init__(self, blocks: list[GroundedBlock], page_sizes):
        self.response = GroundedResponse(blocks=blocks, page_sizes=page_sizes)
        self.called_with: list[str] = []
        self.progress_calls: list[tuple] = []

    async def ocr_document(self, pdf_path: str, progress=None) -> GroundedResponse:
        self.called_with.append(pdf_path)
        if progress is not None:
            # Mimic a real backend's per-page emission so pipeline-level
            # progress adapters get something meaningful.
            n = len({b.page_index for b in self.response.blocks}) or 1
            await progress("ocr", 0, n, f"Stub grounded OCR (0/{n})...")
            await progress("ocr", n, n, f"Stub grounded OCR ({n}/{n})")
        return self.response


@pytest.mark.parametrize("page_sizes", [[], [(1000, 1300)]])
def test_pipeline_routes_to_grounded_when_backend_provided(
    example_pdfs: dict[str, Path], tmp_path: Path, page_sizes,
):
    """Grounded path skips Surya entirely — no aligner or ocr_processor needed."""
    marker_blocks = [
        GroundedBlock(bbox=[0.10, 0.10, 0.60, 0.14], text="GROUNDED_ALPHA",  page_index=0),
        GroundedBlock(bbox=[0.10, 0.30, 0.60, 0.34], text="GROUNDED_BETA",   page_index=0),
        GroundedBlock(bbox=[0.10, 0.60, 0.60, 0.64], text="GROUNDED_GAMMA",  page_index=0),
    ]
    backend = _StubGroundedBackend(marker_blocks, page_sizes=page_sizes)

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


def test_pipeline_rejects_hybrid_run_without_aligner_or_ocr(
    example_pdfs: dict[str, Path], tmp_path: Path
):
    """Hybrid path needs both `aligner` and `ocr_processor`. If a user forgets
    to pass either (and doesn't supply a grounded backend), `run()` should
    raise an explicit ValueError — not a later AttributeError on None."""
    pipe = OCRPipeline(pdf_handler=PDFHandler())  # no aligner, no ocr
    with pytest.raises(ValueError, match="Hybrid pipeline requires"):
        asyncio.run(pipe.run(str(example_pdfs["digital.pdf"]), str(tmp_path / "x.pdf")))


def test_grounded_path_forwards_progress_callback(
    example_pdfs: dict[str, Path], tmp_path: Path
):
    """The pipeline should forward its progress callback into the grounded
    backend so users see per-page ticks instead of a 0→100 jump."""
    block = GroundedBlock(bbox=[0.1, 0.1, 0.5, 0.15], text="X", page_index=0)
    backend = _StubGroundedBackend([block], page_sizes=[(1000, 1300)])

    stages: list[str] = []

    async def cb(stage, cur, tot, msg):
        stages.append(stage)

    pipe = OCRPipeline(pdf_handler=PDFHandler(), grounded_backend=backend)
    asyncio.run(pipe.run(
        str(example_pdfs["digital.pdf"]), str(tmp_path / "out.pdf"),
        progress=cb,
    ))

    # Backend should have emitted "ocr" stage ticks via the forwarded callback,
    # and the pipeline should still emit "embed" for the output-writing phase.
    assert "ocr" in stages
    assert "embed" in stages


@pytest.mark.parametrize(
    ("page_sizes", "populated_pages", "empty_pages"),
    [
        ([], [], "the document"),
        ([(100, 100)], [], "page(s) 1"),
        ([(100, 100)] * 3, [], "page(s) 1, 2, 3"),
        ([(100, 100)] * 3, [1, 2], "page(s) 1"),
        ([(100, 100)] * 3, [0, 2], "page(s) 2"),
        ([(100, 100)] * 3, [0, 1], "page(s) 3"),
    ],
)
async def test_grounded_rejects_empty_pages_before_writing(
    tmp_path, page_sizes, populated_pages, empty_pages,
):
    blocks = [
        GroundedBlock([0.1, 0.1, 0.5, 0.2], "recognized text", p)
        for p in populated_pages
    ]
    writer = Mock()
    progress = AsyncMock()
    pipe = OCRPipeline(
        pdf_handler=PDFHandler(),
        grounded_backend=_StubGroundedBackend(blocks, page_sizes),
        output_writer=writer,
    )
    output = tmp_path / "existing.pdf"
    output.write_bytes(b"existing output")

    with pytest.raises(RuntimeError, match="Grounded OCR") as exc:
        await pipe.run("in.pdf", str(output), progress=progress)

    message = str(exc.value)
    assert empty_pages in message
    assert "hybrid" in message
    assert "--grounded" in message
    assert "engine=hybrid" in message
    writer.assert_not_called()
    assert output.read_bytes() == b"existing output"
    assert all(call.args[0] != "embed" for call in progress.await_args_list)


async def test_grounded_rejects_whitespace_only_blocks(tmp_path):
    backend = _StubGroundedBackend(
        [GroundedBlock([0.1, 0.1, 0.5, 0.2], " \n\t", 0)], [(100, 100)],
    )
    writer = Mock()
    pipe = OCRPipeline(
        pdf_handler=PDFHandler(), grounded_backend=backend, output_writer=writer,
    )
    with pytest.raises(RuntimeError, match="no text blocks"):
        await pipe.run("in.pdf", str(tmp_path / "out.pdf"))
    writer.assert_not_called()


@pytest.mark.parametrize("raw", [
    "# A page transcribed as Markdown\n\nReadable text without bounding boxes.",
    "",
    "[]",
    '[{"bbox_2d": [0, 0, 100, 50], "content": "truncated"',
    '[{"content": "missing bounding box"}]',
    '[{"bbox_2d": [0, 0, 100, 50], "content": "   "}]',
])
async def test_prompted_grounded_rejects_unusable_model_output(
    monkeypatch, example_pdfs, tmp_path, raw,
):
    complete = AsyncMock(return_value=raw)
    client = SimpleNamespace(complete=complete)
    pipe = OCRPipeline(
        pdf_handler=PDFHandler(),
        grounded_backend=PromptedGroundedOCR(model="ovisocr2", client=client),
    )
    output = tmp_path / "out.pdf"

    with pytest.raises(RuntimeError, match="no text blocks"):
        await pipe.run(str(example_pdfs["digital.pdf"]), str(output))

    assert complete.await_count == (3 if '"truncated"' in raw else 1)
    assert not output.exists()


async def test_one_failing_grounded_page_does_not_lose_others(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "three.pdf"
    with fitz.open() as doc:
        for _ in range(3):
            doc.new_page(width=72, height=72)
        doc.save(source)

    complete = AsyncMock(side_effect=[
        '[{"bbox_2d":[0,0,100,50],"content":"p0"}]',
        RuntimeError("page failed"),
        '[{"bbox_2d":[0,0,100,50],"content":"p2"}]',
    ])
    backend = PromptedGroundedOCR(
        model="vision", client=SimpleNamespace(complete=complete), concurrency=1,
    )
    progress = AsyncMock()
    response = await backend.ocr_document(str(source), progress=progress)
    assert [(b.page_index, b.text) for b in response.blocks] == [(0, "p0"), (2, "p2")]
    assert response.page_indices == [0, 1, 2]
    assert len(response.page_sizes) == 3
    assert [call.args[1:3] for call in progress.await_args_list] == [
        (0, 3), (1, 3), (2, 3), (3, 3),
    ]


# --- prompted grounded JSON parser (Qwen2.5-VL / Qwen3-VL response shapes) --


async def test_truncated_page_is_retried_in_source_regions(tmp_path):
    from pdf_ocr.inference import IncompleteCompletionError

    source = tmp_path / "two.pdf"
    with fitz.open() as doc:
        doc.new_page(width=300, height=900)
        doc.new_page(width=300, height=900)
        doc.save(source)
    line = '[{"bbox_2d":[100,450,900,550],"content":"recognized"}]'
    complete = AsyncMock(side_effect=[IncompleteCompletionError("token limit"), line, line, line])
    backend = PromptedGroundedOCR(
        model="qwen3-vl", client=SimpleNamespace(complete=complete), max_image_dim=300,
    )
    response = await backend.ocr_document(str(source), pages="2")
    assert response.page_indices == [1]
    assert len(response.blocks) == 3
    assert all(b.page_index == 1 for b in response.blocks)
    assert [(b.bbox[1] + b.bbox[3]) / 2 for b in response.blocks] == pytest.approx(
        [(1 / 3 + 0.025) / 2, 1 / 2, 1 - (1 / 3 + 0.025) / 2], abs=0.002,
    )
    assert complete.await_count == 4


@pytest.mark.parametrize("custom_prompt", [None, "Read only the header"])
async def test_grounded_recovers_valid_but_severely_partial_page(tmp_path, custom_prompt):
    source = tmp_path / "ink.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=900)
        page.draw_rect(fitz.Rect(30, 20, 100, 25), fill=(0, 0, 0))
        for y in range(350, 850, 30):
            page.draw_rect(fitz.Rect(30, y, 270, y + 10), fill=(0, 0, 0))
        doc.save(source)
    header = '[{"bbox_2d":[100,20,350,35],"content":"Header"}]'
    line = '[{"bbox_2d":[100,450,900,550],"content":"More source text"}]'
    complete = AsyncMock(side_effect=[header, line, line, line])
    backend = PromptedGroundedOCR(
        model="qwen3-vl", client=SimpleNamespace(complete=complete), prompt=custom_prompt,
    )
    response = await backend.ocr_document(str(source))
    assert response.page_errors == {}
    assert complete.await_count == (1 if custom_prompt else 4)
    assert len(response.blocks) == (1 if custom_prompt else 3)


def test_ink_coverage_accepts_sparse_headers_and_uniform_backgrounds():
    from PIL import Image, ImageDraw
    from pdf_ocr.core.grounded import _text_ink_coverage

    header = GroundedBlock([0.1, 0.02, 0.5, 0.08], "Header", 0)
    for color in ["white", "black", "blue", "gray"]:
        assert _text_ink_coverage(Image.new("RGB", (300, 900), color), [header]) is None
    image = Image.new("RGB", (300, 900), "white")
    ImageDraw.Draw(image).rectangle((40, 30, 140, 60), fill="black")
    assert _text_ink_coverage(image, [header]) == 1.0


async def test_grounded_enforces_deadline_for_injected_clients(tmp_path):
    source = tmp_path / "page.pdf"
    with fitz.open() as doc:
        doc.new_page(width=72, height=72)
        doc.save(source)
    cancelled = asyncio.Event()

    async def stall(*args, **kwargs):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.set()

    backend = PromptedGroundedOCR(
        model="vision", client=SimpleNamespace(complete=stall), timeout_s=0.02,
    )
    started = asyncio.get_running_loop().time()
    response = await backend.ocr_document(str(source))
    assert not response.blocks
    assert asyncio.get_running_loop().time() - started < 0.5
    assert cancelled.is_set()
    assert "deadline" in response.page_errors[0]


@pytest.mark.parametrize("empty_band", [
    '{\n  "blocks": [ ]\n}',
    '```json\n[]\n```',
    '{"blocks":[{"bbox_2d":[0,0,1000,10],"content":""}]}',
    '[{"bbox_2d":[0,0,1000,10],"text":"  \\n"}]',
    '{"blocks":[{"content":"","text_content":" "}]}',
])
async def test_grounded_recovery_accepts_formatted_empty_bands(tmp_path, empty_band):
    from pdf_ocr.inference import IncompleteCompletionError

    source = tmp_path / "page.pdf"
    with fitz.open() as doc:
        doc.new_page(width=300, height=900)
        doc.save(source)
    line = '[{"bbox_2d":[100,450,900,550],"content":"recognized"}]'
    complete = AsyncMock(side_effect=[IncompleteCompletionError("limit"), line, empty_band, line])
    backend = PromptedGroundedOCR(model="vision", client=SimpleNamespace(complete=complete))
    response = await backend.ocr_document(str(source))
    assert len(response.blocks) == 2
    assert not response.page_errors
    assert complete.await_count == 4


@pytest.mark.parametrize("response", [
    '{"blocks":[{"content":"","text":"visible text"}]}',
    '{"blocks":[{"bbox_2d":[0,0,1000,10]}]}',
    '{"blocks":[{"content":null}]}',
    '{"blocks":[null]}',
    '{}',
    'not JSON',
])
def test_empty_band_detection_rejects_nonempty_or_unrecognized_data(response):
    from pdf_ocr.core.grounded import _is_empty_grounded_json

    assert not _is_empty_grounded_json(response)


async def test_grounded_pipeline_reports_failure_cause_on_the_source_page(tmp_path):
    backend = SimpleNamespace(ocr_document=AsyncMock(return_value=SimpleNamespace(
        blocks=[], page_sizes=[(100, 100)], page_indices=[2],
        page_errors={2: "request timed out"},
    )))
    writer = Mock()
    pipe = OCRPipeline(pdf_handler=PDFHandler(), grounded_backend=backend, output_writer=writer)
    with pytest.raises(RuntimeError, match="page 3: request timed out"):
        await pipe.run("in.pdf", str(tmp_path / "out.pdf"))
    writer.assert_not_called()


async def test_grounded_cancellation_drains_page_tasks(tmp_path):
    source = tmp_path / "three.pdf"
    with fitz.open() as doc:
        for _ in range(3):
            doc.new_page(width=72, height=72)
        doc.save(source)
    entered = asyncio.Event()
    exited = asyncio.Event()
    calls = 0

    async def stall(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        try:
            await asyncio.sleep(10)
        finally:
            exited.set()

    backend = PromptedGroundedOCR(model="vision", client=SimpleNamespace(complete=stall))
    task = asyncio.create_task(backend.ocr_document(str(source)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert exited.is_set()
    await asyncio.sleep(0)
    assert calls == 1


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


class TestThousandGridDetection:
    """Qwen3-VL emits bbox_2d on a fixed 0-1000 grid; Qwen2.5-VL emits
    absolute pixels. The parser must tell them apart per page instead of
    silently dividing grid coordinates by the pixel dimensions."""

    def test_coords_exceeding_image_width_treated_as_grid(self):
        # 791px-wide image, x up to 990: impossible as pixels, so the
        # page is on the 0-1000 grid and divides by 1000.
        raw = '[{"bbox_2d":[700,100,990,140],"content":"right-edge line"}]'
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=791, img_h=1024)
        assert blocks[0].bbox == pytest.approx([0.700, 100 / 1000, 0.990, 0.140])

    def test_grid_detection_is_page_level(self):
        # One in-frame box plus one frame-exceeding box: the whole page
        # rescales together — conventions don't vary within a response.
        raw = (
            '[{"bbox_2d":[50,50,400,80],"content":"in frame"},'
            '{"bbox_2d":[700,100,990,140],"content":"beyond width"}]'
        )
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=791, img_h=1024)
        assert blocks[0].bbox == pytest.approx([0.05, 0.05, 0.40, 0.08])

    def test_large_image_with_subgrid_coords_treated_as_grid(self):
        # 2048x2048 image, every coordinate within 1000 in both axes:
        # full-page text confined to the top-left corner is implausible.
        raw = (
            '[{"bbox_2d":[100,100,900,150],"content":"a"},'
            '{"bbox_2d":[100,800,950,900],"content":"b"}]'
        )
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=2048, img_h=2048)
        assert blocks[0].bbox == pytest.approx([0.1, 0.1, 0.9, 0.15])

    def test_coords_beyond_1000_always_pixels(self):
        raw = '[{"bbox_2d":[0,0,1500,800],"content":"wide"}]'
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=2048, img_h=2048)
        assert blocks[0].bbox == pytest.approx(
            [0.0, 0.0, 1500 / 2048, 800 / 2048]
        )

    def test_small_image_in_frame_coords_stay_pixels(self):
        # Default thumbnail sizes (<=1024) sit in the ambiguous zone;
        # coordinates within the frame are taken at face value.
        raw = '[{"bbox_2d":[0,0,1000,1000],"content":"full page"}]'
        blocks = _parse_grounded_json(raw, page_idx=0, img_w=1024, img_h=1024)
        assert blocks[0].bbox == pytest.approx(
            [0.0, 0.0, 1000 / 1024, 1000 / 1024]
        )


class TestPromptedGroundedEnsureModelLoaded:
    """Pre-flight model verification for the grounded path. Issue #7 was
    actually filed against the grounded path specifically — user had
    OlmOCR loaded but requested Qwen3-VL, and got OlmOCR's bad grounded
    output instead of Qwen3-VL's good output."""

    def _patch_openai(self, monkeypatch, model_ids=None, raise_exc=None):
        async def _list():
            if raise_exc is not None:
                raise raise_exc
            return SimpleNamespace(
                data=[SimpleNamespace(id=m) for m in (model_ids or [])]
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(list=_list)
        )

        def _fake_async_openai(*args, **kwargs):
            return fake_client

        # ensure_model_loaded does `from openai import AsyncOpenAI` inside,
        # so we patch the source module's attribute.
        monkeypatch.setattr("openai.AsyncOpenAI", _fake_async_openai)
        return fake_client

    def test_passes_when_model_loaded(self, monkeypatch):
        self._patch_openai(monkeypatch, model_ids=["qwen/qwen3-vl-8b"])
        backend = PromptedGroundedOCR(
            api_base="http://localhost:1234/v1",
            model="qwen/qwen3-vl-8b",
        )
        asyncio.run(backend.ensure_model_loaded())  # no raise

    def test_raises_on_mismatch_with_helpful_message(self, monkeypatch):
        # The exact issue #7 scenario: requested grounded-capable Qwen3-VL,
        # but LM Studio has the OlmOCR text-only model loaded.
        self._patch_openai(monkeypatch, model_ids=["allenai_olmocr-2-7b-1025"])
        backend = PromptedGroundedOCR(
            api_base="http://localhost:1234/v1",
            model="qwen/qwen3-vl-8b",
        )
        with pytest.raises(ModelNotLoadedError) as exc_info:
            asyncio.run(backend.ensure_model_loaded())
        msg = str(exc_info.value)
        assert "qwen/qwen3-vl-8b" in msg
        assert "allenai_olmocr-2-7b-1025" in msg
        assert "--no-verify-model" in msg

    def test_subclass_of_llm_call_error(self):
        # Catchable via the same except-clause as other LLM failures.
        assert issubclass(ModelNotLoadedError, LLMCallError)
