"""Page filtering must bound rendering and inference, not just output text."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fitz
import pytest
from PIL import Image

from pdf_ocr.core.grounded import PromptedGroundedOCR
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.pipeline import OCRPipeline
from pdf_ocr.utils.pages import parse_page_range, select_pages


@pytest.fixture
def source_pdf(tmp_path):
    source = tmp_path / "source.pdf"
    with fitz.open() as doc:
        for i in range(5):
            doc.new_page(width=100 + i * 10, height=160)
        doc.save(source)
    return source


@pytest.fixture
def rasterized_pages(monkeypatch):
    rendered = []
    get_pixmap = fitz.Page.get_pixmap

    def record(page, *args, **kwargs):
        rendered.append(page.number)
        return get_pixmap(page, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_pixmap", record)
    return rendered


@pytest.mark.parametrize("pages, expected", [
    (None, [0, 1, 2, 3, 4]),
    ("1-3", [0, 1, 2]),
    ("4,2,2", [1, 3]),
    ("4-999999999", [3, 4]),
    ("5", [4]),
])
def test_pdf_only_rasterizes_selected_source_pages(source_pdf, rasterized_pages, pages, expected):
    images = PDFHandler().convert_to_images(source_pdf, dpi=72, pages=pages)
    assert list(images) == expected
    assert rasterized_pages == expected


@pytest.mark.parametrize("mode", ["hybrid", "text", "grounded"])
async def test_pipeline_selects_before_rendering_and_reports_selected_total(
    source_pdf, tmp_path, rasterized_pages, mode,
):
    writer = Mock()
    progress = AsyncMock()
    ocr = SimpleNamespace(perform_ocr=AsyncMock(return_value=["recognized"]))
    aligner = SimpleNamespace(
        get_detected_boxes_batch=Mock(side_effect=lambda images: [[[0, 0, 1, 1]]] * len(images)),
        align_text=lambda boxes, text: [(boxes[0][0], text[0])],
    )
    complete = AsyncMock(return_value='[{"bbox_2d":[0,0,100,50],"content":"recognized"}]')
    backend = PromptedGroundedOCR(
        model="vision", client=SimpleNamespace(complete=complete), dpi=72,
    )
    pipe = OCRPipeline(
        aligner=aligner, ocr_processor=ocr, pdf_handler=PDFHandler(),
        grounded_backend=backend if mode == "grounded" else None,
        output_writer=writer,
    )
    result = await pipe.run(
        str(source_pdf), str(tmp_path / "out.pdf"), pages="4,2,2", dpi=72,
        text_only=mode == "text", refine=False, preprocess="never", progress=progress,
    )
    assert set(result) == {1, 3}
    assert rasterized_pages == [1, 3]
    assert set(writer.call_args.args[2]) == {1, 3}
    assert (complete if mode == "grounded" else ocr.perform_ocr).await_count == 2
    ocr_updates = [c.args[1:3] for c in progress.await_args_list if c.args[0] == "ocr"]
    assert ocr_updates == [(0, 2), (1, 2), (2, 2)]
    if mode != "grounded":
        assert any(c.args[3] == "Converted 2 pages." for c in progress.await_args_list)
    if mode == "hybrid":
        assert len(aligner.get_detected_boxes_batch.call_args.args[0]) == 2
    else:
        aligner.get_detected_boxes_batch.assert_not_called()


@pytest.mark.parametrize("mode", ["hybrid", "text", "grounded"])
@pytest.mark.parametrize("pages", ["20-30", "0", "3-1", "invalid"])
async def test_invalid_or_empty_selection_does_no_work(
    source_pdf, tmp_path, rasterized_pages, mode, pages,
):
    writer = Mock()
    complete = AsyncMock()
    ocr = SimpleNamespace(perform_ocr=AsyncMock())
    aligner = Mock()
    pipe = OCRPipeline(
        pdf_handler=PDFHandler(), aligner=aligner, ocr_processor=ocr,
        grounded_backend=PromptedGroundedOCR(
            model="vision", client=SimpleNamespace(complete=complete),
        ) if mode == "grounded" else None,
        output_writer=writer,
    )
    output = tmp_path / "out.pdf"
    output.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="[Pp]age range"):
        await pipe.run(str(source_pdf), str(output), pages=pages, text_only=mode == "text")
    assert rasterized_pages == []
    complete.assert_not_awaited()
    ocr.perform_ocr.assert_not_awaited()
    aligner.get_detected_boxes_batch.assert_not_called()
    writer.assert_not_called()
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("failed_call, missing_page", [(0, 2), (1, 4)])
async def test_grounded_missing_selected_page_reports_source_number(
    source_pdf, tmp_path, failed_call, missing_page,
):
    responses = ['[{"bbox_2d":[0,0,100,50],"content":"recognized"}]'] * 2
    responses[failed_call] = "[]"
    writer = Mock()
    pipe = OCRPipeline(
        pdf_handler=PDFHandler(), output_writer=writer,
        grounded_backend=PromptedGroundedOCR(
            model="vision", concurrency=1,
            client=SimpleNamespace(complete=AsyncMock(side_effect=responses)),
        ),
    )
    with pytest.raises(RuntimeError, match=rf"page\(s\) {missing_page}\."):
        await pipe.run(str(source_pdf), str(tmp_path / "out.pdf"), pages="2,4")
    writer.assert_not_called()


@pytest.mark.parametrize("grounded", [False, True])
async def test_multiframe_images_only_convert_selected_frames(tmp_path, monkeypatch, grounded):
    source = tmp_path / "frames.tiff"
    frames = [Image.new("RGB", (50, 60), (i * 40, 0, 0)) for i in range(5)]
    frames[0].save(source, save_all=True, append_images=frames[1:])
    converted = []
    convert = Image.Image.convert

    def record(image, *args, **kwargs):
        converted.append(image.getpixel((0, 0))[0])
        return convert(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", record)
    if grounded:
        complete = AsyncMock(return_value='[{"bbox_2d":[0,0,50,30],"content":"frame"}]')
        response = await PromptedGroundedOCR(
            model="vision", client=SimpleNamespace(complete=complete),
        ).ocr_document(str(source), pages="2,4")
        assert response.page_indices == [1, 3]
        assert [b.page_index for b in response.blocks] == [1, 3]
        assert response.page_sizes == [(50, 60), (50, 60)]
        assert complete.await_count == 2
    else:
        assert list(PDFHandler().convert_to_images(source, pages="2,4")) == [1, 3]
    assert converted == [40, 120]


async def test_grounded_sparse_selection_embeds_on_original_pages(source_pdf, tmp_path):
    complete = AsyncMock(side_effect=[
        '[{"bbox_2d":[0,0,100,50],"content":"SECOND"}]',
        '[{"bbox_2d":[0,0,100,50],"content":"FOURTH"}]',
    ])
    pipe = OCRPipeline(
        pdf_handler=PDFHandler(),
        grounded_backend=PromptedGroundedOCR(
            model="vision", client=SimpleNamespace(complete=complete), concurrency=1,
        ),
    )
    output = tmp_path / "out.pdf"
    await pipe.run(str(source_pdf), str(output), pages="2,4", dpi=72)
    with fitz.open(output) as doc:
        assert len(doc) == 5
        assert "SECOND" in doc[1].get_text()
        assert "FOURTH" in doc[3].get_text()
        assert all(not doc[i].get_text().strip() for i in (0, 2, 4))


def test_parser_public_import_and_empty_default():
    from pdf_ocr import parse_page_range as public_parser

    assert public_parser is parse_page_range
    assert select_pages("", 3) == [0, 1, 2]
