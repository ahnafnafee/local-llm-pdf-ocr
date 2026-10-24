"""Native model conventions must not leak into the transport contract."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import pytest

from pdf_ocr.core.grounded import (
    PromptedGroundedOCR, _parse_lighton_grounding, _parse_paddle_spotting,
)
from pdf_ocr.inference import LLMCallError
from pdf_ocr.core.ocr import OCRProcessor
from pdf_ocr.core.profiles import resolve_profile


def test_profile_alias_override_keeps_model_identity():
    ocr = OCRProcessor(model="served-alias", profile="lightonocr-3")
    assert ocr.model == "served-alias"
    assert ocr.page_prompt == ocr.crop_prompt == ""
    assert ocr.client.image_first
    assert ocr.client.generation_options == {"top_p": 0.9}
    assert resolve_profile("unrecognized-model").name == "generic"
    assert resolve_profile("lightonai/LightOnOCR-3-4B", "generic").name == "generic"
    with pytest.raises(ValueError, match="Unknown model profile"):
        resolve_profile("model", "misspelled-profile")


def test_profile_output_budget_preserves_explicit_limits():
    assert PromptedGroundedOCR(model="Qwen/Qwen3.5-9B").max_tokens == 8192
    assert PromptedGroundedOCR(model="Qwen/Qwen3.5-9B", max_tokens=2000).max_tokens == 2000
    assert PromptedGroundedOCR(model="other").max_tokens == 4096


def test_native_grounding_preserves_text_order_and_excludes_descriptions():
    raw = ("![title](100,20,800,90)\n# A title\n"
           "![image](100,100,800,300)\nA generated picture description\n"
           "![text+](100,310,800,400)\nLiteral words\nSecond line\n"
           "![chart](100,450,800,900)\nGenerated chart table")
    blocks = _parse_lighton_grounding(raw, 7)
    assert [b.text for b in blocks] == ["# A title", "Literal words\nSecond line"]
    assert [b.label for b in blocks] == ["title", "text"]
    assert [b.page_index for b in blocks] == [7, 7]
    assert blocks[1].bbox == [0.1, 0.31, 0.8, 0.4]


@pytest.mark.parametrize("raw", [
    "Just text without boxes",
    "Unexpected preface\n![text](0,0,100,100)\ntext",
    "![text](100,0,0,100)\ntext",
    "![text](0,0,1001,100)\ntext",
    "![text](0,0,100,100)\ntext\n![text](0,200,",
])
def test_invalid_native_grounding_is_recoverable_error(raw):
    with pytest.raises(ValueError):
        _parse_lighton_grounding(raw, 0)


async def test_native_grounding_uses_native_completion_and_source_page_index(tmp_path):
    source = tmp_path / "two-pages.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(source)
    client = SimpleNamespace(
        complete=AsyncMock(return_value="![text](100,200,800,300)\nSource words"),
        complete_json=AsyncMock(side_effect=AssertionError("Native output is not JSON")),
    )
    backend = PromptedGroundedOCR(model="lightonai/LightOnOCR-3-4B", client=client)
    result = await backend.ocr_document(str(source), pages="2")
    assert result.page_indices == [1]
    assert result.page_errors == {}
    assert result.blocks[0].page_index == 1
    assert result.blocks[0].text == "Source words"
    client.complete_json.assert_not_called()
    assert client.complete.call_args.args[0] == "grounding"
    assert client.complete.call_args.kwargs["temperature"] == 0.2


def test_paddle_spotting_preserves_polygon_text_and_source_page():
    first = "".join(f"<|LOC_{n}|>" for n in [100, 20, 800, 30, 790, 90, 110, 80])
    second = "".join(f"<|LOC_{n}|>" for n in [50, 110, 500, 110, 500, 180, 50, 180])
    blocks = _parse_paddle_spotting(
        f"<|TEXT_START|>Source spelling<|TEXT_END|><|LOC_BEGIN|>{first}<|LOC_END|>\n"
        f"Another line{second}", 3,
    )
    assert [b.text for b in blocks] == ["Source spelling", "Another line"]
    assert [b.page_index for b in blocks] == [3, 3]
    assert blocks[0].bbox == [0.1, 0.02, 0.8, 0.09]
    assert resolve_profile("PaddlePaddle/PaddleOCR-VL-1.6").grounding_prompt == "Spotting:"


@pytest.mark.parametrize("coordinates", [
    [0, 0, 100, 0, 100, 100, 0],
    [0, 0, 1001, 0, 1001, 100, 0, 100],
    [100, 0, 100, 0, 100, 100, 100, 100],
])
def test_paddle_rejects_partial_or_invalid_polygons(coordinates):
    with pytest.raises(ValueError):
        _parse_paddle_spotting("Text" + "".join(f"<|LOC_{n}|>" for n in coordinates), 0)


def test_paddle_requires_preserved_special_tokens():
    with pytest.raises(LLMCallError, match="preserve special tokens"):
        _parse_paddle_spotting("Text without the server's stripped location tokens", 0)


def test_paddle_rejects_unlocated_trailing_text():
    polygon = "".join(f"<|LOC_{n}|>" for n in [0, 0, 100, 0, 100, 100, 0, 100])
    with pytest.raises(ValueError, match="trailing text"):
        _parse_paddle_spotting(f"First{polygon}Unlocated second line", 0)
