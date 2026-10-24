"""Exercise server-neutral configuration and the real SDK over local HTTP mocks."""

import base64
import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fitz
import httpx
import pytest
from openai import AsyncOpenAI
from PIL import Image

from pdf_ocr.core.grounded import PromptedGroundedOCR
from pdf_ocr.core.ocr import OCRProcessor, OLMOCR_PAGE_PROMPT, PAGE_PROMPT
from pdf_ocr.inference import (
    LLMCallError, LLMConfig, OpenAICompatibleClient,
)


def image_base64(fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 30), "white").save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


def test_configuration_uses_environment_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1/")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    config = LLMConfig.from_env()
    assert (config.api_base, config.model, config.api_key) == (
        "http://localhost:11434/v1", "environment-model", "environment-key",
    )
    config = LLMConfig.from_env(
        api_base="https://gateway.test/api/v1/", model="override", api_key="override-key",
    )
    assert (config.api_base, config.model, config.api_key) == (
        "https://gateway.test/api/v1", "override", "override-key",
    )
    assert "override-key" not in repr(config)


@pytest.mark.parametrize("backend_type, expected_model", [
    (OCRProcessor, "lightonai/LightOnOCR-3-4B"),
    (PromptedGroundedOCR, "lightonai/LightOnOCR-3-4B"),
])
def test_unconfigured_backends_use_lighton_and_preserve_endpoint(monkeypatch, backend_type, expected_model):
    monkeypatch.delenv("LLM_API_BASE")
    monkeypatch.delenv("LLM_MODEL")
    backend = backend_type()
    assert backend.api_base == "http://localhost:1234/v1"
    assert backend.model == expected_model
    assert backend.client.config.api_base == backend.api_base
    assert backend.client.config.model == expected_model
    assert backend.client.config.api_key == "not-required"


@pytest.mark.parametrize("backend_type", [OCRProcessor, PromptedGroundedOCR])
def test_environment_and_explicit_settings_override_backend_defaults(backend_type):
    backend = backend_type()
    assert (backend.api_base, backend.model) == ("http://inference.test/v1", "test-vision")
    backend = backend_type(api_base="http://localhost:11434/v1", model="custom-vision")
    assert (backend.api_base, backend.model) == ("http://localhost:11434/v1", "custom-vision")


def test_model_discovery_does_not_require_model_configuration(monkeypatch):
    monkeypatch.delenv("LLM_MODEL")
    config = LLMConfig.from_env(require_model=False)
    assert config.model == ""
    assert config.api_key == "not-required"


@pytest.mark.parametrize("base", ["localhost:8000", "file:///server", "https://user:secret@host/v1"])
def test_invalid_endpoint_has_actionable_error_without_credentials(base):
    with pytest.raises(ValueError, match="URL") as error:
        LLMConfig.from_env(api_base=base)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("endpoint", [
    "http://localhost:11434/v1", "http://localhost:8000/v1", "https://gateway.test/api/v1",
])
async def test_all_ocr_paths_share_standard_requests_and_authentication(tmp_path, endpoint):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        if request.method == "GET":
            assert str(request.url) == endpoint + "/models"
            return httpx.Response(200, json={"data": [{"id": "vision-model"}]})
        assert str(request.url) == endpoint + "/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "vision-model"
        content = body["messages"][0]["content"]
        prompt = content[0]["text"]
        expected_mime = "image/png" if "JSON array" in prompt else "image/jpeg"
        assert content[1]["image_url"]["url"].startswith(f"data:{expected_mime};base64,")
        text = ('[{"bbox_2d":[0,0,20,15],"content":"grounded"}]'
                if "JSON array" in prompt else "transcribed")
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": text}}],
        })

    config = LLMConfig(endpoint, "vision-model", "test-key")
    async with AsyncOpenAI(
        base_url=endpoint, api_key=config.api_key, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        adapter = OpenAICompatibleClient(config, sdk_client=sdk)
        ocr = OCRProcessor(model=config.model, client=adapter)
        grounded = PromptedGroundedOCR(model=config.model, client=adapter)
        await ocr.ensure_model_available()
        await grounded.ensure_model_available()
        assert await ocr.perform_ocr(image_base64()) == ["transcribed"]
        assert await ocr.perform_ocr_on_crop(image_base64()) == "transcribed"
        source = tmp_path / "input.pdf"
        with fitz.open() as doc:
            doc.new_page(width=72, height=72)
            doc.save(source)
        response = await grounded.ocr_document(str(source))
        assert response.blocks[0].text == "grounded"
        await ocr.aclose()
        await grounded.aclose()
        await adapter.aclose()
        assert not sdk.is_closed()  # caller owns the injected client

    inference = [request for request in requests if request.method == "POST"]
    assert [json.loads(r.content)["max_tokens"] for r in inference] == [6144, 256, 4096]
    assert [r.extensions["timeout"]["read"] for r in inference] == [240, 60, 240]
    assert [json.loads(r.content)["temperature"] for r in inference] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("PNG", "image/png")])
async def test_adapter_image_mime_matches_encoded_bytes(fmt, mime):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="text"))])
    create = AsyncMock(return_value=completion)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sdk.with_options = lambda **kwargs: sdk
    adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
    await adapter.complete("read", image_base64(fmt), model="vision", timeout=1, max_tokens=10, temperature=0)
    content = create.call_args.kwargs["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith(f"data:{mime};base64,")


async def test_reasoning_only_response_reports_server_configuration_error():
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="", reasoning_content="![text](0,0,100,100) transcribed",
    ))])
    create = AsyncMock(return_value=completion)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sdk.with_options = lambda **kwargs: sdk
    adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
    with pytest.raises(LLMCallError, match="official chat template and disable thinking"):
        await adapter.complete("grounding", image_base64(), model="vision", timeout=1,
                               max_tokens=100, temperature=0)
    assert create.await_count == 1


@pytest.mark.parametrize("backend_type", [OCRProcessor, PromptedGroundedOCR])
async def test_injected_transport_needs_no_sdk_or_server_configuration(monkeypatch, backend_type):
    monkeypatch.delenv("LLM_API_BASE")
    sdk_factory = Mock(side_effect=AssertionError("must not create an SDK client"))
    monkeypatch.setattr("openai.AsyncOpenAI", sdk_factory)
    client = SimpleNamespace(
        complete=AsyncMock(return_value="text"),
        list_models=AsyncMock(return_value=["custom-vision"]),
        aclose=AsyncMock(),
    )
    backend = backend_type(model="custom-vision", client=client)
    await backend.ensure_model_available()
    await backend.aclose()
    client.aclose.assert_not_awaited()
    sdk_factory.assert_not_called()


@pytest.mark.parametrize("prompt,types", [
    ("grounding", ["image_url", "text"]),
    ("", ["image_url"]),
])
async def test_native_ocr_image_order_and_empty_prompt_reach_http(prompt, types):
    """LightOn's native modes distinguish image-first from text-first input."""
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "recognized"},
            "finish_reason": "stop",
        }]})

    async with AsyncOpenAI(
        base_url="http://inference.test/v1", api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        adapter = OpenAICompatibleClient(
            LLMConfig("http://inference.test/v1"), sdk_client=sdk, image_first=True,
        )
        result = await adapter.complete(prompt, image_base64(), model="vision",
                                        timeout=2, max_tokens=10, temperature=0)
    assert result == "recognized"
    content = bodies[0]["messages"][0]["content"]
    assert [part["type"] for part in content] == types
    if prompt:
        assert content[1]["text"] == prompt


async def test_owned_sdk_is_reused_and_closed(monkeypatch):
    sdk = SimpleNamespace(
        models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(data=[]))),
        close=AsyncMock(),
    )
    factory = Mock(return_value=sdk)
    monkeypatch.setattr("openai.AsyncOpenAI", factory)
    async with OpenAICompatibleClient(LLMConfig("http://inference.test/v1")) as adapter:
        await adapter.list_models()
        await adapter.list_models()
    factory.assert_called_once()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("action", ["discovery", "completion"])
async def test_transport_failures_are_reported_consistently(action):
    def fail(request):
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    async with AsyncOpenAI(
        base_url="http://inference.test/v1", api_key="test-key", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    ) as sdk:
        adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
        with pytest.raises(LLMCallError, match="LLM_API_KEY") as error:
            if action == "discovery":
                await adapter.list_models()
            else:
                await adapter.complete("read", image_base64(), model="vision", timeout=1, max_tokens=10, temperature=0)
        assert "http://inference.test/v1" in str(error.value)
        assert "test-key" not in str(error.value)


def test_prompt_choice_is_independent_of_server():
    client = SimpleNamespace()
    assert OCRProcessor(model="vision", client=client).page_prompt == PAGE_PROMPT
    assert OCRProcessor(model="allenai/olmocr-2", client=client).page_prompt == OLMOCR_PAGE_PROMPT
    assert OCRProcessor(model="vision", client=client, page_prompt="custom").page_prompt == "custom"
    native = OCRProcessor(model="vision", client=client, page_prompt="", crop_prompt="")
    assert native.page_prompt == native.crop_prompt == ""


async def test_completion_deadline_includes_sdk_retries():
    requests = []

    async def stall(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        requests.append(request)
        await asyncio.sleep(1)
        return httpx.Response(500)

    async with AsyncOpenAI(
        base_url="http://inference.test/v1", api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(stall)),
    ) as sdk:
        adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
        await sdk.models.list()  # exclude one-time SDK/platform initialization
        started = asyncio.get_running_loop().time()
        with pytest.raises(LLMCallError, match="deadline"):
            await adapter.complete("read", image_base64(), model="vision",
                                   timeout=0.02, max_tokens=10, temperature=0)
        assert asyncio.get_running_loop().time() - started < 0.5
        assert len(requests) <= 1  # deadline may expire during SDK preparation


async def test_token_limit_is_reported_instead_of_accepting_partial_text():
    from pdf_ocr.inference import IncompleteCompletionError

    def respond(request):
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": '[{"content":"cut off'},
        }]})

    async with AsyncOpenAI(
        base_url="http://inference.test/v1", api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
        with pytest.raises(IncompleteCompletionError, match="token limit"):
            await adapter.complete("read", image_base64(), model="vision",
                                   timeout=1, max_tokens=10, temperature=0)


async def test_structured_output_falls_back_only_when_format_is_unsupported():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {
                "message": "response_format json_schema is not supported",
            }})
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "[]"}, "finish_reason": "stop",
        }]})

    async with AsyncOpenAI(
        base_url="http://inference.test/v1", api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
        for _ in range(2):
            assert await adapter.complete_json(
                "read", image_base64(), schema={"type": "object"}, model="vision",
                timeout=2, max_tokens=10, temperature=0,
            ) == "[]"
    assert len(requests) == 3  # format support is negotiated once per adapter
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert "response_format" not in requests[1]
    assert "response_format" not in requests[2]


async def test_structured_output_does_not_retry_model_errors():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "model has no vision encoder"}})

    async with AsyncOpenAI(
        base_url="http://inference.test/v1", api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        adapter = OpenAICompatibleClient(LLMConfig("http://inference.test/v1"), sdk_client=sdk)
        with pytest.raises(LLMCallError, match="vision encoder"):
            await adapter.complete_json(
                "read", image_base64(), schema={"type": "object"}, model="vision",
                timeout=2, max_tokens=10, temperature=0,
            )
    assert len(requests) == 1
