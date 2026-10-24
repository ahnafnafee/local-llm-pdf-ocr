"""Server configuration and the transport boundary for vision inference.

OCR components depend on VisionClient, not an SDK or a server application.
OpenAICompatibleClient is the default adapter; injected clients remain owned
by their caller and can implement the same small async interface.
"""

from __future__ import annotations

import base64
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_BASE = "http://localhost:1234/v1"
DEFAULT_MODEL = "lightonai/LightOnOCR-3-4B"
DEFAULT_GROUNDED_MODEL = DEFAULT_MODEL


class LLMCallError(RuntimeError):
    """A model discovery or inference request failed."""


class IncompleteCompletionError(LLMCallError):
    """Generation exhausted its token budget; the response is incomplete."""


class _StructuredOutputUnsupported(LLMCallError):
    pass


class ModelNotAvailableError(LLMCallError):
    """The requested model is absent from the server's advertised catalog."""


# Preserve the public exception used by earlier releases. Model listing does
# not indicate whether weights are resident in memory on a particular server.
ModelNotLoadedError = ModelNotAvailableError


@dataclass(frozen=True)
class LLMConfig:
    api_base: str
    model: str = ""
    api_key: str = field(default="not-required", repr=False)

    @classmethod
    def from_env(
        cls, *, api_base: str | None = None, model: str | None = None,
        api_key: str | None = None, require_model: bool = True,
        require_endpoint: bool = True,
        default_model: str = DEFAULT_MODEL,
    ) -> LLMConfig:
        """Explicit values override LLM_* environment variables and .env.

        Unset values use the local endpoint and LightOnOCR-3 model defaults.
        Discovery needs only an endpoint; an injected VisionClient needs only
        a model. Neither the defaults nor the adapter require a particular server.
        """
        base = (api_base or os.getenv("LLM_API_BASE", DEFAULT_API_BASE)).strip().rstrip("/")
        selected_model = (model or os.getenv(
            "LLM_MODEL", default_model if require_model else "",
        )).strip()
        if require_endpoint and not base:
            raise ValueError(
                "Configure an OpenAI-compatible endpoint with --api-base or LLM_API_BASE "
                "(for example, http://localhost:8000/v1)."
            )
        if base:
            url = urlsplit(base)
            if url.scheme not in ("http", "https") or not url.netloc:
                raise ValueError("LLM_API_BASE / --api-base must be an HTTP(S) API base URL.")
            if url.username or url.password or url.query or url.fragment:
                raise ValueError(
                    "Use an API base URL without credentials, query, or fragment; "
                    "set authentication with LLM_API_KEY."
                )
        if require_model and not selected_model:
            raise ValueError("Choose a vision model with --model or LLM_MODEL.")
        key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        return cls(base, selected_model, key or "not-required")


class VisionClient(Protocol):
    """Transport interface shared by page, crop, and grounded OCR."""

    async def complete(
        self, prompt: str, image_base64: str, *, model: str,
        timeout: float, max_tokens: int, temperature: float,
    ) -> str: ...

    async def list_models(self) -> list[str]: ...


class StructuredVisionClient(VisionClient, Protocol):
    """Optional capability; ordinary injected VisionClients remain supported."""

    async def complete_json(
        self, prompt: str, image_base64: str, *, schema: dict,
        model: str, timeout: float, max_tokens: int, temperature: float,
    ) -> str: ...


async def ensure_model_available(client: VisionClient, model: str) -> None:
    """Check advertised IDs, without assuming how the server loads weights."""
    available = await client.list_models()
    if not any(candidate.lower() == model.lower() for candidate in available):
        listing = ", ".join(available) or "(none)"
        raise ModelNotAvailableError(
            f"Model {model!r} is not advertised by the configured endpoint. "
            f"Available models: {listing}. Configure the server to serve this model "
            "or choose an advertised ID with --model. This prevents unintended model "
            "fallback. If the server does not advertise its usable models, pass "
            "--no-verify-model (or disable verification in the web UI)."
        )


def _image_mime_type(image_base64: str) -> str:
    """Match the data URL to the encoded bytes, including legacy PNG callers."""
    header = base64.b64decode(image_base64[:32])
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise ValueError("Vision inference requires a base64-encoded JPEG or PNG image.")


class OpenAICompatibleClient:
    """Adapter for the standard models and vision chat-completions endpoints.

    The SDK client is created lazily, reused for discovery and inference, and
    closed by aclose() or the async context manager. An optional sdk_client
    supports custom HTTP transports; it remains owned by the caller.
    """

    def __init__(self, config: LLMConfig, *, sdk_client=None, generation_options=None,
                 image_first: bool = False):
        self.config = config
        self._client = sdk_client
        self._owns_client = sdk_client is None
        self._structured_output_supported = True
        self.image_first = image_first
        # Standard, optional sampling settings. Request identity and budgets
        # remain controlled by complete(), including for injected SDK clients.
        self.generation_options = dict(generation_options or {})
        allowed = {"top_p", "presence_penalty", "frequency_penalty", "seed"}
        if self.generation_options.keys() - allowed:
            raise ValueError("Unsupported generation option; use standard sampling settings.")

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.config.api_base, api_key=self.config.api_key, timeout=30.0,
            )
        return self._client

    async def complete(
        self, prompt: str, image_base64: str, *, model: str,
        timeout: float, max_tokens: int, temperature: float,
    ) -> str:
        return await self._complete(
            prompt, image_base64, model=model, timeout=timeout,
            max_tokens=max_tokens, temperature=temperature,
        )

    async def complete_json(
        self, prompt: str, image_base64: str, *, schema: dict,
        model: str, timeout: float, max_tokens: int, temperature: float,
    ) -> str:
        deadline = time.monotonic() + timeout
        if self._structured_output_supported:
            try:
                return await self._complete(
                    prompt, image_base64, model=model, timeout=timeout,
                    max_tokens=max_tokens, temperature=temperature,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "ocr", "strict": True, "schema": schema,
                    }},
                )
            except _StructuredOutputUnsupported:
                self._structured_output_supported = False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMCallError("Vision request exceeded its deadline during format negotiation.")
        return await self._complete(
            prompt, image_base64, model=model, timeout=remaining,
            max_tokens=max_tokens, temperature=temperature,
        )

    async def _complete(
        self, prompt: str, image_base64: str, *, model: str,
        timeout: float, max_tokens: int, temperature: float, response_format=None,
    ) -> str:
        mime_type = _image_mime_type(image_base64)
        image_part = {"type": "image_url", "image_url": {
            "url": f"data:{mime_type};base64,{image_base64}",
        }}
        text_part = {"type": "text", "text": prompt}
        content = ([image_part, text_part] if self.image_first else [text_part, image_part])
        if not prompt:
            content = [image_part]
        try:
            response = await asyncio.wait_for(
                self._get_client().with_options(timeout=timeout, max_retries=0).chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    **self.generation_options,
                    **({"response_format": response_format} if response_format else {}),
                    messages=[{
                        "role": "user",
                        "content": content,
                    }],
                ), timeout=timeout,
            )
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raise IncompleteCompletionError(
                    f"Vision response reached the {max_tokens} token limit "
                    f"(model={model!r}); retry with smaller regions."
                )
            message = response.choices[0].message
            content = (message.content or "").strip()
            if not content and getattr(message, "reasoning_content", None):
                raise LLMCallError(
                    "The server returned reasoning without OCR content. Use the model's "
                    "official chat template and disable thinking for OCR-only models "
                    "such as LightOnOCR-3."
                )
            return content
        except asyncio.TimeoutError as exc:
            raise LLMCallError(
                f"Vision request exceeded its {timeout:g}s deadline (model={model!r})."
            ) from exc
        except IncompleteCompletionError:
            raise
        except Exception as exc:
            if (response_format and getattr(exc, "status_code", None) in (400, 422)
                    and any(term in str(exc).lower() for term in (
                        "response_format", "json_schema", "structured output", "grammar",
                    ))):
                raise _StructuredOutputUnsupported(str(exc)) from exc
            raise LLMCallError(
                f"Vision request failed at {self.config.api_base} (model={model!r}): "
                f"{type(exc).__name__}: {exc}. Check the endpoint, LLM_API_KEY, "
                "and the model's vision support. For context or timeout errors, "
                "try reducing --concurrency or --max-image-dim."
            ) from exc

    async def list_models(self) -> list[str]:
        try:
            page = await self._get_client().models.list()
            return [model.id for model in page.data]
        except Exception as exc:
            raise LLMCallError(
                f"Could not list models on {self.config.api_base}: "
                f"{type(exc).__name__}: {exc}. Check the endpoint and LLM_API_KEY. "
                "If the server lacks /models, use --no-verify-model and specify --model."
            ) from exc

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
