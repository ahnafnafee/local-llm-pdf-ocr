"""Server configuration and the transport boundary for vision inference.

OCR components depend on VisionClient, not an SDK or a server application.
OpenAICompatibleClient is the default adapter; injected clients remain owned
by their caller and can implement the same small async interface.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()


class LLMCallError(RuntimeError):
    """A model discovery or inference request failed."""


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
    ) -> LLMConfig:
        """Explicit values override LLM_* environment variables and .env.

        No server address or model ID is guessed. Discovery needs only an
        endpoint; an injected VisionClient needs only a model.
        """
        base = (api_base or os.getenv("LLM_API_BASE", "")).strip().rstrip("/")
        selected_model = (model or os.getenv("LLM_MODEL", "")).strip()
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

    def __init__(self, config: LLMConfig, *, sdk_client=None):
        self.config = config
        self._client = sdk_client
        self._owns_client = sdk_client is None

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
        mime_type = _image_mime_type(image_base64)
        try:
            response = await self._get_client().with_options(timeout=timeout).chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                        }},
                    ],
                }],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMCallError(
                f"Vision request failed at {self.config.api_base} (model={model!r}): "
                f"{type(exc).__name__}: {exc}. Check the endpoint, LLM_API_KEY, "
                "and the model's vision support. For context or timeout errors, "
                "try reducing --max-image-dim."
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
