"""
OCRProcessor - LLM-based OCR processing.

Uses a local vision LLM (OlmOCR via LM Studio by default; any OpenAI-compatible
endpoint works, including GLM OCR via Ollama — set LLM_API_BASE/LLM_MODEL or
pass --api-base/--model).
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class LLMCallError(RuntimeError):
    """Raised when a call to the local LLM OCR endpoint fails.

    Wraps the underlying exception (connection refused, model not loaded,
    timeout, auth, ...) with a message that names the api-base and model
    so the user can diagnose without digging through a stack trace.
    """


# Canonical OlmOCR-2 prompt (the model was RL-trained on this exact string).
# Source: github.com/allenai/olmocr olmocr/prompts/prompts.py
# :func:`build_no_anchoring_v4_yaml_prompt`.
OLMOCR_PAGE_PROMPT = (
    "Attached is one page of a document that you must process. Just return "
    "the plain text representation of this document as if you were reading it "
    "naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following "
    "markdown syntax ![Alt text describing the contents of the figure]"
    "(page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top "
    "specifying values for the primary_language, is_rotation_valid, "
    "rotation_correction, is_table, and is_diagram parameters."
)

# Prompt for cropped box regions — we want raw text only, no metadata
# (the YAML front matter is nonsensical for a single line/region).
CROP_PROMPT = (
    "Transcribe the text in this image exactly as it appears. "
    "Return only the plain text on a single line when possible, "
    "with no metadata, no markdown, no formatting, no explanation."
)


class OCRProcessor:
    """LLM-based OCR processor over an OpenAI-compatible async client."""

    def __init__(self, api_base: str | None = None, model: str | None = None):
        self.api_base = api_base or os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
        self.model = model or os.getenv("LLM_MODEL", "allenai/olmocr-2-7b")
        self.client = AsyncOpenAI(base_url=self.api_base, api_key="lm-studio")

    async def perform_ocr(self, image_base64: str) -> list[str]:
        """
        OCR a full page image. Returns a list of non-empty lines in reading order.

        YAML front matter emitted by OlmOCR (rotation/language/is_table flags)
        is stripped before returning.
        """
        text = await self._chat(OLMOCR_PAGE_PROMPT, image_base64)
        if not text:
            return []
        body = _strip_yaml_front_matter(text)
        return [line.strip() for line in body.split("\n") if line.strip()]

    async def perform_ocr_on_crop(self, image_base64: str) -> str:
        """
        OCR a single cropped box region. Returns a single whitespace-joined
        string (the crop is small, so we don't try to preserve line structure).
        """
        text = await self._chat(CROP_PROMPT, image_base64)
        if not text:
            return ""
        body = _strip_yaml_front_matter(text)
        return " ".join(line.strip() for line in body.split("\n") if line.strip())

    async def _chat(self, prompt: str, image_base64: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise LLMCallError(
                f"LLM OCR call failed against {self.api_base} "
                f"(model={self.model!r}): {type(e).__name__}: {e}\n"
                f"  - Is your local LLM server (LM Studio / Ollama / vLLM) running at "
                f"{self.api_base}?\n"
                f"  - Is model {self.model!r} loaded and serving vision inputs?\n"
                f"  - Override with LLM_API_BASE / LLM_MODEL in .env, "
                f"or pass --api-base / --model."
            ) from e


def _strip_yaml_front_matter(text: str) -> str:
    """
    If the response begins with a YAML front matter block (--- ... ---),
    return the body after it. Otherwise return the input unchanged.
    Robust to models that ignore the front-matter instruction.
    """
    t = text.lstrip()
    if not t.startswith("---"):
        return text
    # Find the closing fence on its own line, after the opening fence.
    rest = t[3:]
    close_idx = rest.find("\n---")
    if close_idx == -1:
        return text  # malformed; return as-is
    body = rest[close_idx + len("\n---"):]
    # Trim the newline directly after the closing fence.
    return body.lstrip("\n").strip()
