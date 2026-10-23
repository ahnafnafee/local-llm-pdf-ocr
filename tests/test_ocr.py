"""Unit tests for OCRProcessor prompt/parsing concerns (no LLM calls)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from pdf_ocr.core.ocr import (
    CROP_PROMPT,
    OLMOCR_PAGE_PROMPT,
    LLMCallError,
    ModelNotLoadedError,
    OCRProcessor,
    _strip_runaway_repetition,
    _strip_yaml_front_matter,
)


@pytest.mark.parametrize("field", ["is_table", "is_diagram"])
async def test_front_matter_flags_non_linear_layout(field):
    from unittest.mock import AsyncMock

    processor = OCRProcessor(model="vision", client=SimpleNamespace(complete=AsyncMock(
        return_value=f"---\n{field}: True\n---\nrecognized line",
    )))
    lines = await processor.perform_ocr("image")
    assert lines == ["recognized line"]
    assert lines.requires_region_ocr


async def test_body_text_does_not_set_a_layout_hint():
    from unittest.mock import AsyncMock

    processor = OCRProcessor(model="vision", client=SimpleNamespace(complete=AsyncMock(
        return_value="a technical example\nis_table: True",
    )))
    assert not (await processor.perform_ocr("image")).requires_region_ocr


async def test_crop_retry_discards_repeatedly_truncated_text():
    from unittest.mock import AsyncMock
    from pdf_ocr.inference import IncompleteCompletionError

    complete = AsyncMock(side_effect=IncompleteCompletionError("token limit"))
    processor = OCRProcessor(model="vision", client=SimpleNamespace(complete=complete))
    assert await processor.perform_ocr_on_crop("image") == ""
    assert complete.await_count == 2


class TestYAMLFrontMatter:
    def test_strips_canonical_olmocr_response(self):
        response = (
            "---\n"
            "primary_language: en\n"
            "is_rotation_valid: true\n"
            "rotation_correction: 0\n"
            "is_table: false\n"
            "is_diagram: false\n"
            "---\n"
            "# Document Title\n\nBody paragraph text.\n"
        )
        body = _strip_yaml_front_matter(response)
        assert "primary_language" not in body
        assert "Body paragraph text" in body
        assert body.startswith("# Document Title")

    def test_passthrough_when_no_front_matter(self):
        response = "Plain text response with no YAML.\nSecond line."
        assert _strip_yaml_front_matter(response) == response

    def test_malformed_front_matter_returned_unchanged(self):
        # Opening fence but no closing fence — don't guess; preserve text.
        response = "---\nprimary_language: en\nbody text with no closing fence"
        assert _strip_yaml_front_matter(response) == response

    def test_leading_whitespace_handled(self):
        response = "  \n---\nkey: val\n---\nbody"
        out = _strip_yaml_front_matter(response)
        assert out == "body"


class TestStripRunawayRepetition:
    def test_passes_short_unique_lines_through(self):
        lines = ["a", "b", "c", "d"]
        assert _strip_runaway_repetition(lines) == lines

    def test_admits_legitimate_table_repetition(self):
        # HTML table from OlmOCR's "tables to HTML" instruction — many <tr>
        # tags are expected and shouldn't be clipped.
        lines = (
            ["<table>"]
            + ["<tr>", "<td>x</td>", "</tr>"] * 10
            + ["</table>"]
        )
        out = _strip_runaway_repetition(lines, max_repeat=20)
        assert out == lines, "10x table repetition must survive"

    def test_clips_runaway_repetition(self):
        # Model stuck in a loop emitting the same line 100 times — should
        # leave only the first 20 occurrences.
        lines = ["unique header"] + ["LOOP"] * 100 + ["unique footer"]
        out = _strip_runaway_repetition(lines, max_repeat=20)
        assert out.count("LOOP") == 20
        assert out[0] == "unique header"
        # The repetition cap applies only to the repeated "LOOP" line;
        # unrelated later lines (the footer) are still preserved.
        assert "unique footer" in out

    def test_clips_multiple_runaway_lines_independently(self):
        lines = ["A"] * 50 + ["B"] * 50
        out = _strip_runaway_repetition(lines, max_repeat=10)
        assert out.count("A") == 10
        assert out.count("B") == 10

    def test_empty_input(self):
        assert _strip_runaway_repetition([]) == []


class TestHallucinationFilter:
    def test_pangram_response_treated_as_blank(self):
        """OlmOCR-2 falls back to 'The quick brown fox...' on blank/unreadable
        crops. perform_ocr_on_crop must drop those instead of placing them
        in the searchable text layer."""
        import asyncio
        from pdf_ocr.core.ocr import OCRProcessor

        ocr = OCRProcessor(model="vision", client=SimpleNamespace())

        async def _fake_pangram(*a, **kw):
            return "The quick brown fox jumps over the lazy dog."

        ocr._chat = _fake_pangram
        ocr.CROP_TIMEOUT_S = 60.0
        ocr.CROP_MAX_TOKENS = 256
        result = asyncio.run(ocr.perform_ocr_on_crop("ignored"))
        assert result == ""

    def test_normal_crop_response_passes_through(self):
        import asyncio
        from pdf_ocr.core.ocr import OCRProcessor

        ocr = OCRProcessor(model="vision", client=SimpleNamespace())

        async def _fake(*a, **kw):
            return "real handwritten content"

        ocr._chat = _fake
        ocr.CROP_TIMEOUT_S = 60.0
        ocr.CROP_MAX_TOKENS = 256
        assert asyncio.run(ocr.perform_ocr_on_crop("ignored")) == "real handwritten content"

    def test_real_text_containing_pangram_is_preserved(self):
        # A document that legitimately contains the pangram (e.g. a typing
        # exercise) must NOT be silently dropped. The filter only fires
        # when the response IS the pangram, not when it merely contains it.
        import asyncio
        from pdf_ocr.core.ocr import OCRProcessor

        ocr = OCRProcessor(model="vision", client=SimpleNamespace())

        sentence = (
            "Practice typing: The quick brown fox jumps over the lazy dog. "
            "Repeat ten times."
        )

        async def _fake(*a, **kw):
            return sentence

        ocr._chat = _fake
        ocr.CROP_TIMEOUT_S = 60.0
        ocr.CROP_MAX_TOKENS = 256
        assert asyncio.run(ocr.perform_ocr_on_crop("ignored")) == sentence

    def test_pangram_with_quotes_or_trailing_punct_still_dropped(self):
        # OlmOCR sometimes wraps the pangram in quotes or appends ! / ? —
        # normalization must still recognise it as the fallback.
        import asyncio
        from pdf_ocr.core.ocr import OCRProcessor

        def _make_fake(response: str):
            async def _fake(*a, **kw):
                return response
            return _fake

        for variant in (
            'The quick brown fox jumps over the lazy dog!',
            '"The quick brown fox jumps over the lazy dog."',
            'the quick brown fox jumps over the lazy dog',
        ):
            ocr = OCRProcessor(model="vision", client=SimpleNamespace())
            ocr._chat = _make_fake(variant)
            ocr.CROP_TIMEOUT_S = 60.0
            ocr.CROP_MAX_TOKENS = 256
            assert asyncio.run(ocr.perform_ocr_on_crop("ignored")) == "", (
                f"variant {variant!r} should be dropped"
            )


def _fake_models_client(model_ids=None, raise_exc=None):
    """Build a stand-in for AsyncOpenAI exposing only `client.models.list()`.

    Mirrors the SDK shape: ``await client.models.list()`` returns an object
    with a ``.data`` attribute that's a list of objects each with an ``.id``.
    """
    async def _list():
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(
            data=[SimpleNamespace(id=m) for m in (model_ids or [])]
        )
    return SimpleNamespace(models=SimpleNamespace(list=_list))


def _make_ocr_with_fake_client(model: str, fake_client) -> OCRProcessor:
    """Exercise the real adapter over an injected SDK-shaped test client."""
    from pdf_ocr.inference import LLMConfig, OpenAICompatibleClient

    adapter = OpenAICompatibleClient(
        LLMConfig("http://inference.test/v1", model), sdk_client=fake_client,
    )
    return OCRProcessor(model=model, client=adapter)


class TestEnsureModelLoaded:
    """Pre-flight check that the requested model is loaded on the LLM
    server. LM Studio silently falls back to whatever is loaded on
    mismatch, so without this check users get bad OCR with no error
    (issue #7)."""

    def test_passes_when_model_in_loaded_list(self):
        ocr = _make_ocr_with_fake_client(
            "qwen/qwen3-vl-8b",
            _fake_models_client(["qwen/qwen3-vl-8b", "allenai/olmocr-2-7b"]),
        )
        # No raise — exact match found.
        asyncio.run(ocr.ensure_model_loaded())

    def test_passes_case_insensitive(self):
        # User passes "Qwen/Qwen3-VL-8B" but server returns "qwen/qwen3-vl-8b"
        # — same model file, just shifted case. Don't make the user fight casing.
        ocr = _make_ocr_with_fake_client(
            "Qwen/Qwen3-VL-8B",
            _fake_models_client(["qwen/qwen3-vl-8b"]),
        )
        asyncio.run(ocr.ensure_model_loaded())

    def test_raises_with_helpful_message_on_mismatch(self):
        # The exact scenario from the issue: user passed qwen3-vl-8b but
        # LM Studio has olmocr loaded. Must surface this loudly.
        ocr = _make_ocr_with_fake_client(
            "qwen/qwen3-vl-8b",
            _fake_models_client(["allenai_olmocr-2-7b-1025"]),
        )
        with pytest.raises(ModelNotLoadedError) as exc_info:
            asyncio.run(ocr.ensure_model_loaded())

        msg = str(exc_info.value)
        # Must name the requested model (so the user knows what they asked for)…
        assert "qwen/qwen3-vl-8b" in msg
        # …and what the server actually has loaded (so they can either
        # change --model or load the right one)…
        assert "allenai_olmocr-2-7b-1025" in msg
        # …and tell them about the escape hatch for non-LM-Studio servers.
        assert "--no-verify-model" in msg
        # …and explain WHY this matters (silent fallback) so they don't
        # treat the check as a bug to disable and forget.
        assert "silently" in msg.lower() or "fallback" in msg.lower()

    def test_raises_with_none_listing_when_no_models_loaded(self):
        # LM Studio with no model loaded at all. The error message
        # should still be informative, not say "Loaded models: " followed
        # by nothing (which reads like a parse error).
        ocr = _make_ocr_with_fake_client(
            "qwen/qwen3-vl-8b",
            _fake_models_client([]),
        )
        with pytest.raises(ModelNotLoadedError) as exc_info:
            asyncio.run(ocr.ensure_model_loaded())
        assert "(none)" in str(exc_info.value)

    def test_subclass_of_llm_call_error(self):
        # Existing callers of LLMCallError (e.g. CLI's generic
        # except-and-print path) must continue to catch this without
        # special-casing.
        assert issubclass(ModelNotLoadedError, LLMCallError)

    def test_server_failure_wrapped_as_llm_call_error(self):
        # If /v1/models fails (server down, wrong endpoint, auth error)
        # surface a single-paragraph LLMCallError rather than the bare
        # ConnectionError stack — match the diagnostic style of _chat.
        ocr = _make_ocr_with_fake_client(
            "qwen/qwen3-vl-8b",
            _fake_models_client(raise_exc=ConnectionError("connection refused")),
        )
        with pytest.raises(LLMCallError) as exc_info:
            asyncio.run(ocr.ensure_model_loaded())
        # Must point the user at the server (where to look) and at the
        # opt-out flag (how to bypass for non-conforming servers).
        assert "http://inference.test/v1" in str(exc_info.value)
        assert "--no-verify-model" in str(exc_info.value)


class TestPromptConstants:
    def test_olmocr_prompt_is_canonical(self):
        # Guard against accidental prompt drift — this string was lifted
        # verbatim from allenai/olmocr. If you change it, expect worse OCR.
        assert "Attached is one page of a document" in OLMOCR_PAGE_PROMPT
        assert "Convert equations to LateX and tables to HTML" in OLMOCR_PAGE_PROMPT
        assert "front matter section" in OLMOCR_PAGE_PROMPT

    def test_crop_prompt_is_minimal(self):
        # For crops we want plain text — no metadata/markdown ceremony.
        assert "no markdown" in CROP_PROMPT.lower()
        assert "plain text" in CROP_PROMPT.lower()
