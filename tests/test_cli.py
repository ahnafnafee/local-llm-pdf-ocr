"""
Tests for `pdf_ocr.cli` argument parsing and output-path resolution.

Heavy LLM / pipeline behavior is tested elsewhere; this file is about
the CLI surface: the `--format` flag, `resolve_output_path` decisions,
and how explicit output paths interact with the format flag.
"""

from __future__ import annotations

import sys

import pytest

from pdf_ocr.cli import build_parser, format_duration, resolve_output_path


def test_grounded_empty_output_exits_with_actionable_error(monkeypatch, capsys, tmp_path):
    from unittest.mock import AsyncMock

    from pdf_ocr.cli import main
    from pdf_ocr.core.grounded import GroundedResponse, PromptedGroundedOCR

    monkeypatch.setattr(
        PromptedGroundedOCR, "ocr_document",
        AsyncMock(return_value=GroundedResponse(blocks=[], page_sizes=[(100, 100)])),
    )
    output = tmp_path / "out.pdf"
    monkeypatch.setattr(sys, "argv", [
        "local-llm-pdf-ocr", "in.pdf", str(output),
        "--grounded", "--no-verify-model",
    ])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = capsys.readouterr().err
    assert "Grounded OCR" in printed
    assert "page(s) 1" in printed
    assert "hybrid" in printed
    assert "Done!" not in printed
    assert not output.exists()


def test_invalid_endpoint_is_reported_even_in_quiet_mode(monkeypatch, capsys):
    from pdf_ocr.cli import main

    monkeypatch.setenv("LLM_API_BASE", "not-a-url")
    monkeypatch.setattr(sys, "argv", ["local-llm-pdf-ocr", "in.pdf", "--text-only", "--quiet"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "--api-base" in capsys.readouterr().err


@pytest.mark.parametrize("mode, expected_model, response", [
    ("--text-only", "lightonai/LightOnOCR-3-4B", "recognized"),
    ("--grounded", "lightonai/LightOnOCR-3-4B", '![text](0,0,100,50)\nrecognized'),
])
def test_cli_uses_lighton_default_without_configuration(monkeypatch, tmp_path, mode, expected_model, response):
    import fitz

    from pdf_ocr.cli import main
    from pdf_ocr.inference import OpenAICompatibleClient

    monkeypatch.delenv("LLM_API_BASE")
    monkeypatch.delenv("LLM_MODEL")
    source = tmp_path / "input.pdf"
    with fitz.open() as doc:
        doc.new_page(width=72, height=72)
        doc.save(source)
    requests = []

    async def complete(self, prompt, image_base64, **kwargs):
        requests.append((self.config.api_base, kwargs["model"]))
        return response

    monkeypatch.setattr(OpenAICompatibleClient, "complete", complete)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_json", complete)
    output = tmp_path / "out.md"
    monkeypatch.setattr(sys, "argv", [
        "local-llm-pdf-ocr", str(source), str(output), mode, "--no-verify-model",
    ])
    main()
    assert requests == [("http://localhost:1234/v1", expected_model)]
    assert "recognized" in output.read_text(encoding="utf-8")


def test_grounded_cli_selects_pages_and_forwards_rendering_options(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    import fitz

    from pdf_ocr.cli import main
    from pdf_ocr.core.grounded import PromptedGroundedOCR
    from pdf_ocr.inference import OpenAICompatibleClient

    source = tmp_path / "input.pdf"
    with fitz.open() as doc:
        for _ in range(5):
            doc.new_page(width=72, height=72)
        doc.save(source)
    complete = AsyncMock(return_value='[{"bbox_2d":[0,0,100,50],"content":"recognized"}]')
    monkeypatch.setattr(OpenAICompatibleClient, "complete", complete)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_json", complete)
    close = AsyncMock()
    monkeypatch.setattr(OpenAICompatibleClient, "aclose", close)
    observed = []
    original = PromptedGroundedOCR.ocr_document

    async def record(self, *args, **kwargs):
        observed.append((self.dpi, self.concurrency, kwargs["pages"]))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(PromptedGroundedOCR, "ocr_document", record)
    output = tmp_path / "out.md"
    monkeypatch.setattr(sys, "argv", [
        "local-llm-pdf-ocr", str(source), str(output), "--grounded",
        "--no-verify-model", "--pages", "2,4", "--dpi", "72", "--concurrency", "3",
    ])
    main()
    assert observed == [(72, 3, "2,4")]
    assert complete.await_count == 2
    close.assert_awaited_once()
    text = output.read_text(encoding="utf-8")
    assert "## Page 2" in text and "## Page 4" in text
    assert "## Page 1" not in text and "## Page 3" not in text


class TestFormatDuration:
    def test_sub_minute_shows_seconds(self):
        assert format_duration(0.0) == "0.0s"
        assert format_duration(12.34) == "12.3s"
        assert format_duration(59.9) == "59.9s"

    def test_over_a_minute_shows_minutes_and_seconds(self):
        assert format_duration(60) == "1m 00s"
        assert format_duration(83) == "1m 23s"
        assert format_duration(3661) == "61m 01s"


class TestResolveOutputPath:
    def test_no_output_no_format_defaults_to_pdf(self):
        out = resolve_output_path("/tmp/in.pdf", None)
        assert out.endswith("in_ocr.pdf")

    def test_no_output_with_format_uses_format_suffix(self):
        assert resolve_output_path("/tmp/in.pdf", None, "html").endswith("in_ocr.html")
        assert resolve_output_path("/tmp/in.pdf", None, "md").endswith("in_ocr.md")
        assert resolve_output_path("/tmp/in.pdf", None, "pdf").endswith("in_ocr.pdf")
        assert resolve_output_path("/tmp/in.pdf", None, "txt").endswith("in_ocr.txt")

    def test_explicit_output_wins_over_format(self):
        # User provides an explicit `.md` output but also `--format html`.
        # Explicit output wins so the actual writer ends up Markdown.
        out = resolve_output_path("/tmp/in.pdf", "out.md", "html")
        assert out == "out.md"

    def test_explicit_output_no_format(self):
        out = resolve_output_path("/tmp/in.pdf", "out.html", None)
        assert out == "out.html"

    def test_image_input_with_format(self):
        out = resolve_output_path("/tmp/scan.png", None, "html")
        assert out.endswith("scan_ocr.html")


class TestParserFormatFlag:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_format_flag_accepts_pdf_html_md_txt(self):
        for fmt in ("pdf", "html", "md", "txt"):
            ns = self._parse("--format", fmt)
            assert ns.format == fmt

    def test_format_flag_default_is_none(self):
        # When --format isn't given, the namespace value is None — the
        # caller (resolve_output_path) treats that as "default to pdf".
        ns = self._parse()
        assert ns.format is None

    def test_format_flag_rejects_unknown_value(self, capsys):
        with pytest.raises(SystemExit):
            self._parse("--format", "docx")
        # argparse prints to stderr; either captured stream is fine.
        err = capsys.readouterr().err
        assert "invalid choice" in err

    def test_explicit_output_path_still_parsed(self):
        ns = self._parse("custom.html")
        assert ns.input_pdf == "input.pdf"
        assert ns.output_pdf == "custom.html"
        assert ns.format is None


class TestTextOnlyFlag:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_text_only_defaults_off(self):
        assert self._parse().text_only is False

    def test_text_only_flag_sets_true(self):
        assert self._parse("--text-only").text_only is True


class TestHtmlImageFlags:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_html_inline_images_defaults_to_false(self):
        ns = self._parse()
        assert ns.html_inline_images is False

    def test_html_inline_images_flag_sets_true(self):
        ns = self._parse("--html-inline-images")
        assert ns.html_inline_images is True

    def test_html_mode_default_is_scaled(self):
        # milahu requested this default — letter-spacing was unreadable
        # when negative spacing made characters overlap.
        ns = self._parse()
        assert ns.html_mode == "scaled"

    def test_html_mode_letter_spacing_still_selectable(self):
        ns = self._parse("--html-mode", "letter-spacing")
        assert ns.html_mode == "letter-spacing"


class TestQuadFidelityFlags:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["input.pdf", *argv])

    def test_min_box_confidence_default_is_none(self):
        # Filtering detector boxes changes recall — it must be opt-in.
        assert self._parse().min_box_confidence is None

    def test_min_box_confidence_parses_float(self):
        ns = self._parse("--min-box-confidence", "0.25")
        assert ns.min_box_confidence == 0.25

    def test_html_hover_text_defaults_off(self):
        # The text layer stays invisible unless explicitly opted in.
        assert self._parse().html_hover_text is False

    def test_html_hover_text_opt_in(self):
        assert self._parse("--html-hover-text").html_hover_text is True
