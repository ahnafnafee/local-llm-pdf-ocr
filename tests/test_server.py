"""
Tests for the FastAPI server's format dispatch and HTTP surface.

The full pipeline is stubbed to avoid loading Surya / hitting an LLM —
we patch `OCRPipeline` so `/process` returns a tiny synthetic output
file in the requested format. That's enough to verify:

  * the `format` form field is accepted and validated
  * the response Content-Type / Content-Disposition track the format
  * unsupported formats return HTTP 400
  * the default format (no `format` field) is still PDF

A small smoke test also exercises the `/` and `/text/{job_id}` endpoints
to keep regressions visible.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


_HAS_FASTAPI = True
try:
    from fastapi.testclient import TestClient  # noqa: F401
except ImportError:  # pragma: no cover - optional dep
    _HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(
    not _HAS_FASTAPI,
    reason="fastapi extras not installed (uv sync --extra web)",
)


@pytest.fixture
def client(monkeypatch):
    """A TestClient with `OCRPipeline` stubbed to skip Surya / LLM."""
    from fastapi.testclient import TestClient

    # Stub OCRPipeline before importing pdf_ocr.server (which imports it).
    class _StubPipeline:
        def __init__(self, *_, **kwargs):
            self._writer = kwargs.get("output_writer")
            self.ocr_processor = self  # has ensure_model_loaded

        async def ensure_model_loaded(self):
            return None

        async def run(self, input_path, output_path, **_):
            # Mimic the real writer's behavior just enough to make the
            # /process response have a non-empty body of the right type.
            ext = Path(output_path).suffix.lower()
            if ext == ".html":
                Path(output_path).write_text(
                    "<!doctype html><html><body>stub</body></html>",
                    encoding="utf-8",
                )
            elif ext in (".md", ".markdown"):
                Path(output_path).write_text("# stub\n\nbody\n", encoding="utf-8")
            elif ext == ".txt":
                Path(output_path).write_text("stub text body\n", encoding="utf-8")
            else:
                Path(output_path).write_bytes(b"%PDF-1.4\nstub\n%%EOF\n")
            return {0: ["stub line"]}

    import pdf_ocr.server as server_mod
    monkeypatch.setattr(server_mod, "OCRPipeline", _StubPipeline)
    # The HybridAligner / OCRProcessor / PDFHandler are instantiated by
    # the endpoint but never used by the stub — we still need them to
    # construct without side effects, which they do on import.

    return TestClient(server_mod.app)


def _post_process(
    client, *, format: str | None, filename: str = "scan.pdf", text_only: bool | None = None,
):
    files = {"file": (filename, io.BytesIO(b"%PDF-1.4\ntest\n%%EOF\n"), "application/pdf")}
    data = {"client_id": "test-client"}
    if format is not None:
        data["format"] = format
    if text_only is not None:
        data["text_only"] = "true" if text_only else "false"
    return client.post("/process", files=files, data=data)


class TestProcessFormatDispatch:
    def test_default_format_is_pdf(self, client):
        resp = _post_process(client, format=None)
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("application/pdf")
        cd = resp.headers.get("content-disposition", "")
        assert ".pdf" in cd
        assert resp.content.startswith(b"%PDF-")

    def test_explicit_pdf_format(self, client):
        resp = _post_process(client, format="pdf")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("application/pdf")

    def test_html_format(self, client):
        resp = _post_process(client, format="html")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/html")
        cd = resp.headers.get("content-disposition", "")
        assert ".html" in cd
        assert b"<!doctype html>" in resp.content

    def test_md_format(self, client):
        resp = _post_process(client, format="md")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/markdown")
        cd = resp.headers.get("content-disposition", "")
        assert ".md" in cd
        assert resp.content.startswith(b"# stub")

    def test_txt_format(self, client):
        resp = _post_process(client, format="txt")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/plain")
        cd = resp.headers.get("content-disposition", "")
        assert ".txt" in cd
        assert resp.content.startswith(b"stub text")

    def test_text_only_returns_plain_text(self, client):
        # text_only forces a .txt dump even with the default (pdf) format.
        resp = _post_process(client, format=None, text_only=True)
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/plain")
        cd = resp.headers.get("content-disposition", "")
        assert "ocr_scan.txt" in cd

    def test_text_only_overrides_chosen_format(self, client):
        # Even if the dropdown said html, text_only wins → .txt.
        resp = _post_process(client, format="html", text_only=True)
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/plain")
        assert ".txt" in resp.headers.get("content-disposition", "")

    def test_unknown_format_returns_400(self, client):
        resp = _post_process(client, format="docx")
        assert resp.status_code == 400, resp.text
        assert "unsupported format" in resp.json().get("error", "")

    def test_download_filename_uses_chosen_suffix(self, client):
        # The original file is .pdf; with format=html, the download
        # should be `ocr_scan.html` (NOT `ocr_scan.pdf.html`).
        resp = _post_process(client, format="html", filename="scan.pdf")
        cd = resp.headers.get("content-disposition", "")
        assert "ocr_scan.html" in cd
        assert "ocr_scan.pdf.html" not in cd


class TestUnchangedRoutes:
    def test_index_returns_static_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"].lower()

    def test_text_endpoint_returns_404_for_unknown_job(self, client):
        resp = client.get("/text/no-such-job")
        assert resp.status_code == 404


def _post(client, *, filename: str = "scan.pdf", content_type: str = "application/pdf", body: bytes = b"%PDF-1.4\ntest\n%%EOF\n", **fields):
    """POST /process with arbitrary option fields (all stringified)."""
    files = {"file": (filename, io.BytesIO(body), content_type)}
    data = {"client_id": "test-client"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post("/process", files=files, data=data)


class TestModelsEndpoint:
    def test_lists_loaded_models(self, client, monkeypatch):
        import pdf_ocr.core.ocr as ocr_mod

        async def fake_list(_client, _base):
            return ["allenai/olmocr-2-7b", "qwen/qwen3-vl-8b"]

        monkeypatch.setattr(ocr_mod, "_list_loaded_model_ids", fake_list)
        resp = client.get("/models")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["models"] == ["allenai/olmocr-2-7b", "qwen/qwen3-vl-8b"]
        assert "endpoint" in body

    def test_fails_soft_when_endpoint_unreachable(self, client, monkeypatch):
        import pdf_ocr.core.ocr as ocr_mod

        async def boom(_client, _base):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(ocr_mod, "_list_loaded_model_ids", boom)
        resp = client.get("/models?api_base=http://localhost:9/v1")
        # Fails soft: 200 with empty list + error so the UI degrades to
        # manual model entry instead of breaking.
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["models"] == []
        assert "error" in body
        assert body["endpoint"] == "http://localhost:9/v1"


class TestProcessOptions:
    def test_engine_text_dumps_txt(self, client):
        resp = _post(client, engine="text")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/plain")
        assert ".txt" in resp.headers.get("content-disposition", "")

    def test_engine_grounded_returns_pdf(self, client):
        resp = _post(client, engine="grounded", format="pdf")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("application/pdf")

    def test_invalid_engine_400(self, client):
        resp = _post(client, engine="magic")
        assert resp.status_code == 400
        assert "engine" in resp.json().get("error", "")

    def test_invalid_dense_mode_400(self, client):
        resp = _post(client, dense_mode="sometimes")
        assert resp.status_code == 400
        assert "dense_mode" in resp.json().get("error", "")

    def test_invalid_html_mode_400(self, client):
        resp = _post(client, format="html", html_mode="bogus")
        assert resp.status_code == 400
        assert "html_mode" in resp.json().get("error", "")

    def test_bad_min_confidence_400(self, client):
        resp = _post(client, min_box_confidence="high")
        assert resp.status_code == 400
        assert "min_box_confidence" in resp.json().get("error", "")

    def test_image_input_accepted(self, client):
        # A .png upload must not 400 on an unexpected suffix, and the
        # temp file keeps its extension so PDFHandler routes it as an image.
        resp = _post(
            client, engine="text", filename="scan.png",
            content_type="image/png", body=b"\x89PNG\r\n\x1a\n stub",
        )
        assert resp.status_code == 200, resp.text
        assert ".txt" in resp.headers.get("content-disposition", "")


@pytest.fixture
def recorder(monkeypatch):
    """A client whose pipeline records the kwargs `/process` forwards."""
    from fastapi.testclient import TestClient

    calls: dict = {}

    class _Rec:
        def __init__(self, *_, **kwargs):
            self.ocr_processor = self  # verify uses this (no grounded_backend attr)
            calls["init"] = kwargs

        async def ensure_model_loaded(self):
            return None

        async def run(self, input_path, output_path, **kw):
            calls["run"] = kw
            p = Path(output_path)
            if p.suffix.lower() == ".txt":
                p.write_text("t", encoding="utf-8")
            else:
                p.write_bytes(b"%PDF-1.4\n%%EOF\n")
            return {0: ["x"]}

    import pdf_ocr.server as server_mod

    monkeypatch.setattr(server_mod, "OCRPipeline", _Rec)

    async def _fake_aligner():
        return object()

    # Keep the hybrid path from loading Surya in this unit test.
    monkeypatch.setattr(server_mod, "_get_aligner", _fake_aligner)
    return TestClient(server_mod.app), calls


class TestOptionForwarding:
    def test_hybrid_forwards_tuning_knobs(self, recorder):
        client, calls = recorder
        resp = _post(
            client, engine="hybrid", model="my-model", dpi=300,
            refine="false", min_box_confidence="0.3", dense_mode="always",
            concurrency=5, preprocess="never",
        )
        assert resp.status_code == 200, resp.text
        run = calls["run"]
        assert run["dpi"] == 300
        assert run["refine"] is False
        assert run["min_box_confidence"] == 0.3
        assert run["dense_mode"] == "always"
        assert run["preprocess"] == "never"
        assert run["concurrency"] == 5
        assert run["text_only"] is False
        # per-request model override reached the OCRProcessor
        assert calls["init"]["ocr_processor"].model == "my-model"

    def test_text_engine_sets_text_only(self, recorder):
        client, calls = recorder
        resp = _post(client, engine="text", model="foo")
        assert resp.status_code == 200, resp.text
        assert calls["run"]["text_only"] is True
        assert calls["init"]["ocr_processor"].model == "foo"

    def test_dpi_clamped_server_side(self, recorder):
        client, calls = recorder
        # 99999 DPI is nonsense; the server clamps to the 600 ceiling.
        resp = _post(client, engine="hybrid", dpi=99999)
        assert resp.status_code == 200, resp.text
        assert calls["run"]["dpi"] == 600
