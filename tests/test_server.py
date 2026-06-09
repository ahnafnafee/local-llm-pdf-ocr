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
            else:
                Path(output_path).write_bytes(b"%PDF-1.4\nstub\n%%EOF\n")
            return {0: ["stub line"]}

    import pdf_ocr.server as server_mod
    monkeypatch.setattr(server_mod, "OCRPipeline", _StubPipeline)
    # The HybridAligner / OCRProcessor / PDFHandler are instantiated by
    # the endpoint but never used by the stub — we still need them to
    # construct without side effects, which they do on import.

    return TestClient(server_mod.app)


def _post_process(client, *, format: str | None, filename: str = "scan.pdf"):
    files = {"file": (filename, io.BytesIO(b"%PDF-1.4\ntest\n%%EOF\n"), "application/pdf")}
    data = {"client_id": "test-client"}
    if format is not None:
        data["format"] = format
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
