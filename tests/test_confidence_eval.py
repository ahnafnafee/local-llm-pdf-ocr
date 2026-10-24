"""Validate the live evaluator's exit status and connection ownership offline."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/confidence_eval.py"
    spec = importlib.util.spec_from_file_location("confidence_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "JOBS", [("handwritten.pdf", "ground_truth_handwritten.json")])
    monkeypatch.setattr(module, "BASELINES_DIR", tmp_path / "baselines")
    monkeypatch.setattr(module, "CHECKS_DIR", tmp_path / "checks")
    monkeypatch.setattr(module, "HISTORY_CSV", tmp_path / "history.csv")
    monkeypatch.setattr(module, "render_report", Mock())
    return module


@pytest.mark.parametrize("mode", ["hybrid", "grounded"])
@pytest.mark.parametrize("failure", ["empty", "exception"])
@pytest.mark.parametrize("update_baselines", [False, True])
async def test_failed_ocr_exits_nonzero_without_writing_baselines(
    evaluator, monkeypatch, mode, failure, update_baselines,
):
    runner = AsyncMock(return_value=[])
    if failure == "exception":
        runner.side_effect = RuntimeError("inference unavailable")
    monkeypatch.setattr(evaluator, f"run_{mode}", runner)
    write_baseline = Mock()
    monkeypatch.setattr(evaluator, "write_baseline", write_baseline)
    argv = ["confidence_eval.py", "--path", mode]
    if update_baselines:
        argv.append("--update-baselines")
    monkeypatch.setattr("sys.argv", argv)

    assert await evaluator.main() == 1
    runner.assert_awaited_once()
    write_baseline.assert_not_called()


async def test_successful_ocr_without_a_baseline_exits_zero(evaluator, monkeypatch):
    blocks = [([0, 0, 1, 1], "recognized text")]
    monkeypatch.setattr(evaluator, "run_hybrid", AsyncMock(return_value=blocks))
    monkeypatch.setattr(evaluator, "run_grounded", AsyncMock(return_value=blocks))
    monkeypatch.setattr("sys.argv", ["confidence_eval.py", "--path", "both"])

    assert await evaluator.main() == 0
    assert len(evaluator.HISTORY_CSV.read_text(encoding="utf-8").splitlines()) == 3


async def test_partial_failure_does_not_update_any_baselines(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator, "run_grounded", AsyncMock(side_effect=RuntimeError("offline")))
    monkeypatch.setattr(evaluator, "run_hybrid", AsyncMock(return_value=[([0, 0, 1, 1], "text")]))
    write_baseline = Mock()
    monkeypatch.setattr(evaluator, "write_baseline", write_baseline)
    monkeypatch.setattr("sys.argv", ["confidence_eval.py", "--path", "both", "--update-baselines"])

    assert await evaluator.main() == 1
    write_baseline.assert_not_called()


@pytest.mark.parametrize("mode", ["hybrid", "grounded"])
@pytest.mark.parametrize("update", [False, True])
async def test_failed_content_check_fails_without_a_baseline(evaluator, monkeypatch, mode, update):
    evaluator.CHECKS_DIR.mkdir()
    (evaluator.CHECKS_DIR / "handwritten.jsonl").write_text(
        '{"type":"present","text":"Halting"}\n', encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, f"run_{mode}", AsyncMock(
        return_value=[([0, 0, 1, 1], "Nothing")],
    ))
    write = Mock()
    monkeypatch.setattr(evaluator, "write_baseline", write)
    monkeypatch.setattr("sys.argv", ["confidence_eval.py", "--path", mode]
                        + (["--update-baselines"] if update else []))
    assert await evaluator.main() == 1
    write.assert_not_called()


async def test_grounded_evaluator_rejects_a_missing_selected_page(evaluator, monkeypatch):
    from pdf_ocr.core.grounded import GroundedBlock, GroundedResponse

    backend = SimpleNamespace(
        ocr_document=AsyncMock(return_value=GroundedResponse(
            blocks=[GroundedBlock([0, 0, 1, 1], "only one page", 1)],
            page_sizes=[(100, 100), (100, 100)], page_indices=[1, 3],
        )), aclose=AsyncMock(),
    )
    monkeypatch.setattr(evaluator, "PromptedGroundedOCR", lambda **kwargs: backend)
    with pytest.raises(RuntimeError, match="4"):
        await evaluator.run_grounded(Path("input.pdf"), "http://inference.test/v1", "vision", 1024)
    backend.aclose.assert_awaited_once()


@pytest.mark.parametrize("failing", ["hybrid", "grounded", "both"])
async def test_image_evaluator_returns_failure_for_empty_ocr(monkeypatch, failing):
    script = Path(__file__).resolve().parents[1] / "scripts/confidence_image.py"
    spec = importlib.util.spec_from_file_location("confidence_image", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for mode in ["hybrid", "grounded"]:
        blocks = [] if failing in (mode, "both") else [([0, 0, 1, 1], "text")]
        monkeypatch.setattr(module, f"run_{mode}", AsyncMock(return_value=blocks))
    monkeypatch.setattr("sys.argv", ["confidence_image.py"])
    assert await module.main() == 1


@pytest.mark.parametrize("mode", ["hybrid", "grounded"])
@pytest.mark.parametrize("raises", [False, True])
async def test_evaluator_closes_owned_clients_after_success_and_failure(
    evaluator, monkeypatch, mode, raises,
):
    close = AsyncMock()
    work = AsyncMock(return_value=SimpleNamespace(blocks=[], page_indices=[], page_sizes=[]))
    if raises:
        work.side_effect = RuntimeError("inference failed")
    if mode == "grounded":
        backend = SimpleNamespace(ocr_document=work, aclose=close)
        monkeypatch.setattr(evaluator, "PromptedGroundedOCR", lambda **kwargs: backend)
    else:
        monkeypatch.setattr(evaluator, "OCRProcessor", lambda **kwargs: SimpleNamespace(aclose=close))
        monkeypatch.setattr(evaluator, "HybridAligner", lambda: object())
        monkeypatch.setattr(evaluator, "PDFHandler", lambda: object())
        monkeypatch.setattr(evaluator, "OCRPipeline", lambda **kwargs: SimpleNamespace(run=work))
    runner = getattr(evaluator, f"run_{mode}")
    if raises:
        with pytest.raises(RuntimeError, match="inference failed"):
            await runner(Path("input.pdf"), "http://inference.test/v1", "vision", 1024)
    else:
        await runner(Path("input.pdf"), "http://inference.test/v1", "vision", 1024)
    close.assert_awaited_once()


def test_baseline_requires_matching_reference_and_scoring_protocol(evaluator):
    evaluator.BASELINES_DIR.mkdir()
    path = evaluator.BASELINES_DIR / "handwritten__hybrid.json"
    context = {"protocol": "page-aware-v2", "fixture_sha256": "verified"}
    path.write_text(json.dumps({"context": context, "metrics": {"bow_f1": 0.9}}))
    assert evaluator.compare_to_baseline(
        "handwritten.pdf", "hybrid", {"bow_f1": 0.9}, context=context,
    ) == []
    errors = evaluator.compare_to_baseline(
        "handwritten.pdf", "hybrid", {"bow_f1": 1.0},
        context={**context, "fixture_sha256": "different"},
    )
    assert len(errors) == 1 and "incompatible" in errors[0].lower()


def test_legacy_scores_cannot_silently_gate_a_new_evaluation(evaluator):
    evaluator.BASELINES_DIR.mkdir()
    (evaluator.BASELINES_DIR / "handwritten__hybrid.json").write_text('{"bow_f1":0.9}')
    errors = evaluator.compare_to_baseline(
        "handwritten.pdf", "hybrid", {"bow_f1": 1.0}, context={"protocol": "page-aware-v2"},
    )
    assert len(errors) == 1 and "provenance" in errors[0].lower()


def test_compatible_baseline_retains_existing_metric_tolerances(evaluator):
    context = {"protocol": "page-aware-v2"}
    evaluator.write_baseline("handwritten.pdf", "hybrid", {"bow_f1": 0.9, "recall": 0.9}, context=context)
    errors = evaluator.compare_to_baseline(
        "handwritten.pdf", "hybrid", {"bow_f1": 0.85, "recall": 0.87}, context=context,
    )
    assert len(errors) == 2


def test_reference_hashes_are_independent_of_checkout_newlines(evaluator, tmp_path):
    document = tmp_path / "source.pdf"
    fixture = tmp_path / "reference.json"
    checks = tmp_path / "checks.jsonl"
    document.write_bytes(b"source")
    fixture.write_bytes(b'{"text":"source"}\n')
    checks.write_bytes(b'{"type":"present","text":"source"}\n')
    before = evaluator.evaluation_context(document, fixture, checks, model="vision")
    fixture.write_bytes(fixture.read_bytes().replace(b"\n", b"\r\n"))
    checks.write_bytes(checks.read_bytes().replace(b"\n", b"\r\n"))
    assert evaluator.evaluation_context(document, fixture, checks, model="vision") == before


def test_shipped_controls_recompute_from_frozen_predictions(evaluator):
    root = Path(__file__).resolve().parents[1]
    for baseline_path in (root / "evals/baselines").glob("*__hybrid.json"):
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        raw = (root / baseline["provenance"]["capture"]).read_text(encoding="utf-8")
        assert hashlib.sha256(raw.encode()).hexdigest() == baseline["provenance"]["capture_sha256"]
        capture = json.loads(raw)
        stem = Path(capture["document"]).stem
        fixture = root / f"tests/fixtures/ground_truth_{stem}.json"
        checks_path = root / f"evals/checks/{stem}.jsonl"
        gt, _ = evaluator.load_ground_truth(fixture)
        report = evaluator.compute_report(capture["document"], gt, capture["blocks"],
                                           pipeline_page_indices=capture["page_indices"])
        checks = evaluator.run_doc_checks("\n".join(t for _, t in capture["blocks"]),
                                          evaluator.load_doc_checks(checks_path))
        assert evaluator._metrics_row(report, checks) == baseline["metrics"]
        assert evaluator.evaluation_context(root / "examples" / capture["document"], fixture,
                                             checks_path, model=capture["model"]) == baseline["context"]
        if stem == "notes":
            assert checks[:2] == (4, 5)  # the control is deliberately the failed pre-fix run
