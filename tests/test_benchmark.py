
import json

import pytest

from backend import benchmark


def test_load_dataset_success(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id": "1", "question": "What is Python?"}\n'
        '{"id": "2", "question": "What is FastAPI?"}\n',
        encoding="utf-8"
    )

    result = benchmark.load_dataset(dataset)

    assert len(result) == 2
    assert result[0]["question"] == "What is Python?"


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        benchmark.load_dataset(tmp_path / "missing.jsonl")


def test_load_dataset_empty_file(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset empty"):
        benchmark.load_dataset(dataset)


def test_average():
    results = [
        {"score": 0.5},
        {"score": 0.7},
        {"score": 0.9}
    ]

    assert benchmark.average(results, "score") == 0.7


def test_run_success(monkeypatch):
    items = [
        {
            "id": "1",
            "question": "What is Python?",
            "context": "Python is a programming language.",
            "reference_answer": "Python is a programming language."
        },
        {
            "id": "2",
            "question": "What is FastAPI?",
            "context": "FastAPI is a Python web framework.",
            "reference_answer": "FastAPI is a web framework."
        }
    ]

    metrics = {
        "reference_overlap_proxy": 0.8,
        "reference_f1_proxy": 0.8,
        "keyword_relevance": 0.8,
        "semantic_similarity": 0.8,
        "groundedness_proxy": 0.8
    }

    monkeypatch.setattr(benchmark, "load_dataset", lambda path: items)
    monkeypatch.setattr(
        benchmark,
        "generate",
        lambda question, context: ("Mock answer", "mock-model")
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate",
        lambda question, context, answer, reference: metrics
    )
    monkeypatch.setattr(
        benchmark.storage,
        "save",
        lambda data: "test-run-id"
    )

    monkeypatch.setattr(benchmark.config, "PROVIDER", "mock")
    monkeypatch.setattr(benchmark.config, "MLFLOW_ENABLED", False)

    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_REFERENCE_F1", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_KEYWORD_RELEVANCE", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_SEMANTIC_SIMILARITY", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_GROUNDEDNESS", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MAX_AVG_LATENCY_MS", 10000
    )

    result = benchmark.run("mock-dataset.jsonl")

    assert result["provider"] == "mock"
    assert result["model"] == "mock-model"
    assert result["total_items"] == 2
    assert result["passed_gate"] is True
    assert result["run_id"] == "test-run-id"
    assert len(result["results"]) == 2


def test_run_quality_gate_failure(monkeypatch):
    items = [
        {
            "id": "1",
            "question": "Test question",
            "context": "",
            "reference_answer": "Expected answer"
        }
    ]

    low_metrics = {
        "reference_overlap_proxy": 0.1,
        "reference_f1_proxy": 0.1,
        "keyword_relevance": 0.1,
        "semantic_similarity": 0.1,
        "groundedness_proxy": 0.1
    }

    monkeypatch.setattr(benchmark, "load_dataset", lambda path: items)
    monkeypatch.setattr(
        benchmark,
        "generate",
        lambda question, context: ("Poor answer", "mock-model")
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate",
        lambda *args: low_metrics
    )
    monkeypatch.setattr(
        benchmark.storage,
        "save",
        lambda data: "failed-run-id"
    )

    monkeypatch.setattr(benchmark.config, "PROVIDER", "mock")
    monkeypatch.setattr(benchmark.config, "MLFLOW_ENABLED", False)

    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_REFERENCE_F1", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_KEYWORD_RELEVANCE", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_SEMANTIC_SIMILARITY", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_GROUNDEDNESS", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MAX_AVG_LATENCY_MS", 10000
    )

    result = benchmark.run()

    assert result["passed_gate"] is False
    assert result["gate_checks"]["reference_f1"] is False


def test_run_mlflow_failure(monkeypatch):
    import sys
    from types import SimpleNamespace

    items = [
        {
            "id": "1",
            "question": "Test question",
            "context": "",
            "reference_answer": "Expected answer"
        }
    ]

    metrics = {
        "reference_overlap_proxy": 0.8,
        "reference_f1_proxy": 0.8,
        "keyword_relevance": 0.8,
        "semantic_similarity": 0.8,
        "groundedness_proxy": 0.8
    }

    monkeypatch.setattr(benchmark, "load_dataset", lambda path: items)
    monkeypatch.setattr(
        benchmark,
        "generate",
        lambda *args: ("Mock answer", "mock-model")
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate",
        lambda *args: metrics
    )
    monkeypatch.setattr(
        benchmark.storage,
        "save",
        lambda data: "mlflow-test-id"
    )

    monkeypatch.setattr(benchmark.config, "PROVIDER", "mock")
    monkeypatch.setattr(benchmark.config, "MLFLOW_ENABLED", True)

    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_REFERENCE_F1", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_KEYWORD_RELEVANCE", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_SEMANTIC_SIMILARITY", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MIN_GROUNDEDNESS", 0.5
    )
    monkeypatch.setattr(
        benchmark.config, "QUALITY_GATE_MAX_AVG_LATENCY_MS", 10000
    )

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda *args: None,
        start_run=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("MLflow unavailable")
        ),
        log_params=lambda *args: None,
        log_metrics=lambda *args: None
    )

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(
        benchmark.config, "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )

    result = benchmark.run()

    assert result["run_id"] == "mlflow-test-id"
    assert "mlflow_note" in result
    assert "MLflow unavailable" in result["mlflow_note"]



def test_run_mlflow_success(monkeypatch):
    import sys
    from types import SimpleNamespace
    from contextlib import contextmanager

    items = [
        {
            "id": "1",
            "question": "What is Python?",
            "context": "Python is a programming language.",
            "reference_answer": "Python is a programming language."
        }
    ]

    metrics = {
        "reference_overlap_proxy": 0.8,
        "reference_f1_proxy": 0.8,
        "keyword_relevance": 0.8,
        "semantic_similarity": 0.8,
        "groundedness_proxy": 0.8
    }

    monkeypatch.setattr(benchmark, "load_dataset", lambda path: items)
    monkeypatch.setattr(
        benchmark, "generate",
        lambda *args: ("Python is a programming language.", "mock-model")
    )
    monkeypatch.setattr(benchmark, "evaluate", lambda *args: metrics)
    monkeypatch.setattr(
        benchmark.storage, "save", lambda data: "mlflow-success-id"
    )

    monkeypatch.setattr(benchmark.config, "PROVIDER", "mock")
    monkeypatch.setattr(benchmark.config, "MLFLOW_ENABLED", True)
    monkeypatch.setattr(benchmark.config, "MLFLOW_TRACKING_URI", "sqlite:///test.db")

    monkeypatch.setattr(benchmark.config, "QUALITY_GATE_MIN_REFERENCE_F1", 0.5)
    monkeypatch.setattr(benchmark.config, "QUALITY_GATE_MIN_KEYWORD_RELEVANCE", 0.5)
    monkeypatch.setattr(benchmark.config, "QUALITY_GATE_MIN_SEMANTIC_SIMILARITY", 0.5)
    monkeypatch.setattr(benchmark.config, "QUALITY_GATE_MIN_GROUNDEDNESS", 0.5)
    monkeypatch.setattr(benchmark.config, "QUALITY_GATE_MAX_AVG_LATENCY_MS", 10000)

    @contextmanager
    def fake_start_run(**kwargs):
        yield

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        start_run=fake_start_run,
        log_params=lambda params: None,
        log_metrics=lambda values: None
    )

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    result = benchmark.run()

    assert result["run_id"] == "mlflow-success-id"
    assert result["passed_gate"] is True
    assert "mlflow_note" not in result
