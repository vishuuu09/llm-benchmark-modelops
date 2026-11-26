
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend import main
from backend.providers import ProviderError


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint_mock_provider(monkeypatch):
    monkeypatch.setattr(main.config, "PROVIDER", "mock")

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["model"] == "mock-baseline"


def test_models_endpoint_ollama_provider(monkeypatch):
    monkeypatch.setattr(main.config, "PROVIDER", "ollama")
    monkeypatch.setattr(main.config, "OLLAMA_MODEL", "llama3.2:3b")

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["model"] == "llama3.2:3b"


def test_query_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate",
        lambda question, context: ("Python is a programming language.", "mock-model")
    )

    response = client.post(
        "/query",
        json={
            "question": "What is Python?",
            "context": "Python is a programming language."
        }
    )

    assert response.status_code == 200
    assert response.json()["model"] == "mock-model"
    assert "Python" in response.json()["answer"]


def test_query_provider_error(monkeypatch):
    def mock_generate(question, context):
        raise ProviderError("Provider unavailable")

    monkeypatch.setattr(main, "generate", mock_generate)

    response = client.post(
        "/query",
        json={"question": "What is Python?"}
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Provider unavailable"


def test_query_validation_error():
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_benchmark_success(monkeypatch):
    expected = {
        "provider": "mock",
        "model": "mock-model",
        "total_items": 2,
        "passed_gate": True
    }

    monkeypatch.setattr(main, "run", lambda dataset: expected)

    response = client.post("/benchmark?dataset=test.jsonl")

    assert response.status_code == 200
    assert response.json()["passed_gate"] is True


def test_benchmark_error(monkeypatch):
    def mock_run(dataset):
        raise ValueError("Dataset not found")

    monkeypatch.setattr(main, "run", mock_run)

    response = client.post("/benchmark")

    assert response.status_code == 400
    assert response.json()["detail"] == "Dataset not found"


def test_runs_endpoint(monkeypatch):
    sample_runs = [
        {"run_id": "run-1", "model": "mock-model"}
    ]

    monkeypatch.setattr(
        main.storage, "list_runs", lambda: sample_runs
    )

    response = client.get("/runs")

    assert response.status_code == 200
    assert response.json() == sample_runs


def test_run_details_success(monkeypatch):
    sample_run = {
        "run_id": "run-1",
        "model": "mock-model",
        "total_items": 2
    }

    monkeypatch.setattr(
        main.storage, "get_run", lambda rid: sample_run
    )

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"


def test_run_details_not_found(monkeypatch):
    monkeypatch.setattr(
        main.storage, "get_run", lambda rid: None
    )

    response = client.get("/runs/unknown")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_metrics_with_runs(monkeypatch):
    sample_runs = [
        {"run_id": "run-1", "model": "mock-model"},
        {"run_id": "run-2", "model": "llama3.2:3b"}
    ]

    monkeypatch.setattr(
        main.storage, "list_runs", lambda: sample_runs
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["run_count"] == 2
    assert response.json()["latest"] == sample_runs[0]


def test_metrics_without_runs(monkeypatch):
    monkeypatch.setattr(
        main.storage, "list_runs", lambda: []
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["run_count"] == 0
    assert response.json()["latest"] is None
