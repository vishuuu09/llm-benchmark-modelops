
import json
import sqlite3

import pytest

from backend import storage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Use a separate temporary SQLite database for each test."""
    db_path = tmp_path / "test_benchmark.db"

    monkeypatch.setattr(storage.config, "DATABASE_PATH", str(db_path))

    storage.init_db()

    return db_path


def sample_run():
    return {
        "provider": "ollama",
        "model": "llama3.2:3b",
        "dataset_path": "data/benchmark_test.jsonl",
        "total_items": 3,
        "average_latency": 1200.5,
        "average_reference_overlap": 0.65,
        "passed_gate": True,
        "results": [
            {
                "question": "What is Python?",
                "reference_overlap": 0.65,
            }
        ],
    }


def test_init_db_creates_runs_table(temp_db):
    with sqlite3.connect(temp_db) as conn:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        ).fetchone()

    assert result is not None


def test_save_returns_unique_run_id(temp_db):
    first_id = storage.save(sample_run())
    second_id = storage.save(sample_run())

    assert first_id
    assert second_id
    assert first_id != second_id


def test_save_and_get_run(temp_db):
    data = sample_run()

    run_id = storage.save(data)
    saved_run = storage.get_run(run_id)

    assert saved_run is not None
    assert saved_run["run_id"] == run_id
    assert saved_run["provider"] == data["provider"]
    assert saved_run["model"] == data["model"]
    assert saved_run["total_items"] == data["total_items"]
    assert saved_run["average_latency"] == data["average_latency"]
    assert saved_run["average_reference_overlap"] == data["average_reference_overlap"]
    assert saved_run["passed_gate"] == 1
    assert saved_run["results"] == data["results"]


def test_get_run_returns_none_for_unknown_id(temp_db):
    result = storage.get_run("unknown-run-id")

    assert result is None


def test_list_runs_returns_saved_runs(temp_db):
    first_id = storage.save(sample_run())
    second_id = storage.save(sample_run())

    runs = storage.list_runs()

    assert len(runs) == 2
    assert {run["run_id"] for run in runs} == {first_id, second_id}


def test_list_runs_does_not_include_results(temp_db):
    storage.save(sample_run())

    runs = storage.list_runs()

    assert len(runs) == 1
    assert "results" not in runs[0]
    assert "results_json" not in runs[0]


def test_save_preserves_failed_gate(temp_db):
    data = sample_run()
    data["passed_gate"] = False

    run_id = storage.save(data)
    saved_run = storage.get_run(run_id)

    assert saved_run["passed_gate"] == 0


def test_save_preserves_results_json(temp_db):
    data = sample_run()

    run_id = storage.save(data)
    saved_run = storage.get_run(run_id)

    assert isinstance(saved_run["results"], list)
    assert saved_run["results"][0]["question"] == "What is Python?"
