
import json
import time
from pathlib import Path

from backend import config, storage
from backend.providers import generate
from backend.evaluator import evaluate


def load_dataset(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    items = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not items:
        raise ValueError("Dataset empty")

    return items


def average(results, key):
    values = [item[key] for item in results]
    return round(sum(values) / len(values), 4)


def run(path="data/benchmark.jsonl"):
    items = load_dataset(path)
    results = []
    model = "unknown"

    for item in items:
        start = time.perf_counter()

        answer, model = generate(
            item["question"],
            item.get("context", "")
        )

        latency = (time.perf_counter() - start) * 1000

        metrics = evaluate(
            item["question"],
            item.get("context", ""),
            answer,
            item.get("reference_answer", "")
        )

        results.append({
            "id": item["id"],
            "question": item["question"],
            "context": item.get("context", ""),
            "reference_answer": item.get("reference_answer", ""),
            "answer": answer,
            "latency_ms": round(latency, 2),
            **metrics
        })

    # Calculate average metrics
    avg_latency = average(results, "latency_ms")
    avg_reference = average(results, "reference_overlap_proxy")
    avg_f1 = average(results, "reference_f1_proxy")
    avg_keyword = average(results, "keyword_relevance")
    avg_semantic = average(results, "semantic_similarity")
    avg_groundedness = average(results, "groundedness_proxy")

    # Quality gate checks
    gate_checks = {
        "reference_f1": (
            avg_f1 >= config.QUALITY_GATE_MIN_REFERENCE_F1
        ),
        "keyword_relevance": (
            avg_keyword >= config.QUALITY_GATE_MIN_KEYWORD_RELEVANCE
        ),
        "semantic_similarity": (
            avg_semantic >= config.QUALITY_GATE_MIN_SEMANTIC_SIMILARITY
        ),
        "groundedness": (
            avg_groundedness >= config.QUALITY_GATE_MIN_GROUNDEDNESS
        ),
        "average_latency": (
            avg_latency <= config.QUALITY_GATE_MAX_AVG_LATENCY_MS
        )
    }

    # Final benchmark output
    data = {
        "provider": config.PROVIDER,
        "model": model,
        "dataset_path": path,
        "total_items": len(results),
        "average_latency": avg_latency,
        "average_reference_overlap": avg_reference,
        "average_reference_f1": avg_f1,
        "average_keyword_relevance": avg_keyword,
        "average_semantic_similarity": avg_semantic,
        "average_groundedness": avg_groundedness,
        "gate_checks": gate_checks,
        "passed_gate": all(gate_checks.values()),
        "results": results
    }

    data["run_id"] = storage.save(data)

    # Optional MLflow experiment tracking
    if config.MLFLOW_ENABLED:
        try:
            import mlflow

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

            with mlflow.start_run(run_name=f"benchmark-{model}"):
                mlflow.log_params({
                    "provider": config.PROVIDER,
                    "model": model,
                    "dataset": path
                })

                mlflow.log_metrics({
                    "latency_ms": avg_latency,
                    "reference_overlap_proxy": avg_reference,
                    "reference_f1_proxy": avg_f1,
                    "keyword_relevance": avg_keyword,
                    "semantic_similarity": avg_semantic,
                    "groundedness_proxy": avg_groundedness,
                    "quality_gate_pass": int(data["passed_gate"])
                })

        except Exception as exc:
            data["mlflow_note"] = str(exc)

    return data
