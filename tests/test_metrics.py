
from backend.evaluator import evaluate, tokens


def test_tokenize():
    assert "retrieval" in tokens("Retrieval uses vectors")


def test_tokenize_empty_and_punctuation():
    assert tokens("") == set()
    assert tokens("!!! ??? ...") == set()
    assert tokens(None) == set()


def test_evaluation_returns_metrics():
    metrics = evaluate(
        "What is RAG?",
        "RAG retrieves information.",
        "RAG retrieves information.",
        "RAG retrieves information."
    )

    assert metrics["reference_overlap_proxy"] > 0
    assert "context_overlap_proxy" in metrics
    assert "semantic_similarity" in metrics


def test_evaluation_empty_question():
    metrics = evaluate(
        "",
        "Python is a programming language.",
        "Python is useful.",
        "Python is useful."
    )

    assert metrics["keyword_relevance"] == 0.0


def test_evaluation_empty_answer():
    metrics = evaluate(
        "What is Python?",
        "Python is a programming language.",
        "",
        "Python is a programming language."
    )

    assert metrics["context_overlap_proxy"] == 0.0
    assert metrics["reference_precision_proxy"] == 0.0
    assert metrics["reference_f1_proxy"] == 0.0
    assert metrics["semantic_similarity"] == 0.0
    assert metrics["groundedness_proxy"] == 0.0
    assert metrics["answer_token_count"] == 0


def test_evaluation_empty_context():
    metrics = evaluate(
        "What is Python?",
        "",
        "Python is a programming language.",
        "Python is a programming language."
    )

    assert metrics["context_overlap_proxy"] == 0.0
    assert metrics["groundedness_proxy"] == 0.0


def test_evaluation_empty_reference():
    metrics = evaluate(
        "What is Python?",
        "Python is a programming language.",
        "Python is a programming language.",
        ""
    )

    assert metrics["reference_recall_proxy"] if "reference_recall_proxy" in metrics else metrics["reference_overlap_proxy"] == 0.0
    assert metrics["reference_f1_proxy"] == 0.0
    assert metrics["semantic_similarity"] == 0.0
