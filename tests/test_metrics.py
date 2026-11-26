from backend.evaluator import evaluate,tokens
def test_tokenize():
    assert "retrieval" in tokens("Retrieval uses vectors")
def test_evaluation_returns_metrics():
    m=evaluate("What is RAG?","RAG retrieves information.","RAG retrieves information.","RAG retrieves information.")
    assert m["reference_overlap_proxy"]>0
    assert "context_overlap_proxy" in m
