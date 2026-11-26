from backend.benchmark import load_dataset
def test_data_has_cases():
    data=load_dataset("data/benchmark.jsonl")
    assert len(data)>=15
    assert all("question" in x and "context" in x and "reference_answer" in x for x in data)
