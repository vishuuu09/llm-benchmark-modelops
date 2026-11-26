# LLM Benchmarking & ModelOps Platform (MVP)

Starter application to benchmark configurable LLM providers on a fixed dataset.

### Included
- Offline mock provider for smoke tests; Ollama and OpenAI-compatible adapters
- FastAPI backend, SQLite run history, Streamlit dashboard
- Evaluation proxy metrics: keyword relevance, context overlap, reference overlap, completeness, latency
- Quality gate, optional MLflow tracking, pytest, Docker, GitHub Actions

**Important:** lexical metrics are proxies, not validated semantic or faithfulness scores. Mock results are not real model results. Fine-tuning, production-grade drift monitoring and rollback are not implemented in this starter.

## Windows quick start
1. Python 3.11+ installed.
2. Open folder in VS Code / PowerShell.
3. `py -3.11 -m venv .venv`
4. `.\.venv\Scripts\Activate.ps1`
5. `pip install -r requirements.txt`
6. `Copy-Item .env.example .env`
7. Terminal 1: `uvicorn backend.main:app --reload`
8. Terminal 2: activate venv, then `streamlit run dashboard/app.py`
9. Visit Streamlit URL and click **Run benchmark**. API docs: http://127.0.0.1:8000/docs

## Connect a real model
Ollama: install it, then `ollama pull qwen2.5:3b` (model/hardware requirements vary). In `.env`, set `PROVIDER=ollama` and `OLLAMA_MODEL=qwen2.5:3b`.

Hosted OpenAI-compatible: set `PROVIDER=openai_compatible`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. Keep keys out of Git.

## APIs
- `GET /health`
- `GET /models`
- `POST /query`
- `POST /benchmark?dataset=data/benchmark.jsonl`
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /metrics`

## Tests
`pytest -q`

## Sprint plan
See `docs/2-3-hour-build-plan.md`. Use the first 2–3 hours to run the MVP, connect one real model, benchmark, add dataset cases, review outputs manually, and document the model/hardware/settings. LoRA and deeper ModelOps can be follow-up milestones.
