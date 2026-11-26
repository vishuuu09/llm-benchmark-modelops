# LLM Benchmarking & ModelOps Platform

A lightweight platform for benchmarking locally hosted Large Language Models (LLMs), evaluating response quality, measuring inference performance, and tracking model experiments.

Built using **Python, FastAPI, Ollama, Sentence Transformers, SQLite, Streamlit, and optional MLflow integration**.

## Overview

Evaluating LLMs requires more than checking whether a model generates a response. Models must also be assessed for relevance, similarity to expected answers, context alignment, and inference latency.

This project provides an evaluation pipeline that runs benchmark datasets against an LLM, calculates multiple quality metrics, applies configurable quality gates, and stores benchmark results for analysis.

## Key Features

- **LLM Inference:** Supports locally hosted models through Ollama.
- **Automated Benchmarking:** Executes evaluation datasets containing questions, reference answers, and supporting context.
- **Multi-dimensional Evaluation:** Calculates keyword relevance, reference precision, recall, F1, semantic similarity, and context-overlap-based groundedness.
- **Performance Monitoring:** Measures inference latency and response token counts.
- **Quality Gates:** Evaluates benchmark results against configurable thresholds.
- **Experiment Tracking:** Supports optional MLflow integration for recording benchmark metrics.
- **Results Storage:** Uses SQLite to persist benchmark execution results.
- **Dashboard:** Provides a Streamlit interface for viewing benchmark results.
- **Automated Testing:** Includes tests for API endpoints, benchmark execution, evaluation metrics, providers, dataset validation, and database operations.

## Architecture

```text
                  Benchmark Dataset
                         |
                         v
                  FastAPI Backend
                         |
                         v
                   Benchmark Engine
                         |
              +----------+----------+
              |                     |
              v                     v
         Ollama Provider      Evaluation Engine
              |                     |
              v                     v
           LLM Model           Quality Metrics
              |                     |
              +----------+----------+
                         |
                         v
                    Quality Gates
                         |
              +----------+----------+
              |          |          |
              v          v          v
           SQLite      MLflow    Streamlit
           Storage    Tracking   Dashboard
```

## Evaluation Metrics

The platform evaluates model responses using multiple metrics:

| Metric | Purpose |
|---|---|
| Keyword Relevance | Measures matching between generated responses and expected keywords |
| Reference Precision | Measures the proportion of generated answer terms matching reference terms |
| Reference Recall | Measures reference-term coverage in generated responses |
| Reference F1 | Combines reference precision and recall |
| Semantic Similarity | Measures semantic similarity between generated and reference answers using Sentence Transformers |
| Context Overlap | Measures lexical overlap between generated responses and supporting context |
| Groundedness Proxy | Estimates context alignment using lexical token overlap |
| Inference Latency | Measures response generation time |
| Answer Token Count | Tracks generated response length |

**Important:** Groundedness is currently a lexical-overlap proxy, not an NLI-based factuality or hallucination detector. These metrics are useful for benchmarking but should not be treated as definitive measures of factual correctness.

## Quality Gates

The platform uses configurable thresholds to determine whether a benchmark run passes or fails.

The quality gates include:

- Minimum reference F1 score
- Minimum keyword relevance
- Minimum semantic similarity
- Minimum groundedness proxy
- Maximum average inference latency

Thresholds can be configured through environment variables.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Backend API | FastAPI |
| LLM Inference | Ollama |
| Embeddings | Sentence Transformers |
| Evaluation | Custom evaluation pipeline |
| Database | SQLite |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow (optional) |
| Testing | Pytest, pytest-cov |
| Containerization | Docker, Docker Compose |
| CI | GitHub Actions |

## Project Structure

```text
llm_benchmark_modelops_starter/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── benchmark.py
│   ├── providers.py
│   ├── evaluator.py
│   ├── config.py
│   └── storage.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── benchmark.jsonl
│   └── benchmark_test.jsonl
│
├── tests/
│   ├── test_dataset.py
│   ├── test_benchmark.py
│   ├── test_main.py
│   ├── test_metrics.py
│   ├── test_providers.py
│   └── test_storage.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vishuuu09/llm-benchmark-modelops.git

cd llm-benchmark-modelops
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using the example configuration:

```powershell
Copy-Item .env.example .env
```

Configure the required values:

```env
PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

DATABASE_PATH=benchmark.db

QUALITY_GATE_MIN_REFERENCE_F1=0.15
QUALITY_GATE_MIN_KEYWORD_RELEVANCE=0.15
QUALITY_GATE_MIN_SEMANTIC_SIMILARITY=0.55
QUALITY_GATE_MIN_GROUNDEDNESS=0.15
QUALITY_GATE_MAX_AVG_LATENCY_MS=60000

MLFLOW_ENABLED=false
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### 5. Start Ollama

Install Ollama and download the model:

```bash
ollama pull llama3.2:3b
```

Ensure the Ollama service is running before executing benchmarks.

### 6. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

Access the API documentation at:

```text
http://127.0.0.1:8000/docs
```

### 7. Run the Streamlit dashboard

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run dashboard/app.py
```

## Sample Benchmark Results

A benchmark run was executed using the `llama3.2:3b` model through Ollama with a 20-question evaluation dataset.

| Metric | Result |
|---|---:|
| Evaluation Questions | 20 |
| Average Latency | 9,009 ms |
| Reference Overlap | 0.5119 |
| Reference F1 | 0.3112 |
| Keyword Relevance | 0.4550 |
| Semantic Similarity | 0.7171 |
| Groundedness Proxy | 0.4763 |
| Quality Gate | Passed |

*These results represent one benchmark run and are not a guarantee of performance across different models, hardware, or datasets.*

## Testing and Code Coverage

The project includes **45 automated tests** covering the backend components and evaluation pipeline.

### Test Coverage

| Backend Module | Statement Coverage |
|---|---:|
| Benchmark Engine | 100% |
| Configuration | 100% |
| Evaluation Engine | 100% |
| FastAPI Backend | 100% |
| LLM Providers | 100% |
| SQLite Storage | 100% |
| **Total Backend Coverage** | **100%** |

### Verified Test Results

- 45 automated tests passed.
- 170 backend statements measured.
- 0 uncovered backend statements.
- MLflow success and failure paths tested.
- HTML coverage report generated successfully.

### Run the test suite

```bash
python -m pytest -v
```

### Generate the coverage report

```bash
python -m pytest --cov=backend --cov-report=term-missing --cov-report=html
```

Open the detailed HTML report:

```text
htmlcov/index.html
```

**Note:** 100% coverage represents executed backend statements during automated tests. It does not guarantee that the application is bug-free or that LLM responses are factually correct.

## Future Improvements

- Implement NLI-based groundedness and factual consistency evaluation.
- Add side-by-side comparisons across multiple LLM providers.
- Introduce additional benchmark datasets and evaluation scenarios.
- Improve experiment tracking and model version comparisons.
- Add richer dashboard visualizations and historical trend analysis.
- Expand automated testing and CI/CD workflows.

## Author

**Sai Vishal Mittapalli**

GitHub: [@vishuuu09](https://github.com/vishuuu09)

---

**Project Repository:** [LLM Benchmarking & ModelOps Platform](https://github.com/vishuuu09/llm-benchmark-modelops)