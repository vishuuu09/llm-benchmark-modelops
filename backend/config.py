
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "mock").lower()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://api.openai.com/v1"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

DATABASE_PATH = os.getenv("DATABASE_PATH", "benchmark.db")

# Legacy threshold retained for compatibility
QUALITY_GATE_MIN_REFERENCE_OVERLAP = float(
    os.getenv("QUALITY_GATE_MIN_REFERENCE_OVERLAP", "0.15")
)

# New quality gate thresholds
QUALITY_GATE_MIN_REFERENCE_F1 = float(
    os.getenv("QUALITY_GATE_MIN_REFERENCE_F1", "0.15")
)
QUALITY_GATE_MIN_KEYWORD_RELEVANCE = float(
    os.getenv("QUALITY_GATE_MIN_KEYWORD_RELEVANCE", "0.15")
)
QUALITY_GATE_MAX_AVG_LATENCY_MS = float(
    os.getenv("QUALITY_GATE_MAX_AVG_LATENCY_MS", "60000")
)

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
)

QUALITY_GATE_MIN_SEMANTIC_SIMILARITY = float(
    os.getenv("QUALITY_GATE_MIN_SEMANTIC_SIMILARITY", "0.55")
)
QUALITY_GATE_MIN_GROUNDEDNESS = float(
    os.getenv("QUALITY_GATE_MIN_GROUNDEDNESS", "0.15")
)