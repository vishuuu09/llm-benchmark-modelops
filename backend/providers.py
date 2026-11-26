
import requests
from backend import config


class ProviderError(RuntimeError):
    pass


def generate(question: str, context: str = ""):
    prompt = f"""Answer the question using the context when provided.
If the context is insufficient, say so.
Keep your answer concise and accurate.

Context:
{context}

Question:
{question}

Answer:"""

    if config.PROVIDER == "mock":
        return (
            f"Mock response: {question}. Context: {context[:160]}",
            "mock-baseline"
        )

    if config.PROVIDER == "ollama":
        try:
            response = requests.post(
                config.OLLAMA_BASE_URL.rstrip("/") + "/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 120
                    }
                },
                timeout=180
            )

            response.raise_for_status()

            return (
                response.json().get("response", ""),
                config.OLLAMA_MODEL
            )

        except requests.RequestException as exc:
            raise ProviderError(f"Ollama failed: {exc}") from exc

    if config.PROVIDER == "openai_compatible":
        if not config.OPENAI_API_KEY or not config.OPENAI_MODEL:
            raise ProviderError(
                "Set OPENAI_API_KEY and OPENAI_MODEL in .env"
            )

        try:
            response = requests.post(
                config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}"
                },
                json={
                    "model": config.OPENAI_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 150
                },
                timeout=180
            )

            response.raise_for_status()

            return (
                response.json()["choices"][0]["message"]["content"],
                config.OPENAI_MODEL
            )

        except requests.RequestException as exc:
            raise ProviderError(
                f"API request failed: {exc}"
            ) from exc

    raise ProviderError(
        "Provider must be mock, ollama, or openai_compatible"
    )
