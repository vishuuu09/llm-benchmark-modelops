
import pytest
import requests

from backend import providers
from backend.providers import ProviderError


class MockResponse:
    def __init__(self, data=None, error=None):
        self.data = data or {}
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.data


def test_mock_provider(monkeypatch):
    monkeypatch.setattr(providers.config, "PROVIDER", "mock")

    answer, model = providers.generate(
        "What is Python?",
        "Python is a programming language."
    )

    assert "What is Python?" in answer
    assert "Python is a programming language." in answer
    assert model == "mock-baseline"


def test_ollama_success(monkeypatch):
    monkeypatch.setattr(providers.config, "PROVIDER", "ollama")
    monkeypatch.setattr(
        providers.config, "OLLAMA_BASE_URL", "http://localhost:11434/"
    )
    monkeypatch.setattr(
        providers.config, "OLLAMA_MODEL", "llama3.2:3b"
    )

    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *args, **kwargs: MockResponse(
            {"response": "Python is a programming language."}
        )
    )

    answer, model = providers.generate("What is Python?", "Python context")

    assert answer == "Python is a programming language."
    assert model == "llama3.2:3b"


def test_ollama_request_failure(monkeypatch):
    monkeypatch.setattr(providers.config, "PROVIDER", "ollama")

    def mock_post(*args, **kwargs):
        raise requests.ConnectionError("Ollama server unavailable")

    monkeypatch.setattr(providers.requests, "post", mock_post)

    with pytest.raises(ProviderError, match="Ollama failed"):
        providers.generate("What is Python?")


def test_openai_compatible_success(monkeypatch):
    monkeypatch.setattr(
        providers.config, "PROVIDER", "openai_compatible"
    )
    monkeypatch.setattr(
        providers.config, "OPENAI_API_KEY", "test-api-key"
    )
    monkeypatch.setattr(
        providers.config, "OPENAI_MODEL", "test-model"
    )
    monkeypatch.setattr(
        providers.config, "OPENAI_BASE_URL", "https://example.com/v1/"
    )

    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *args, **kwargs: MockResponse({
            "choices": [
                {"message": {"content": "Python is a programming language."}}
            ]
        })
    )

    answer, model = providers.generate("What is Python?")

    assert answer == "Python is a programming language."
    assert model == "test-model"


def test_openai_missing_api_key(monkeypatch):
    monkeypatch.setattr(
        providers.config, "PROVIDER", "openai_compatible"
    )
    monkeypatch.setattr(providers.config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(
        providers.config, "OPENAI_MODEL", "test-model"
    )

    with pytest.raises(ProviderError, match="Set OPENAI_API_KEY"):
        providers.generate("What is Python?")


def test_openai_missing_model(monkeypatch):
    monkeypatch.setattr(
        providers.config, "PROVIDER", "openai_compatible"
    )
    monkeypatch.setattr(
        providers.config, "OPENAI_API_KEY", "test-api-key"
    )
    monkeypatch.setattr(providers.config, "OPENAI_MODEL", "")

    with pytest.raises(ProviderError, match="Set OPENAI_API_KEY"):
        providers.generate("What is Python?")


def test_openai_request_failure(monkeypatch):
    monkeypatch.setattr(
        providers.config, "PROVIDER", "openai_compatible"
    )
    monkeypatch.setattr(
        providers.config, "OPENAI_API_KEY", "test-api-key"
    )
    monkeypatch.setattr(
        providers.config, "OPENAI_MODEL", "test-model"
    )

    def mock_post(*args, **kwargs):
        raise requests.Timeout("API request timed out")

    monkeypatch.setattr(providers.requests, "post", mock_post)

    with pytest.raises(ProviderError, match="API request failed"):
        providers.generate("What is Python?")


def test_invalid_provider(monkeypatch):
    monkeypatch.setattr(providers.config, "PROVIDER", "invalid")

    with pytest.raises(
        ProviderError,
        match="Provider must be mock, ollama, or openai_compatible"
    ):
        providers.generate("What is Python?")
