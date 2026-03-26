from __future__ import annotations

import pytest

from rfc_rag.config import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_HOST, load_settings


def test_load_settings_uses_ollama_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)

    settings = load_settings()

    assert settings.database_url == "postgresql://example"
    assert settings.ollama_host == DEFAULT_OLLAMA_HOST
    assert settings.ollama_embed_model == DEFAULT_EMBED_MODEL


def test_load_settings_allows_ollama_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OLLAMA_HOST", "http://example:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "embeddinggemma")

    settings = load_settings()

    assert settings.ollama_host == "http://example:11434"
    assert settings.ollama_embed_model == "embeddinggemma"


def test_load_settings_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL is required."):
        load_settings()
