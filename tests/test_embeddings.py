from __future__ import annotations

import pytest

from rfc_rag.embeddings import OllamaEmbedder


def test_ollama_embedder_uses_batch_embed_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, list[str]]] = []

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def embed(self, *, model: str, input: list[str]) -> dict[str, list[list[float]]]:
            calls.append((self.host, model, input))
            return {"embeddings": [[0.1] * 768, [0.2] * 768]}

    monkeypatch.setattr("rfc_rag.embeddings.Client", FakeClient)

    embedder = OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")
    embeddings = embedder.embed_texts(["alpha", "beta"])

    assert calls == [
        ("http://localhost:11434", "nomic-embed-text", ["alpha", "beta"]),
    ]
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768


def test_ollama_embedder_rejects_wrong_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def embed(self, *, model: str, input: list[str]) -> dict[str, list[list[float]]]:
            return {"embeddings": [[0.1] * 767]}

    monkeypatch.setattr("rfc_rag.embeddings.Client", FakeClient)

    embedder = OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")

    with pytest.raises(ValueError, match="Expected 768-dimension embeddings, got 767."):
        embedder.embed_text("alpha")
