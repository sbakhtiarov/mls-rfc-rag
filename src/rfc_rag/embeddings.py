from __future__ import annotations

from ollama import Client

from rfc_rag.config import EMBEDDING_DIMENSION


class OllamaEmbedder:
    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model
        self._client = Client(host=host)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self.model, input=texts)
        embeddings = response["embeddings"] if isinstance(response, dict) else response.embeddings
        for embedding in embeddings:
            _validate_embedding_dimension(embedding)
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def _validate_embedding_dimension(embedding: list[float]) -> None:
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimension embeddings, got {len(embedding)}."
        )
