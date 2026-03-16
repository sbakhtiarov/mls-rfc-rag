from __future__ import annotations

from openai import OpenAI

from rfc_rag.config import EMBEDDING_DIMENSION


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        embeddings = [item.embedding for item in response.data]
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
