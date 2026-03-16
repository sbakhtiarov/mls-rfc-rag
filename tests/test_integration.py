from __future__ import annotations

import os
from pathlib import Path

import pytest

from rfc_rag.chunking import chunk_sections
from rfc_rag.db import Database
from rfc_rag.models import IngestionRun
from rfc_rag.parser import parse_sections


RFC_PATH = Path(__file__).resolve().parents[1] / "rfc9420.txt"


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_fake_embedding(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        return _fake_embedding(text)


def _fake_embedding(text: str) -> list[float]:
    seed = sum(ord(char) for char in text) % 97
    return [float(seed)] * 1536


@pytest.mark.integration
def test_init_db_ingest_and_query_round_trip() -> None:
    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL or DATABASE_URL is required for integration tests.")

    database = Database(database_url)
    database.init_db()

    raw_text = RFC_PATH.read_text(encoding="utf-8-sig")
    sections = parse_sections(raw_text)
    chunks = chunk_sections(sections, source=str(RFC_PATH), strategy="fixed", chunk_size=700)
    sample_chunks = chunks[:8]
    embedder = FakeEmbedder()
    embeddings = embedder.embed_texts([chunk.content for chunk in sample_chunks])

    run_id = database.create_run_with_chunks(
        name="integration-fixed-700",
        source=str(RFC_PATH),
        strategy="fixed",
        chunk_size=700,
        embedding_model="fake-embedding-model",
        chunks=sample_chunks,
        embeddings=embeddings,
    )

    run = database.get_run(run_id)
    assert isinstance(run, IngestionRun)
    assert run.id == run_id
    assert run.strategy == "fixed"

    results = database.query_chunks(
        run_id=run_id,
        query_embedding=embedder.embed_text("Messaging Layer Security"),
        top_k=3,
    )

    assert results
    assert all(result.chunk_id for result in results)
    assert all(result.source == str(RFC_PATH) for result in results)
    assert all(result.section for result in results)
    assert all(result.content for result in results)
