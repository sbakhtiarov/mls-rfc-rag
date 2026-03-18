from datetime import datetime, timezone

import pytest

from rfc_rag.models import IngestionRun, QueryResult
from rfc_rag.search_service import SearchResponse, search_chunks, serialize_search_response


def test_search_chunks_returns_shared_payload() -> None:
    run = _make_run(is_active=True)
    results = [
        QueryResult(
            chunk_id="fixed:1-introduction:0",
            source="rfc9420.txt",
            section="1 | Introduction",
            content="Chunk body",
            score=0.9,
        )
    ]
    response = search_chunks(
        database=FakeDatabase(run=run, active_run=run, results=results),
        embedder=FakeEmbedder(),
        query="external commits",
        top_k=5,
    )

    payload = serialize_search_response(response)

    assert isinstance(response, SearchResponse)
    assert payload["run"]["id"] == run.id
    assert payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"
    assert payload["results"][0]["content"] == "Chunk body"


def test_search_chunks_requires_active_run_when_run_id_missing() -> None:
    with pytest.raises(ValueError, match="No active run is configured"):
        search_chunks(
            database=FakeDatabase(run=None, active_run=None, results=[]),
            embedder=FakeEmbedder(),
            query="external commits",
        )


def test_search_chunks_rejects_invalid_top_k() -> None:
    run = _make_run(is_active=True)
    with pytest.raises(ValueError, match="top_k must be less than or equal to 20"):
        search_chunks(
            database=FakeDatabase(run=run, active_run=run, results=[]),
            embedder=FakeEmbedder(),
            query="external commits",
            top_k=21,
        )


class FakeDatabase:
    def __init__(self, *, run, active_run, results):
        self._run = run
        self._active_run = active_run
        self._results = results

    def get_run(self, run_id: int):
        return self._run

    def get_active_run(self):
        return self._active_run

    def query_chunks(self, *, run_id: int, query_embedding: list[float], top_k: int):
        assert query_embedding == [0.5, 0.25]
        return self._results[:top_k]


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        assert text == "external commits"
        return [0.5, 0.25]


def _make_run(*, is_active: bool) -> IngestionRun:
    return IngestionRun(
        id=11,
        name="active-run",
        source="rfc9420.txt",
        strategy="fixed",
        chunk_size=700,
        embedding_model="fake-embedding-model",
        created_at=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc),
        is_active=is_active,
    )
