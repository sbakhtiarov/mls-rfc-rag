from datetime import datetime, timezone

import pytest

from rfc_rag.models import IngestionRun, QueryResult
from rfc_rag.search_service import (
    SearchExecutionError,
    SearchResponse,
    execute_search,
    search_chunks,
    serialize_search_response,
    validate_score_threshold,
)


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
    assert payload["results"][0]["citations"][0]["section"] == "1 | Introduction"
    assert payload["results"][0]["citations"][0]["quote"] == "Chunk body"


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


def test_search_chunks_uses_saved_default_top_k_when_omitted() -> None:
    run = _make_run(is_active=True)
    results = [
        QueryResult(
            chunk_id="fixed:1-introduction:0",
            source="rfc9420.txt",
            section="1 | Introduction",
            content="First chunk",
            score=0.9,
        ),
        QueryResult(
            chunk_id="fixed:1-introduction:1",
            source="rfc9420.txt",
            section="1 | Introduction",
            content="Second chunk",
            score=0.8,
        ),
    ]
    response = search_chunks(
        database=FakeDatabase(run=run, active_run=run, results=results, default_top_k=1),
        embedder=FakeEmbedder(),
        query="external commits",
    )

    assert len(response.results) == 1
    assert response.results[0].chunk_id == "fixed:1-introduction:0"


def test_search_chunks_uses_saved_score_threshold_when_present() -> None:
    run = _make_run(is_active=True)
    results = [
        QueryResult(
            chunk_id="fixed:1-introduction:0",
            source="rfc9420.txt",
            section="1 | Introduction",
            content="First chunk",
            score=0.9,
        )
    ]
    response = search_chunks(
        database=FakeDatabase(
            run=run,
            active_run=run,
            results=results,
            default_score_threshold=0.75,
        ),
        embedder=FakeEmbedder(),
        query="external commits",
    )

    assert len(response.results) == 1
    assert response.results[0].chunk_id == "fixed:1-introduction:0"
    assert response.results[0].citations[0].quote == "First chunk"


def test_execute_search_exposes_resolved_metadata_for_logging() -> None:
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

    execution = execute_search(
        database=FakeDatabase(
            run=run,
            active_run=run,
            results=results,
            default_top_k=4,
            default_score_threshold=0.75,
        ),
        embedder=FakeEmbedder(),
        query="external commits",
    )

    assert execution.response.results[0].chunk_id == "fixed:1-introduction:0"
    assert execution.response.results[0].citations[0].chunk_id == "fixed:1-introduction:0"
    assert execution.metadata.requested_top_k is None
    assert execution.metadata.effective_top_k == 4
    assert execution.metadata.similarity_score_threshold == 0.75


def test_search_chunks_extracts_relevant_quote_for_each_result() -> None:
    run = _make_run(is_active=True)
    results = [
        QueryResult(
            chunk_id="fixed:12-1-commit:0",
            source="rfc9420.txt",
            section="12.1 | Commit",
            content=(
                "A Commit message can include proposals from prior epochs. "
                "External commits are Commit messages used to add new members to a group. "
                "Applications can process these commits differently."
            ),
            score=0.92,
        )
    ]

    response = search_chunks(
        database=FakeDatabase(run=run, active_run=run, results=results),
        embedder=FakeEmbedder(),
        query="external commits",
        top_k=5,
    )

    citation = response.results[0].citations[0]
    assert citation.section == "12.1 | Commit"
    assert citation.chunk_id == "fixed:12-1-commit:0"
    assert citation.quote == "External commits are Commit messages used to add new members to a group."
    assert response.results[0].content[citation.quote_start : citation.quote_end] == citation.quote


def test_execute_search_wraps_runtime_failures_with_partial_metadata() -> None:
    run = _make_run(is_active=True)

    with pytest.raises(SearchExecutionError, match="Embedding failed") as exc_info:
        execute_search(
            database=FakeDatabase(
                run=run,
                active_run=run,
                results=[],
                default_top_k=3,
                default_score_threshold=0.8,
            ),
            embedder=FailingEmbedder(),
            query="external commits",
        )

    assert exc_info.value.metadata.requested_top_k is None
    assert exc_info.value.metadata.effective_top_k == 3
    assert exc_info.value.metadata.similarity_score_threshold == 0.8
    assert exc_info.value.run == run


def test_search_chunks_rejects_invalid_saved_score_threshold() -> None:
    run = _make_run(is_active=True)
    with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0 inclusive"):
        search_chunks(
            database=FakeDatabase(
                run=run,
                active_run=run,
                results=[],
                default_score_threshold=1.1,
            ),
            embedder=FakeEmbedder(),
            query="external commits",
        )


def test_validate_score_threshold_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0 inclusive"):
        validate_score_threshold(-0.1)

    with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0 inclusive"):
        validate_score_threshold(1.1)


class FakeDatabase:
    def __init__(
        self,
        *,
        run,
        active_run,
        results,
        default_top_k=None,
        default_score_threshold=None,
    ):
        self._run = run
        self._active_run = active_run
        self._results = results
        self._default_top_k = default_top_k
        self._default_score_threshold = default_score_threshold

    def get_run(self, run_id: int):
        return self._run

    def get_active_run(self):
        return self._active_run

    def get_default_top_k(self):
        return self._default_top_k

    def get_default_score_threshold(self):
        return self._default_score_threshold

    def query_chunks(
        self,
        *,
        run_id: int,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float | None = None,
    ):
        assert query_embedding == [0.5, 0.25]
        assert similarity_threshold == self._default_score_threshold
        return self._results[:top_k]


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        assert text == "external commits"
        return [0.5, 0.25]


class FailingEmbedder:
    def embed_text(self, text: str) -> list[float]:
        assert text == "external commits"
        raise RuntimeError("Embedding failed")


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
