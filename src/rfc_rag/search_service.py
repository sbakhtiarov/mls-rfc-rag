from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Protocol

from rfc_rag.db import Database
from rfc_rag.models import IngestionRun, QueryResult


DEFAULT_TOP_K = 5
MAX_TOP_K = 20


class EmbedderProtocol(Protocol):
    def embed_text(self, text: str) -> list[float]: ...


class EmbedderFactoryProtocol(Protocol):
    def __call__(self, model: str) -> EmbedderProtocol: ...


@dataclass(frozen=True)
class SearchResponse:
    run: IngestionRun
    results: list[QueryResult]


@dataclass(frozen=True)
class SearchMetadata:
    requested_top_k: int | None
    effective_top_k: int | None
    similarity_score_threshold: float | None


@dataclass(frozen=True)
class SearchExecution:
    response: SearchResponse
    metadata: SearchMetadata


class SearchExecutionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        metadata: SearchMetadata,
        run: IngestionRun | None = None,
    ) -> None:
        super().__init__(message)
        self.metadata = metadata
        self.run = run


def search_chunks(
    *,
    database: Database,
    embedder: EmbedderProtocol | None = None,
    embedder_factory: EmbedderFactoryProtocol | None = None,
    query: str,
    top_k: int | None = None,
    run_id: int | None = None,
) -> SearchResponse:
    try:
        return _perform_search(
            database=database,
            embedder=embedder,
            embedder_factory=embedder_factory,
            query=query,
            top_k=top_k,
            run_id=run_id,
        ).response
    except _SearchStateError as exc:
        cause = exc.__cause__
        if cause is None:
            raise
        raise cause from exc


def execute_search(
    *,
    database: Database,
    embedder: EmbedderProtocol | None = None,
    embedder_factory: EmbedderFactoryProtocol | None = None,
    query: str,
    top_k: int | None = None,
    run_id: int | None = None,
) -> SearchExecution:
    metadata = SearchMetadata(
        requested_top_k=top_k,
        effective_top_k=None,
        similarity_score_threshold=None,
    )
    run: IngestionRun | None = None

    try:
        return _perform_search(
            database=database,
            embedder=embedder,
            embedder_factory=embedder_factory,
            query=query,
            top_k=top_k,
            run_id=run_id,
            metadata=metadata,
        )
    except Exception as exc:
        if isinstance(exc, SearchExecutionError):
            raise
        if isinstance(exc, _SearchStateError):
            metadata = exc.metadata
            run = exc.run
            message = str(exc.__cause__ or exc)
        else:
            message = str(exc)
        raise SearchExecutionError(message, metadata=metadata, run=run) from exc


def serialize_search_response(response: SearchResponse) -> dict[str, object]:
    return {
        "run": _serialize_run(response.run),
        "results": [_serialize_result(result) for result in response.results],
    }


def resolve_top_k(*, database: Database, top_k: int | None) -> int:
    if top_k is not None:
        return validate_top_k(top_k)

    default_top_k = database.get_default_top_k()
    if default_top_k is not None:
        return validate_top_k(default_top_k)

    return DEFAULT_TOP_K


def resolve_score_threshold(*, database: Database) -> float | None:
    score_threshold = database.get_default_score_threshold()
    if score_threshold is None:
        return None
    return validate_score_threshold(score_threshold)


def validate_top_k(top_k: int) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be less than or equal to {MAX_TOP_K}.")
    return top_k


def validate_score_threshold(score_threshold: float) -> float:
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise ValueError("score_threshold must be between 0.0 and 1.0 inclusive.")
    return score_threshold


def _serialize_run(run: IngestionRun) -> dict[str, object]:
    payload = asdict(run)
    payload["created_at"] = _serialize_datetime(run.created_at)
    return payload


def _serialize_result(result: QueryResult) -> dict[str, object]:
    return asdict(result)


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


@dataclass(frozen=True)
class _SearchStateError(Exception):
    metadata: SearchMetadata
    run: IngestionRun | None = None


def _perform_search(
    *,
    database: Database,
    embedder: EmbedderProtocol | None = None,
    embedder_factory: EmbedderFactoryProtocol | None = None,
    query: str,
    top_k: int | None = None,
    run_id: int | None = None,
    metadata: SearchMetadata | None = None,
) -> SearchExecution:
    active_metadata = metadata or SearchMetadata(
        requested_top_k=top_k,
        effective_top_k=None,
        similarity_score_threshold=None,
    )
    run: IngestionRun | None = None

    try:
        validated_top_k = resolve_top_k(database=database, top_k=top_k)
        active_metadata = replace(active_metadata, effective_top_k=validated_top_k)
        similarity_threshold = resolve_score_threshold(database=database)
        active_metadata = replace(
            active_metadata,
            similarity_score_threshold=similarity_threshold,
        )
        run = database.get_run(run_id) if run_id is not None else database.get_active_run()
        if run is None:
            if run_id is not None:
                raise ValueError(f"Run {run_id} does not exist.")
            raise ValueError("No active run is configured. Run `rfc-rag set-active-run --run-id <id>`.")

        active_embedder = embedder
        if active_embedder is None:
            if embedder_factory is None:
                raise ValueError("An embedder or embedder_factory is required.")
            active_embedder = embedder_factory(run.embedding_model)

        query_embedding = active_embedder.embed_text(query)
        results = database.query_chunks(
            run_id=run.id,
            query_embedding=query_embedding,
            top_k=active_metadata.effective_top_k or DEFAULT_TOP_K,
            similarity_threshold=active_metadata.similarity_score_threshold,
        )
        return SearchExecution(
            response=SearchResponse(run=run, results=results),
            metadata=active_metadata,
        )
    except Exception as exc:
        raise _SearchStateError(metadata=active_metadata, run=run) from exc
