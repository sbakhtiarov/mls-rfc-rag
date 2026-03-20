from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import re
from typing import Protocol

from rfc_rag.db import Database
from rfc_rag.models import Citation, IngestionRun, QueryResult


DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MAX_CITATION_LENGTH = 280
MIN_QUERY_TOKEN_LENGTH = 3
_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?](?:\s+|$)|[^.!?]+$")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


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
        raw_results = database.query_chunks(
            run_id=run.id,
            query_embedding=query_embedding,
            top_k=active_metadata.effective_top_k or DEFAULT_TOP_K,
            similarity_threshold=active_metadata.similarity_score_threshold,
        )
        results = _attach_citations(results=raw_results, query=query)
        return SearchExecution(
            response=SearchResponse(run=run, results=results),
            metadata=active_metadata,
        )
    except Exception as exc:
        raise _SearchStateError(metadata=active_metadata, run=run) from exc


def _attach_citations(*, results: list[QueryResult], query: str) -> list[QueryResult]:
    return [replace(result, citations=[_extract_citation(result=result, query=query)]) for result in results]


def _extract_citation(*, result: QueryResult, query: str) -> Citation:
    content = result.content
    quote_start, quote_end = _select_quote_span(content=content, query=query)
    quote = content[quote_start:quote_end].strip()
    if not quote:
        quote = content.strip()
        quote_start = content.find(quote) if quote else 0
        quote_end = quote_start + len(quote)

    return Citation(
        source=result.source,
        section=result.section,
        chunk_id=result.chunk_id,
        quote=quote,
        quote_start=quote_start,
        quote_end=quote_end,
    )


def _select_quote_span(*, content: str, query: str) -> tuple[int, int]:
    candidates = _build_quote_candidates(content)
    if not candidates:
        return (0, len(content))

    query_tokens = _query_tokens(query)
    best_candidate = max(
        candidates,
        key=lambda candidate: _score_candidate(candidate[2], query_tokens),
    )
    return _trim_span(
        content=content,
        start=best_candidate[0],
        end=best_candidate[1],
        query_tokens=query_tokens,
    )


def _build_quote_candidates(content: str) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []

    for block_start, block_end in _iter_block_spans(content):
        block_text = content[block_start:block_end]
        if not block_text.strip():
            continue

        sentence_spans = _iter_sentence_spans(block_text, block_start)
        if sentence_spans:
            candidates.extend(sentence_spans)
            if len(sentence_spans) > 1:
                for first, second in zip(sentence_spans, sentence_spans[1:], strict=False):
                    combined_text = content[first[0] : second[1]]
                    candidates.append((first[0], second[1], combined_text))
        else:
            candidates.append((block_start, block_end, block_text))

    if not candidates and content.strip():
        candidates.append((0, len(content), content))

    return candidates


def _iter_block_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BLOCK_SPLIT_PATTERN.finditer(content):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(content)))
    return spans


def _iter_sentence_spans(block_text: str, block_start: int) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _SENTENCE_PATTERN.finditer(block_text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        leading_padding = len(match.group(0)) - len(match.group(0).lstrip())
        absolute_start = block_start + match.start() + leading_padding
        absolute_end = absolute_start + len(sentence)
        spans.append((absolute_start, absolute_end, sentence))
    return spans


def _query_tokens(text: str) -> set[str]:
    return {
        token
        for token in _QUERY_TOKEN_PATTERN.findall(text.lower())
        if len(token) >= MIN_QUERY_TOKEN_LENGTH and token not in _STOP_WORDS
    }


def _score_candidate(candidate_text: str, query_tokens: set[str]) -> tuple[float, int, int]:
    candidate_tokens = _query_tokens(candidate_text)
    overlap = query_tokens & candidate_tokens
    overlap_score = float(len(overlap))
    exact_phrase_bonus = 0
    lowered_candidate = candidate_text.lower()
    for token in query_tokens:
        if token in lowered_candidate:
            exact_phrase_bonus += 1
    nonempty_bonus = 1 if candidate_text.strip() else 0
    return (
        overlap_score * 10 + exact_phrase_bonus,
        nonempty_bonus,
        -len(candidate_text.strip()),
    )


def _trim_span(
    *,
    content: str,
    start: int,
    end: int,
    query_tokens: set[str],
) -> tuple[int, int]:
    quote = content[start:end].strip()
    if len(quote) <= MAX_CITATION_LENGTH:
        return _normalize_span(content=content, start=start, end=end)

    if not query_tokens:
        return _normalize_span(content=content, start=start, end=min(start + MAX_CITATION_LENGTH, end))

    lowered_quote = quote.lower()
    anchor = next((lowered_quote.find(token) for token in query_tokens if lowered_quote.find(token) >= 0), -1)
    if anchor < 0:
        return _normalize_span(content=content, start=start, end=min(start + MAX_CITATION_LENGTH, end))

    window_start = max(0, anchor - MAX_CITATION_LENGTH // 3)
    window_end = min(len(quote), window_start + MAX_CITATION_LENGTH)
    relative_start = _advance_to_word_boundary(quote, window_start, forward=True)
    relative_end = _advance_to_word_boundary(quote, window_end, forward=False)
    trimmed_start = start + relative_start
    trimmed_end = start + max(relative_end, relative_start + 1)
    return _normalize_span(content=content, start=trimmed_start, end=trimmed_end)


def _normalize_span(*, content: str, start: int, end: int) -> tuple[int, int]:
    bounded_start = max(0, min(start, len(content)))
    bounded_end = max(bounded_start, min(end, len(content)))
    while bounded_start < bounded_end and content[bounded_start].isspace():
        bounded_start += 1
    while bounded_end > bounded_start and content[bounded_end - 1].isspace():
        bounded_end -= 1
    return (bounded_start, bounded_end)


def _advance_to_word_boundary(text: str, index: int, *, forward: bool) -> int:
    bounded_index = max(0, min(index, len(text)))
    if forward:
        while bounded_index < len(text) and not text[bounded_index].isalnum():
            bounded_index += 1
    else:
        while bounded_index > 0 and not text[bounded_index - 1].isalnum():
            bounded_index -= 1
    return bounded_index
