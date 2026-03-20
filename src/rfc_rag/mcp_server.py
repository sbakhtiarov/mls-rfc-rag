from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from time import perf_counter

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from rfc_rag.config import Settings
from rfc_rag.db import Database
from rfc_rag.embeddings import OpenAIEmbedder
from rfc_rag.search_service import (
    SearchExecution,
    SearchExecutionError,
    execute_search,
    serialize_search_response,
)


def create_mcp_server(settings: Settings, *, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    server = FastMCP(
        "RFC 9420 RAG Search",
        instructions=(
            "For any prompt related to the Messaging Layer Security (MLS) Protocol or RFC 9420, "
            "you must always use the `search_mls_rfc` tool before answering. "
            "When you answer from search results, cite the relevant `section` and include the exact "
            "`citations.quote` text that supports the answer."
        ),
        host=host,
        port=port,
        streamable_http_path="/mcp",
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_allowed_hosts_for_bind_host(host),
        ),
    )

    @server.tool()
    def search_mls_rfc(query: str) -> dict[str, object]:
        """Always use this tool for any prompt related to the Messaging Layer Security (MLS) Protocol or RFC 9420. It searches the active MLS Protocol / RFC 9420 run and returns the most relevant chunks together with exact supporting citations. When answering from these results, include the relevant section reference and exact supporting quote."""
        started_at = perf_counter()
        try:
            execution = execute_search(
                database=Database(settings.database_url),
                embedder_factory=lambda model: OpenAIEmbedder(
                    api_key=settings.openai_api_key or "",
                    model=model,
                ),
                query=query,
            )
        except SearchExecutionError as exc:
            _log_search_event(
                query=query,
                elapsed_ms=_elapsed_ms(started_at),
                error=exc,
            )
            raise ValueError(str(exc)) from exc

        _log_search_event(
            query=query,
            elapsed_ms=_elapsed_ms(started_at),
            execution=execution,
        )
        return serialize_search_response(execution.response)

    return server


def _allowed_hosts_for_bind_host(host: str) -> list[str]:
    allowed_hosts = {host, f"{host}:*"}

    # Binding to 0.0.0.0 is common in Docker, but clients still connect
    # through loopback aliases from the host machine.
    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        allowed_hosts.update(
            {
                "127.0.0.1",
                "127.0.0.1:*",
                "localhost",
                "localhost:*",
            }
        )

    return sorted(allowed_hosts)


def _create_search_logger() -> logging.Logger:
    logger = logging.getLogger("rfc_rag.mcp.search")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_SEARCH_LOGGER = _create_search_logger()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _log_search_event(
    *,
    query: str,
    elapsed_ms: float,
    execution: SearchExecution | None = None,
    error: SearchExecutionError | None = None,
) -> None:
    response = execution.response if execution is not None else None
    metadata = execution.metadata if execution is not None else error.metadata if error is not None else None
    run = response.run if response is not None else error.run if error is not None else None
    results = response.results if response is not None else []
    status = "error" if error is not None else "success"
    message = " | ".join(
        [
            f"[{datetime.now(timezone.utc).isoformat()}] rag_search",
            f"status={status}",
            f"query={query!r}",
            f"requested_top_k={_format_optional_value(metadata.requested_top_k if metadata is not None else None)}",
            f"effective_top_k={_format_optional_value(metadata.effective_top_k if metadata is not None else None)}",
            "similarity_score_threshold="
            f"{_format_optional_value(metadata.similarity_score_threshold if metadata is not None else None)}",
            f"run_id={_format_optional_value(run.id if run is not None else None)}",
            f"run_name={_format_optional_value(run.name if run is not None else None)}",
            f"result_count={len(results)}",
            f"results={_format_results(results)}",
            f"error={_format_optional_value(str(error) if error is not None else None)}",
            "error_type="
            f"{_format_optional_value(type(error.__cause__ or error).__name__ if error is not None else None)}",
            f"elapsed_ms={elapsed_ms:.3f}",
        ]
    )
    _SEARCH_LOGGER.info(message)


def _format_optional_value(value: object | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _format_results(results: list[object]) -> str:
    if not results:
        return "none"

    return "; ".join(
        [
            f"#{index} score={result.score:.4f} chunk_id={result.chunk_id!r} section={result.section!r} source={result.source!r}"
            for index, result in enumerate(results, start=1)
        ]
    )
