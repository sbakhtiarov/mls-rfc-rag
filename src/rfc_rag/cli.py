from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import typer

from rfc_rag.chunking import chunk_sections
from rfc_rag.config import Settings, load_settings
from rfc_rag.db import Database
from rfc_rag.embeddings import OpenAIEmbedder
from rfc_rag.mcp_server import create_mcp_server
from rfc_rag.parser import parse_sections
from rfc_rag.search_service import (
    SearchResponse,
    serialize_search_response,
    search_chunks,
    validate_score_threshold,
    validate_top_k,
)


app = typer.Typer(help="RFC 9420 RAG experimentation CLI.")


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    SECTION = "section"


@app.command("init-db")
def init_db() -> None:
    """Create the database schema and pgvector index."""
    settings = _load_cli_settings(require_openai=False)
    Database(settings.database_url).init_db()
    typer.echo("Database initialized.")


@app.command("ingest")
def ingest(
    source: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    strategy: ChunkStrategy = typer.Option(..., help="Chunking strategy."),
    chunk_size: int = typer.Option(..., min=1, help="Chunk size in characters."),
    name: str | None = typer.Option(None, help="Optional label for this ingestion run."),
) -> None:
    """Parse, chunk, embed, and store a source file."""
    settings = _load_cli_settings(require_openai=True)
    raw_text = source.read_text(encoding="utf-8-sig")
    sections = parse_sections(raw_text)
    source_label = str(source)
    chunks = chunk_sections(
        sections,
        source=source_label,
        strategy=strategy.value,
        chunk_size=chunk_size,
    )

    if not chunks:
        raise typer.BadParameter("No chunks were produced from the source document.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key or "",
        model=settings.openai_embed_model,
    )
    embeddings = embedder.embed_texts([chunk.content for chunk in chunks])

    run_name = name or f"{source.stem}-{strategy.value}-{chunk_size}"
    run_id = Database(settings.database_url).create_run_with_chunks(
        name=run_name,
        source=source_label,
        strategy=strategy.value,
        chunk_size=chunk_size,
        embedding_model=settings.openai_embed_model,
        chunks=chunks,
        embeddings=embeddings,
    )

    typer.echo(f"Ingested {len(chunks)} chunks into run {run_id}. Active: no.")


@app.command("list-runs")
def list_runs() -> None:
    """List ingestion runs."""
    settings = _load_cli_settings(require_openai=False)
    runs = Database(settings.database_url).list_runs()
    if not runs:
        typer.echo("No ingestion runs found.")
        return

    for run in runs:
        typer.echo(
            " | ".join(
                [
                    f"id={run.id}",
                    f"name={run.name}",
                    f"active={'yes' if run.is_active else 'no'}",
                    f"strategy={run.strategy}",
                    f"chunk_size={run.chunk_size}",
                    f"model={run.embedding_model}",
                    f"source={run.source}",
                    f"created_at={run.created_at.isoformat()}",
                ]
            )
        )


@app.command("set-active-run")
def set_active_run(
    run_id: int = typer.Option(..., min=1, help="Ingestion run ID to mark as active."),
) -> None:
    """Mark one ingestion run as the active default for queries."""
    settings = _load_cli_settings(require_openai=False)
    run = Database(settings.database_url).set_active_run(run_id)
    if run is None:
        raise typer.BadParameter(f"Run {run_id} does not exist.")

    typer.echo(f"Active run set to {run.id} ({run.name}).")


@app.command("set-top-k")
def set_top_k(
    top_k: int = typer.Option(..., help="Default number of results to return."),
) -> None:
    """Persist the default top_k used when a query does not provide one."""
    settings = _load_cli_settings(require_openai=False)
    try:
        validated_top_k = validate_top_k(top_k)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    saved_top_k = Database(settings.database_url).set_default_top_k(validated_top_k)
    typer.echo(f"Default top_k set to {saved_top_k}.")


@app.command("set-score-threshold")
def set_score_threshold(
    score_threshold: float = typer.Option(
        ...,
        help="Default similarity score threshold to apply to query results.",
    ),
) -> None:
    """Persist the default score threshold used to filter query results."""
    settings = _load_cli_settings(require_openai=False)
    try:
        validated_score_threshold = validate_score_threshold(score_threshold)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    saved_score_threshold = Database(settings.database_url).set_default_score_threshold(
        validated_score_threshold
    )
    typer.echo(f"Default score threshold set to {saved_score_threshold:g}.")


@app.command("clear-score-threshold")
def clear_score_threshold() -> None:
    """Remove the default score threshold so queries return unfiltered results."""
    settings = _load_cli_settings(require_openai=False)
    Database(settings.database_url).clear_default_score_threshold()
    typer.echo("Default score threshold cleared.")


@app.command("query")
def query(
    run_id: int | None = typer.Option(None, min=1, help="Ingestion run ID."),
    query: str = typer.Option(..., help="Search text."),
    top_k: int | None = typer.Option(None, help="Number of results to return."),
    json_output: bool = typer.Option(False, "--json", help="Return structured JSON output."),
) -> None:
    """Run a similarity search against one ingestion run."""
    settings = _load_cli_settings(require_openai=True)
    database = Database(settings.database_url)
    try:
        response = search_chunks(
            database=database,
            embedder_factory=lambda model: OpenAIEmbedder(
                api_key=settings.openai_api_key or "",
                model=model,
            ),
            query=query,
            top_k=top_k,
            run_id=run_id,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        typer.echo(json.dumps(serialize_search_response(response), indent=2))
        return

    if not response.results:
        if json_output:
            typer.echo(json.dumps(serialize_search_response(response), indent=2))
            return
        typer.echo("No matching chunks found.")
        return

    for index, result in enumerate(response.results, start=1):
        typer.echo(f"[{index}] score={result.score:.4f}")
        typer.echo(f"chunk_id={result.chunk_id}")
        typer.echo(f"source={result.source}")
        typer.echo(f"section={result.section}")
        if result.citations:
            typer.echo(f"citation_quote={_preview_text(result.citations[0].quote)}")
        typer.echo(f"preview={_preview_text(result.content)}")
        typer.echo("")


@app.command("serve-mcp")
def serve_mcp(
    host: str = typer.Option("127.0.0.1", help="Host interface for the MCP HTTP server."),
    port: int = typer.Option(8000, min=1, max=65535, help="Port for the MCP HTTP server."),
) -> None:
    """Serve the RAG search tool as an MCP streamable HTTP server."""
    settings = _load_cli_settings(require_openai=True)
    server = create_mcp_server(settings, host=host, port=port)
    server.run(transport="streamable-http")


def main() -> None:
    app()


def _load_cli_settings(*, require_openai: bool) -> Settings:
    try:
        return load_settings(require_openai=require_openai)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _preview_text(content: str, limit: int = 200) -> str:
    condensed = " ".join(content.split())
    if len(condensed) <= limit:
        return condensed
    return condensed[: limit - 3].rstrip() + "..."
