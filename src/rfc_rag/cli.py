from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from rfc_rag.chunking import chunk_sections
from rfc_rag.config import Settings, load_settings
from rfc_rag.db import Database
from rfc_rag.embeddings import OpenAIEmbedder
from rfc_rag.parser import parse_sections


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

    typer.echo(f"Ingested {len(chunks)} chunks into run {run_id}.")


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
                    f"strategy={run.strategy}",
                    f"chunk_size={run.chunk_size}",
                    f"model={run.embedding_model}",
                    f"source={run.source}",
                    f"created_at={run.created_at.isoformat()}",
                ]
            )
        )


@app.command("query")
def query(
    run_id: int = typer.Option(..., min=1, help="Ingestion run ID."),
    query: str = typer.Option(..., help="Search text."),
    top_k: int = typer.Option(5, min=1, help="Number of results to return."),
) -> None:
    """Run a similarity search against one ingestion run."""
    settings = _load_cli_settings(require_openai=True)
    database = Database(settings.database_url)
    run = database.get_run(run_id)
    if run is None:
        raise typer.BadParameter(f"Run {run_id} does not exist.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key or "",
        model=run.embedding_model,
    )
    query_embedding = embedder.embed_text(query)
    results = database.query_chunks(run_id=run_id, query_embedding=query_embedding, top_k=top_k)

    if not results:
        typer.echo("No matching chunks found.")
        return

    for index, result in enumerate(results, start=1):
        typer.echo(f"[{index}] score={result.score:.4f}")
        typer.echo(f"chunk_id={result.chunk_id}")
        typer.echo(f"source={result.source}")
        typer.echo(f"section={result.section}")
        typer.echo(f"preview={_preview_text(result.content)}")
        typer.echo("")


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
