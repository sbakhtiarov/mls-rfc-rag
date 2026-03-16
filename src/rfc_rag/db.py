from __future__ import annotations

from collections.abc import Sequence

import psycopg
from psycopg.rows import dict_row

from rfc_rag.config import EMBEDDING_DIMENSION
from rfc_rag.models import Chunk, IngestionRun, QueryResult


SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    strategy TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES ingestion_runs(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
    UNIQUE (run_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS chunks_run_id_idx ON chunks (run_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_cosine_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def init_db(self) -> None:
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)

    def create_run_with_chunks(
        self,
        *,
        name: str,
        source: str,
        strategy: str,
        chunk_size: int,
        embedding_model: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_runs (name, source, strategy, chunk_size, embedding_model)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (name, source, strategy, chunk_size, embedding_model),
                )
                run_id = cur.fetchone()[0]

                rows = [
                    (
                        run_id,
                        chunk.chunk_id,
                        chunk.source,
                        chunk.section,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.char_count,
                        _format_vector(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ]
                cur.executemany(
                    """
                    INSERT INTO chunks (
                        run_id,
                        chunk_id,
                        source,
                        section,
                        chunk_index,
                        content,
                        char_count,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    rows,
                )

            conn.commit()

        return run_id

    def list_runs(self) -> list[IngestionRun]:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at
                    FROM ingestion_runs
                    ORDER BY created_at DESC, id DESC
                    """
                )
                rows = cur.fetchall()

        return [IngestionRun(**row) for row in rows]

    def get_run(self, run_id: int) -> IngestionRun | None:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at
                    FROM ingestion_runs
                    WHERE id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()

        if row is None:
            return None

        return IngestionRun(**row)

    def query_chunks(
        self,
        *,
        run_id: int,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QueryResult]:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        chunk_id,
                        source,
                        section,
                        content,
                        1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE run_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (_format_vector(query_embedding), run_id, _format_vector(query_embedding), top_k),
                )
                rows = cur.fetchall()

        return [QueryResult(**row) for row in rows]


def _format_vector(values: Sequence[float]) -> str:
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimension vector, got {len(values)}."
        )
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"
