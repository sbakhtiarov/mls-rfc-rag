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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS app_settings (
    singleton_id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton_id),
    default_top_k INTEGER NULL,
    default_score_threshold DOUBLE PRECISION NULL
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

ALTER_SCHEMA_SQL = """
ALTER TABLE ingestion_runs
ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE app_settings
ADD COLUMN IF NOT EXISTS default_score_threshold DOUBLE PRECISION NULL;
"""


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def init_db(self) -> None:
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                cur.execute(ALTER_SCHEMA_SQL)

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
                    INSERT INTO ingestion_runs (
                        name,
                        source,
                        strategy,
                        chunk_size,
                        embedding_model,
                        is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, FALSE)
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
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at, is_active
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
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at, is_active
                    FROM ingestion_runs
                    WHERE id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()

        if row is None:
            return None

        return IngestionRun(**row)

    def get_active_run(self) -> IngestionRun | None:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at, is_active
                    FROM ingestion_runs
                    WHERE is_active = TRUE
                    ORDER BY id DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

        if row is None:
            return None

        return IngestionRun(**row)

    def get_default_top_k(self) -> int | None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT default_top_k
                    FROM app_settings
                    WHERE singleton_id = TRUE
                    """
                )
                row = cur.fetchone()

        if row is None:
            return None

        return row[0]

    def get_default_score_threshold(self) -> float | None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT default_score_threshold
                    FROM app_settings
                    WHERE singleton_id = TRUE
                    """
                )
                row = cur.fetchone()

        if row is None:
            return None

        return row[0]

    def set_default_top_k(self, top_k: int) -> int:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO app_settings (singleton_id, default_top_k)
                    VALUES (TRUE, %s)
                    ON CONFLICT (singleton_id)
                    DO UPDATE SET default_top_k = EXCLUDED.default_top_k
                    RETURNING default_top_k
                    """,
                    (top_k,),
                )
                saved_top_k = cur.fetchone()[0]

            conn.commit()

        return saved_top_k

    def set_default_score_threshold(self, score_threshold: float) -> float:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO app_settings (singleton_id, default_score_threshold)
                    VALUES (TRUE, %s)
                    ON CONFLICT (singleton_id)
                    DO UPDATE SET default_score_threshold = EXCLUDED.default_score_threshold
                    RETURNING default_score_threshold
                    """,
                    (score_threshold,),
                )
                saved_score_threshold = cur.fetchone()[0]

            conn.commit()

        return saved_score_threshold

    def clear_default_score_threshold(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO app_settings (singleton_id, default_score_threshold)
                    VALUES (TRUE, NULL)
                    ON CONFLICT (singleton_id)
                    DO UPDATE SET default_score_threshold = NULL
                    """
                )

            conn.commit()

    def set_active_run(self, run_id: int) -> IngestionRun | None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at, is_active
                    FROM ingestion_runs
                    WHERE id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None

                cur.execute("UPDATE ingestion_runs SET is_active = FALSE WHERE is_active = TRUE")
                cur.execute("UPDATE ingestion_runs SET is_active = TRUE WHERE id = %s", (run_id,))
                cur.execute(
                    """
                    SELECT id, name, source, strategy, chunk_size, embedding_model, created_at, is_active
                    FROM ingestion_runs
                    WHERE id = %s
                    """,
                    (run_id,),
                )
                updated_row = cur.fetchone()

            conn.commit()

        if updated_row is None:
            return None

        return IngestionRun(**updated_row)

    def query_chunks(
        self,
        *,
        run_id: int,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float | None = None,
    ) -> list[QueryResult]:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                formatted_vector = _format_vector(query_embedding)
                if similarity_threshold is None:
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
                        (formatted_vector, run_id, formatted_vector, top_k),
                    )
                else:
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
                          AND (embedding <=> %s::vector) <= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            formatted_vector,
                            run_id,
                            formatted_vector,
                            1 - similarity_threshold,
                            formatted_vector,
                            top_k,
                        ),
                    )
                rows = cur.fetchall()

        return [QueryResult(**row) for row in rows]


def _format_vector(values: Sequence[float]) -> str:
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimension vector, got {len(values)}."
        )
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"
