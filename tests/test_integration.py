from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rfc_rag.cli import app
from rfc_rag.chunking import chunk_sections
from rfc_rag.db import Database
from rfc_rag.models import IngestionRun, QueryResult
from rfc_rag.parser import parse_sections
from rfc_rag.search_service import DEFAULT_TOP_K, MAX_TOP_K


RFC_PATH = Path(__file__).resolve().parents[1] / "rfc9420.txt"
RUNNER = CliRunner()


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
    assert run.is_active is False

    results = database.query_chunks(
        run_id=run_id,
        query_embedding=embedder.embed_text("Messaging Layer Security"),
        top_k=3,
        similarity_threshold=None,
    )

    assert results
    assert all(result.chunk_id for result in results)
    assert all(result.source == str(RFC_PATH) for result in results)
    assert all(result.section for result in results)
    assert all(result.content for result in results)


@pytest.mark.integration
def test_active_run_lifecycle_uses_only_one_active_run() -> None:
    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL or DATABASE_URL is required for integration tests.")

    database = Database(database_url)
    database.init_db()

    raw_text = RFC_PATH.read_text(encoding="utf-8-sig")
    sections = parse_sections(raw_text)
    chunks = chunk_sections(sections, source=str(RFC_PATH), strategy="fixed", chunk_size=700)
    sample_chunks = chunks[:4]
    embedder = FakeEmbedder()
    embeddings = embedder.embed_texts([chunk.content for chunk in sample_chunks])

    run_a = database.create_run_with_chunks(
        name="integration-active-a",
        source=str(RFC_PATH),
        strategy="fixed",
        chunk_size=700,
        embedding_model="fake-embedding-model",
        chunks=sample_chunks,
        embeddings=embeddings,
    )
    run_b = database.create_run_with_chunks(
        name="integration-active-b",
        source=str(RFC_PATH),
        strategy="section",
        chunk_size=700,
        embedding_model="fake-embedding-model",
        chunks=sample_chunks,
        embeddings=embeddings,
    )

    activated_a = database.set_active_run(run_a)
    assert activated_a is not None
    assert activated_a.id == run_a
    assert activated_a.is_active is True
    assert database.get_active_run() is not None
    assert database.get_active_run().id == run_a

    activated_b = database.set_active_run(run_b)
    assert activated_b is not None
    assert activated_b.id == run_b
    assert activated_b.is_active is True
    active_run = database.get_active_run()
    assert active_run is not None
    assert active_run.id == run_b
    previous_run = database.get_run(run_a)
    assert previous_run is not None
    assert previous_run.is_active is False


def test_query_json_outputs_run_metadata_and_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    created_at = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_run(self, run_id: int) -> IngestionRun:
            return IngestionRun(
                id=run_id,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=created_at,
                is_active=True,
            )

        def get_active_run(self) -> IngestionRun:
            return self.get_run(11)

        def get_default_top_k(self) -> int | None:
            return None

        def get_default_score_threshold(self) -> float | None:
            return None

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ) -> list[QueryResult]:
            assert run_id == 11
            assert top_k == 2
            assert similarity_threshold is None
            assert query_embedding == _fake_embedding("external commits")
            return [
                QueryResult(
                    chunk_id="fixed:1-introduction:0",
                    source=str(RFC_PATH),
                    section="1 | Introduction",
                    content="Chunk body",
                    score=0.91,
                )
            ]

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits", "--top-k", "2", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run"]["id"] == 11
    assert payload["run"]["created_at"] == created_at.isoformat()
    assert payload["run"]["is_active"] is True
    assert payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"
    assert payload["results"][0]["content"] == "Chunk body"


def test_query_without_run_id_requires_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return None

        def get_default_score_threshold(self) -> float | None:
            return None

        def get_active_run(self):
            return None

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)

    result = RUNNER.invoke(app, ["query", "--query", "external commits"])

    assert result.exit_code != 0
    assert "No active run is configured" in result.output


def test_query_rejects_top_k_above_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return None

        def get_default_score_threshold(self) -> float | None:
            return None

        def get_active_run(self):
            return IngestionRun(
                id=1,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=_created_at(),
                is_active=True,
            )

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits", "--top-k", str(MAX_TOP_K + 1)])

    assert result.exit_code != 0
    assert f"top_k must be less than or equal to {MAX_TOP_K}" in result.output


def test_set_top_k_persists_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def set_default_top_k(self, top_k: int) -> int:
            assert top_k == 8
            return top_k

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)

    result = RUNNER.invoke(app, ["set-top-k", "--top-k", "8"])

    assert result.exit_code == 0
    assert "Default top_k set to 8." in result.output


def test_set_top_k_rejects_top_k_above_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")

    result = RUNNER.invoke(app, ["set-top-k", "--top-k", str(MAX_TOP_K + 1)])

    assert result.exit_code != 0
    assert f"top_k must be less than or equal to {MAX_TOP_K}" in result.output


def test_set_score_threshold_persists_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def set_default_score_threshold(self, score_threshold: float) -> float:
            assert score_threshold == 0.75
            return score_threshold

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)

    result = RUNNER.invoke(app, ["set-score-threshold", "--score-threshold", "0.75"])

    assert result.exit_code == 0
    assert "Default score threshold set to 0.75." in result.output


def test_set_score_threshold_rejects_out_of_range_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")

    below = RUNNER.invoke(app, ["set-score-threshold", "--score-threshold", "-0.01"])
    above = RUNNER.invoke(app, ["set-score-threshold", "--score-threshold", "1.01"])

    assert below.exit_code != 0
    assert "score_threshold must be between 0.0 and 1.0 inclusive" in below.output
    assert above.exit_code != 0
    assert "score_threshold must be between 0.0 and 1.0 inclusive" in above.output


def test_clear_score_threshold_clears_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    state = {"cleared": False}

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def clear_default_score_threshold(self) -> None:
            state["cleared"] = True

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)

    result = RUNNER.invoke(app, ["clear-score-threshold"])

    assert result.exit_code == 0
    assert state["cleared"] is True
    assert "Default score threshold cleared." in result.output


def test_query_uses_saved_default_top_k_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return 3

        def get_default_score_threshold(self) -> float | None:
            return None

        def get_active_run(self):
            return IngestionRun(
                id=1,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=_created_at(),
                is_active=True,
            )

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ) -> list[QueryResult]:
            assert top_k == 3
            assert similarity_threshold is None
            return [
                QueryResult(
                    chunk_id="fixed:1-introduction:0",
                    source=str(RFC_PATH),
                    section="1 | Introduction",
                    content="Chunk body",
                    score=0.91,
                )
            ]

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"


def test_query_explicit_top_k_overrides_saved_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return 4

        def get_default_score_threshold(self) -> float | None:
            return None

        def get_active_run(self):
            return IngestionRun(
                id=1,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=_created_at(),
                is_active=True,
            )

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ) -> list[QueryResult]:
            assert top_k == 2
            assert similarity_threshold is None
            return [
                QueryResult(
                    chunk_id="fixed:1-introduction:0",
                    source=str(RFC_PATH),
                    section="1 | Introduction",
                    content="Chunk body",
                    score=0.91,
                )
            ]

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits", "--top-k", "2", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"


def test_query_uses_builtin_default_top_k_when_nothing_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return None

        def get_default_score_threshold(self) -> float | None:
            return None

        def get_active_run(self):
            return IngestionRun(
                id=1,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=_created_at(),
                is_active=True,
            )

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ) -> list[QueryResult]:
            assert top_k == DEFAULT_TOP_K
            assert similarity_threshold is None
            return [
                QueryResult(
                    chunk_id="fixed:1-introduction:0",
                    source=str(RFC_PATH),
                    section="1 | Introduction",
                    content="Chunk body",
                    score=0.91,
                )
            ]

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"


def test_query_uses_saved_score_threshold_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return None

        def get_default_score_threshold(self) -> float | None:
            return 0.75

        def get_active_run(self):
            return IngestionRun(
                id=1,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=_created_at(),
                is_active=True,
            )

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ) -> list[QueryResult]:
            assert top_k == DEFAULT_TOP_K
            assert similarity_threshold == 0.75
            return [
                QueryResult(
                    chunk_id="fixed:1-introduction:0",
                    source=str(RFC_PATH),
                    section="1 | Introduction",
                    content="Chunk body",
                    score=0.91,
                )
            ]

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"


def test_query_reports_no_results_when_saved_score_threshold_filters_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        def get_default_top_k(self) -> int | None:
            return None

        def get_default_score_threshold(self) -> float | None:
            return 0.95

        def get_active_run(self):
            return IngestionRun(
                id=1,
                name="active-run",
                source=str(RFC_PATH),
                strategy="fixed",
                chunk_size=700,
                embedding_model="fake-embedding-model",
                created_at=_created_at(),
                is_active=True,
            )

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ) -> list[QueryResult]:
            assert similarity_threshold == 0.95
            return []

    class FakeOpenAIEmbedder:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str) -> list[float]:
            return _fake_embedding(text)

    monkeypatch.setattr("rfc_rag.cli.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.cli.OpenAIEmbedder", FakeOpenAIEmbedder)

    result = RUNNER.invoke(app, ["query", "--query", "external commits"])

    assert result.exit_code == 0
    assert "No matching chunks found." in result.output


def _created_at() -> datetime:
    return datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
