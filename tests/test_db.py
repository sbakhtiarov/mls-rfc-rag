from __future__ import annotations

from rfc_rag.db import Database, SCHEMA_SQL


def test_init_db_executes_schema_with_app_settings(monkeypatch) -> None:
    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_connect(dsn: str, **kwargs):
        calls.append((dsn, kwargs))
        return connection

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", fake_connect)

    Database("postgresql://example").init_db()

    assert calls == [("postgresql://example", {"autocommit": True})]
    assert cursor.executed[0][0] == SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS app_settings" in cursor.executed[0][0]
    assert "default_score_threshold DOUBLE PRECISION NULL" in cursor.executed[0][0]


def test_get_default_top_k_returns_saved_value(monkeypatch) -> None:
    cursor = RecordingCursor(fetchone_result=(8,))
    connection = RecordingConnection(cursor)

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", lambda dsn, **kwargs: connection)

    top_k = Database("postgresql://example").get_default_top_k()

    assert top_k == 8
    assert "SELECT default_top_k" in cursor.executed[0][0]


def test_set_default_top_k_upserts_and_commits(monkeypatch) -> None:
    cursor = RecordingCursor(fetchone_result=(6,))
    connection = RecordingConnection(cursor)

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", lambda dsn, **kwargs: connection)

    top_k = Database("postgresql://example").set_default_top_k(6)

    assert top_k == 6
    assert connection.committed is True
    assert "INSERT INTO app_settings" in cursor.executed[0][0]
    assert cursor.executed[0][1] == (6,)


def test_get_default_score_threshold_returns_saved_value(monkeypatch) -> None:
    cursor = RecordingCursor(fetchone_result=(0.75,))
    connection = RecordingConnection(cursor)

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", lambda dsn, **kwargs: connection)

    score_threshold = Database("postgresql://example").get_default_score_threshold()

    assert score_threshold == 0.75
    assert "SELECT default_score_threshold" in cursor.executed[0][0]


def test_set_default_score_threshold_upserts_and_commits(monkeypatch) -> None:
    cursor = RecordingCursor(fetchone_result=(0.8,))
    connection = RecordingConnection(cursor)

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", lambda dsn, **kwargs: connection)

    score_threshold = Database("postgresql://example").set_default_score_threshold(0.8)

    assert score_threshold == 0.8
    assert connection.committed is True
    assert "INSERT INTO app_settings" in cursor.executed[0][0]
    assert cursor.executed[0][1] == (0.8,)


def test_clear_default_score_threshold_upserts_null_and_commits(monkeypatch) -> None:
    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", lambda dsn, **kwargs: connection)

    Database("postgresql://example").clear_default_score_threshold()

    assert connection.committed is True
    assert "INSERT INTO app_settings" in cursor.executed[0][0]
    assert "DO UPDATE SET default_score_threshold = NULL" in cursor.executed[0][0]


def test_query_chunks_uses_similarity_threshold_when_present(monkeypatch) -> None:
    cursor = RecordingCursor(fetchall_result=[])
    connection = RecordingConnection(cursor)

    monkeypatch.setattr("rfc_rag.db.psycopg.connect", lambda dsn, **kwargs: connection)

    results = Database("postgresql://example").query_chunks(
        run_id=4,
        query_embedding=[0.1] * 1536,
        top_k=3,
        similarity_threshold=0.75,
    )

    assert results == []
    assert "AND (embedding <=> %s::vector) <= %s" in cursor.executed[0][0]
    assert cursor.executed[0][1][3] == 0.25
    assert cursor.executed[0][1][-2:] == (
        cursor.executed[0][1][0],
        3,
    )


class RecordingConnection:
    def __init__(self, cursor: "RecordingCursor") -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self, *args, **kwargs):
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> "RecordingConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class RecordingCursor:
    def __init__(self, *, fetchone_result=None, fetchall_result=None) -> None:
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []
        self._fetchone_result = fetchone_result
        self._fetchall_result = fetchall_result or []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetchone_result

    def fetchall(self):
        return self._fetchall_result

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
