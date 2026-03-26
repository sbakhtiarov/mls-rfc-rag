from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import logging

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from typer.testing import CliRunner

from rfc_rag.cli import app
from rfc_rag.mcp_server import _allowed_hosts_for_bind_host, create_mcp_server
from rfc_rag.models import IngestionRun, QueryResult


RUNNER = CliRunner()


def test_serve_mcp_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    result = RUNNER.invoke(app, ["serve-mcp"])

    assert result.exit_code != 0
    assert "DATABASE_URL is required" in result.output


def test_allowed_hosts_include_loopback_aliases_for_local_and_docker_bindings() -> None:
    docker_hosts = _allowed_hosts_for_bind_host("0.0.0.0")
    assert "0.0.0.0" in docker_hosts
    assert "0.0.0.0:*" in docker_hosts
    assert "127.0.0.1" in docker_hosts
    assert "127.0.0.1:*" in docker_hosts
    assert "localhost" in docker_hosts
    assert "localhost:*" in docker_hosts

    local_hosts = _allowed_hosts_for_bind_host("127.0.0.1")
    assert "127.0.0.1" in local_hosts
    assert "localhost" in local_hosts


def test_mcp_search_tool_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    log_stream = io.StringIO()
    logger = logging.getLogger("test.rfc_rag.mcp.search")
    logger.handlers.clear()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    run = IngestionRun(
        id=7,
        name="active-run",
        source="rfc9420.txt",
        strategy="fixed",
        chunk_size=700,
        embedding_model="fake-embedding-model",
        created_at=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc),
        is_active=True,
    )

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn
            self.return_active = True

        def get_run(self, run_id: int):
            return None

        def get_active_run(self):
            return run if self.return_active else None

        def get_default_top_k(self):
            return 3

        def get_default_score_threshold(self):
            return 0.9

        def query_chunks(
            self,
            *,
            run_id: int,
            query_embedding: list[float],
            top_k: int,
            similarity_threshold: float | None = None,
        ):
            if query_embedding == ["empty"]:
                assert top_k == 3
                assert similarity_threshold == 0.9
                return []
            if query_embedding == ["thresholded"]:
                assert top_k == 3
                assert similarity_threshold == 0.9
                return [
                    QueryResult(
                        chunk_id="fixed:1-introduction:0",
                        source="rfc9420.txt",
                        section="1 | Introduction",
                        content="Chunk body",
                        score=0.91,
                    )
                ]
            assert top_k == 3
            assert similarity_threshold == 0.9
            return [
                QueryResult(
                    chunk_id="fixed:1-introduction:0",
                    source="rfc9420.txt",
                    section="1 | Introduction",
                    content="Chunk body",
                    score=0.91,
                )
            ]

    class FakeEmbedder:
        def __init__(self, host: str, model: str) -> None:
            self.host = host
            self.model = model

        def embed_text(self, text: str):
            if text == "empty":
                return ["empty"]
            if text == "thresholded":
                return ["thresholded"]
            if text == "boom":
                raise RuntimeError("Embedding failed")
            return [0.1, 0.2]

    monkeypatch.setattr("rfc_rag.mcp_server.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.mcp_server.OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr("rfc_rag.mcp_server._SEARCH_LOGGER", logger)

    server = create_mcp_server(
        settings=_settings(),
        host="127.0.0.1",
        port=8001,
    )
    assert server.instructions is not None
    assert "cite the relevant `section`" in server.instructions
    assert "`citations.quote`" in server.instructions
    tool = next(tool for tool in server._tool_manager.list_tools() if tool.name == "search_mls_rfc")
    assert set(tool.parameters["properties"]) == {"query"}
    assert tool.parameters["required"] == ["query"]

    async def scenario() -> None:
        async with _test_client(server) as session:
            tools = await session.list_tools()
            search_tool = next(tool for tool in tools.tools if tool.name == "search_mls_rfc")
            assert "Messaging Layer Security" in (search_tool.description or "")
            assert "RFC 9420" in (search_tool.description or "")
            assert "Always use this tool" in (search_tool.description or "")
            assert "exact supporting citations" in (search_tool.description or "")
            assert "section reference" in (search_tool.description or "")
            assert "exact supporting quote" in (search_tool.description or "")

            success = await session.call_tool("search_mls_rfc", {"query": "external commits"})
            assert success.isError is False
            success_payload = success.structuredContent
            assert success_payload["run"]["id"] == 7
            assert success_payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"
            assert success_payload["results"][0]["citations"][0]["section"] == "1 | Introduction"
            assert success_payload["results"][0]["citations"][0]["quote"] == "Chunk body"

            empty = await session.call_tool("search_mls_rfc", {"query": "empty"})
            assert empty.isError is False
            assert empty.structuredContent["results"] == []

            thresholded = await session.call_tool("search_mls_rfc", {"query": "thresholded"})
            assert thresholded.isError is False
            assert thresholded.structuredContent["results"][0]["chunk_id"] == "fixed:1-introduction:0"
            assert thresholded.structuredContent["results"][0]["citations"][0]["quote"] == "Chunk body"

            failing = await session.call_tool("search_mls_rfc", {"query": "boom"})
            assert failing.isError is True

    anyio.run(scenario)

    log_lines = [line for line in log_stream.getvalue().splitlines() if line.strip()]
    assert len(log_lines) == 4

    success_log = next(
        line for line in log_lines if "status=success" in line and "query='external commits'" in line
    )
    assert "rag_search" in success_log
    assert "requested_top_k=-" in success_log
    assert "effective_top_k=3" in success_log
    assert "similarity_score_threshold=0.9" in success_log
    assert "run_id=7" in success_log
    assert "run_name='active-run'" in success_log
    assert "result_count=1" in success_log
    assert "results=#1 score=0.9100" in success_log
    assert "chunk_id='fixed:1-introduction:0'" in success_log
    assert "section='1 | Introduction'" in success_log
    assert "source='rfc9420.txt'" in success_log
    assert "error=-" in success_log
    assert "error_type=-" in success_log
    assert "elapsed_ms=" in success_log

    empty_log = next(line for line in log_lines if "query='empty'" in line)
    assert "status=success" in empty_log
    assert "result_count=0" in empty_log
    assert "results=none" in empty_log

    failing_log = next(line for line in log_lines if "query='boom'" in line)
    assert "status=error" in failing_log
    assert "requested_top_k=-" in failing_log
    assert "effective_top_k=3" in failing_log
    assert "similarity_score_threshold=0.9" in failing_log
    assert "run_id=7" in failing_log
    assert "run_name='active-run'" in failing_log
    assert "result_count=0" in failing_log
    assert "results=none" in failing_log
    assert "error='Embedding failed'" in failing_log
    assert "error_type='_SearchStateError'" in failing_log


@asynccontextmanager
async def _test_client(server: FastMCP):
    app = Starlette(routes=[Mount("/", app=server.streamable_http_app())])
    transport = ASGITransport(app=app)
    async with server.session_manager.run():
        async with transport:
            async with streamable_http_client("http://127.0.0.1/mcp", http_client=transport.client) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session


class ASGITransport:
    def __init__(self, app: Starlette) -> None:
        import httpx

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()


def _settings():
    from rfc_rag.config import Settings

    return Settings(
        database_url="postgresql://example",
        ollama_host="http://127.0.0.1:11434",
        ollama_embed_model="fake-embedding-model",
    )
