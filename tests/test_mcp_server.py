from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

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


def test_serve_mcp_requires_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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

        def query_chunks(self, *, run_id: int, query_embedding: list[float], top_k: int):
            if query_embedding == ["empty"]:
                return []
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
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def embed_text(self, text: str):
            if text == "empty":
                return ["empty"]
            if text == "boom":
                raise RuntimeError("Embedding failed")
            return [0.1, 0.2]

    monkeypatch.setattr("rfc_rag.mcp_server.Database", FakeDatabase)
    monkeypatch.setattr("rfc_rag.mcp_server.OpenAIEmbedder", FakeEmbedder)

    server = create_mcp_server(
        settings=_settings(),
        host="127.0.0.1",
        port=8001,
    )

    async def scenario() -> None:
        async with _test_client(server) as session:
            tools = await session.list_tools()
            search_tool = next(tool for tool in tools.tools if tool.name == "search_mls_rfc")
            assert "Messaging Layer Security" in (search_tool.description or "")
            assert "RFC 9420" in (search_tool.description or "")
            assert "Always use this tool" in (search_tool.description or "")

            success = await session.call_tool("search_mls_rfc", {"query": "external commits", "top_k": 5})
            assert success.isError is False
            success_payload = success.structuredContent
            assert success_payload["run"]["id"] == 7
            assert success_payload["results"][0]["chunk_id"] == "fixed:1-introduction:0"

            empty = await session.call_tool("search_mls_rfc", {"query": "empty", "top_k": 5})
            assert empty.isError is False
            assert empty.structuredContent["results"] == []

            invalid = await session.call_tool("search_mls_rfc", {"query": "external commits", "top_k": 25})
            assert invalid.isError is True

            failing = await session.call_tool("search_mls_rfc", {"query": "boom", "top_k": 5})
            assert failing.isError is True

    anyio.run(scenario)


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
        openai_api_key="test-key",
        openai_embed_model="fake-embedding-model",
    )
