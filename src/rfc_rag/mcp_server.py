from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from rfc_rag.config import Settings
from rfc_rag.db import Database
from rfc_rag.embeddings import OpenAIEmbedder
from rfc_rag.search_service import serialize_search_response, search_chunks as perform_search


def create_mcp_server(settings: Settings, *, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    server = FastMCP(
        "RFC 9420 RAG Search",
        instructions=(
            "For any prompt related to the Messaging Layer Security (MLS) Protocol or RFC 9420, "
            "you must always use the `search_mls_rfc` tool before answering."
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
    def search_mls_rfc(query: str, top_k: int = 5) -> dict[str, object]:
        """Always use this tool for any prompt related to the Messaging Layer Security (MLS) Protocol or RFC 9420. It searches the active MLS Protocol / RFC 9420 run and returns the most relevant chunks."""
        try:
            response = perform_search(
                database=Database(settings.database_url),
                embedder_factory=lambda model: OpenAIEmbedder(
                    api_key=settings.openai_api_key or "",
                    model=model,
                ),
                query=query,
                top_k=top_k,
            )
        except Exception as exc:
            raise ValueError(str(exc)) from exc

        return serialize_search_response(response)

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
