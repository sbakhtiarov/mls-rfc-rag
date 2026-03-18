# RFC 9420 RAG CLI

This project ingests [`rfc9420.txt`](./rfc9420.txt) into Postgres with `pgvector`, using either fixed-size chunks or section-aware chunks, and lets you run similarity queries against an ingestion run.

## Prerequisites

- Python 3.14+
- A Postgres database with the `pgvector` extension available
- `DATABASE_URL`
- `OPENAI_API_KEY`
- Optional: `OPENAI_EMBED_MODEL` (defaults to `text-embedding-3-small`)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Usage

Initialize the schema:

```bash
rfc-rag init-db
```

Ingest the RFC using fixed-size chunks:

```bash
rfc-rag ingest --source rfc9420.txt --strategy fixed --chunk-size 1200
```

Ingest the RFC using section-aware chunks:

```bash
rfc-rag ingest --source rfc9420.txt --strategy section --chunk-size 1800
```

List runs:

```bash
rfc-rag list-runs
```

Mark one run as active:

```bash
rfc-rag set-active-run --run-id 1
```

Query a run:

```bash
rfc-rag query --run-id 1 --query "How does MLS handle external commits?"
```

Query the active run as JSON:

```bash
rfc-rag query --query "How does MLS handle external commits?" --json
```

Serve the active run as an MCP server over local HTTP:

```bash
rfc-rag serve-mcp --host 127.0.0.1 --port 8000
```

The MCP endpoint is available at `http://127.0.0.1:8000/mcp`, and it exposes a single tool that searches the active run only.

Setup flow:

```bash
rfc-rag init-db
rfc-rag ingest --source rfc9420.txt --strategy fixed --chunk-size 1200
rfc-rag set-active-run --run-id 1
rfc-rag serve-mcp --host 127.0.0.1 --port 8000
```

Example MCP client registration:

```bash
claude mcp add --transport http rfc-rag http://127.0.0.1:8000/mcp
```

## Docker Setup

The repo includes [`docker-compose.yml`](./docker-compose.yml) for a two-container local setup:

- `postgres`: Postgres + `pgvector`
- `mcp`: the Python app serving the MCP search endpoint

Build and start both services with:

```bash
docker compose up -d
```

By default, the MCP endpoint is exposed at:

```bash
http://127.0.0.1:8000/mcp
```

Required environment:

```bash
export OPENAI_API_KEY="..."
```

You can override these Docker-specific settings before startup:

- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_PORT`
- `MCP_PORT`
- `OPENAI_EMBED_MODEL`

The MCP container already receives its internal `DATABASE_URL` automatically through Compose, so you do not need to set that manually for the containerized setup.

Typical Docker workflow:

```bash
export OPENAI_API_KEY="..."
docker compose up -d --build
docker compose exec mcp rfc-rag ingest --source /app/rfc9420.txt --strategy fixed --chunk-size 1200
docker compose exec mcp rfc-rag set-active-run --run-id 1
```

You can then connect your MCP client to `http://127.0.0.1:8000/mcp`.

Useful Docker commands:

```bash
docker compose logs -f mcp
docker compose logs -f postgres
docker compose exec mcp rfc-rag list-runs
docker compose down
```

## Test

```bash
pytest
```

Integration tests are skipped unless `TEST_DATABASE_URL` or `DATABASE_URL` points to a disposable Postgres instance with `pgvector`.
