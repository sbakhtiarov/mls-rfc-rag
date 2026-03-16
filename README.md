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

Query a run:

```bash
rfc-rag query --run-id 1 --query "How does MLS handle external commits?"
```

## Docker Postgres

The repo includes [`docker-compose.yml`](./docker-compose.yml) for a local Postgres + `pgvector` setup.

Start it with:

```bash
docker compose up -d
```

Then point the CLI at it:

```bash
export DATABASE_URL="postgresql://raguser:ragpass@localhost:5432/ragdb"
export OPENAI_API_KEY="..."
```

If you want different credentials or a different port, set `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_PORT` before `docker compose up -d`.

## Test

```bash
pytest
```

Integration tests are skipped unless `TEST_DATABASE_URL` or `DATABASE_URL` points to a disposable Postgres instance with `pgvector`.
