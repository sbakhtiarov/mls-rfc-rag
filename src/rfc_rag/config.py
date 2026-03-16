from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_api_key: str | None
    openai_embed_model: str


def load_settings(require_openai: bool) -> Settings:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if require_openai and not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for this command.")

    return Settings(
        database_url=database_url,
        openai_api_key=openai_api_key,
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )
