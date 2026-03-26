from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768


@dataclass(frozen=True)
class Settings:
    database_url: str
    ollama_host: str
    ollama_embed_model: str


def load_settings() -> Settings:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required.")

    return Settings(
        database_url=database_url,
        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )
