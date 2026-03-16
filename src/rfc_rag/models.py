from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Section:
    label: str
    slug: str
    content: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    section: str
    chunk_index: int
    content: str
    char_count: int


@dataclass(frozen=True)
class IngestionRun:
    id: int
    name: str
    source: str
    strategy: str
    chunk_size: int
    embedding_model: str
    created_at: datetime


@dataclass(frozen=True)
class QueryResult:
    chunk_id: str
    source: str
    section: str
    content: str
    score: float
