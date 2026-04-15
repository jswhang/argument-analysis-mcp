"""Data models for RAG documents, chunks, and search results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Full article/document stored in SQLite (source of truth)."""

    id: str
    title: str
    source: str
    collection: str
    full_text: str
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_count: int = 0


class Chunk(BaseModel):
    """Text chunk stored in SQLite. ChromaDB only holds the embedding + chunk_id."""

    id: str
    doc_id: str
    collection: str
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A ranked search result: chunk text (from SQLite) + similarity score (from ChromaDB)."""

    chunk: Chunk
    score: float
    rank: int
    reranked: bool = False
