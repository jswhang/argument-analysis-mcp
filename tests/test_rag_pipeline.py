"""Tests for RAGPipeline.search()."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aa_mcp.config import ServerConfig
from aa_mcp.models.rag import Chunk, SearchResult
from aa_mcp.rag.pipeline import RAGPipeline


def _make_chunk(text: str, doc_id: str | None = None) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        doc_id=doc_id or str(uuid.uuid4()),
        collection="default",
        text=text,
        index=0,
        start_char=0,
        end_char=len(text),
    )


def _make_pipeline(
    backend_results: list[tuple[str, float]],
    chunks: list[Chunk],
) -> RAGPipeline:
    """Build a RAGPipeline with mocked embedder, backend, and document store."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 384)

    backend = MagicMock()
    backend.search = AsyncMock(return_value=backend_results)

    doc_store = MagicMock()
    doc_store.get_chunks_by_ids = AsyncMock(return_value=chunks)

    config = ServerConfig()
    return RAGPipeline(backend=backend, embedder=embedder, doc_store=doc_store, config=config)


# ── Basic search ─────────────────────────────────────────────────────────────

async def test_search_returns_results_sorted_by_score():
    chunk_a = _make_chunk("Chunk about climate")
    chunk_b = _make_chunk("Chunk about economics")

    backend_results = [(chunk_b.id, 0.9), (chunk_a.id, 0.6)]
    pipeline = _make_pipeline(backend_results, [chunk_a, chunk_b])

    results = await pipeline.search("climate", "default")
    assert len(results) == 2
    assert results[0].score == pytest.approx(0.9)
    assert results[0].chunk.id == chunk_b.id
    assert results[1].score == pytest.approx(0.6)


async def test_search_assigns_rank():
    chunk_a = _make_chunk("First")
    chunk_b = _make_chunk("Second")
    chunk_c = _make_chunk("Third")

    ids_scores = [(chunk_a.id, 0.95), (chunk_b.id, 0.80), (chunk_c.id, 0.60)]
    pipeline = _make_pipeline(ids_scores, [chunk_a, chunk_b, chunk_c])

    results = await pipeline.search("query", "default")
    for i, r in enumerate(results):
        assert r.rank == i + 1


async def test_search_empty_when_backend_returns_nothing():
    pipeline = _make_pipeline([], [])
    results = await pipeline.search("query", "default")
    assert results == []


async def test_search_caps_at_top_k():
    chunks = [_make_chunk(f"Chunk {i}") for i in range(8)]
    backend_results = [(c.id, 1.0 - i * 0.1) for i, c in enumerate(chunks)]
    pipeline = _make_pipeline(backend_results, chunks)

    # top_k_final default is 5
    results = await pipeline.search("query", "default")
    assert len(results) <= 5


async def test_search_explicit_top_k():
    chunks = [_make_chunk(f"Chunk {i}") for i in range(8)]
    backend_results = [(c.id, 1.0 - i * 0.1) for i, c in enumerate(chunks)]
    pipeline = _make_pipeline(backend_results, chunks)

    results = await pipeline.search("query", "default", top_k=3)
    assert len(results) == 3


async def test_search_calls_embed_once():
    chunk = _make_chunk("Some text")
    pipeline = _make_pipeline([(chunk.id, 0.8)], [chunk])

    await pipeline.search("query", "default")
    pipeline._embedder.embed.assert_awaited_once_with("query")


async def test_search_passes_collection_to_backend():
    chunk = _make_chunk("text")
    pipeline = _make_pipeline([(chunk.id, 0.7)], [chunk])

    await pipeline.search("q", "my-collection")
    call_args = pipeline._backend.search.call_args
    assert call_args[0][1] == "my-collection" or call_args[1].get("collection") == "my-collection"


async def test_search_result_type():
    chunk = _make_chunk("Evidence text")
    pipeline = _make_pipeline([(chunk.id, 0.75)], [chunk])

    results = await pipeline.search("query", "default")
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.text == "Evidence text"
    assert results[0].score == pytest.approx(0.75)
