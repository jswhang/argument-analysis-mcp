"""Tests for bulk ingestion tools: ingest_directory, ingest_file_list, ingest_url_list."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aa_mcp.store.document_store import DocumentStore


# ── DocumentStore.get_document_by_source ─────────────────────────────────────

@pytest.fixture
async def doc_store(tmp_db):
    store = DocumentStore(tmp_db)
    await store.initialize()
    return store


async def test_get_document_by_source_not_found(doc_store):
    result = await doc_store.get_document_by_source("/nonexistent/path.txt", "default")
    assert result is None


async def test_get_document_by_source_found(doc_store):
    from aa_mcp.models.rag import Document

    doc = Document(
        id=str(uuid.uuid4()),
        title="Test",
        source="/some/path.txt",
        collection="default",
        full_text="content",
        created_at=datetime.now(timezone.utc),
        chunk_count=1,
    )
    await doc_store.save_document(doc)
    found = await doc_store.get_document_by_source("/some/path.txt", "default")
    assert found == doc.id


async def test_get_document_by_source_different_collection(doc_store):
    """Same source in a different collection should not match."""
    from aa_mcp.models.rag import Document

    doc = Document(
        id=str(uuid.uuid4()),
        title="Test",
        source="/some/path.txt",
        collection="climate",
        full_text="content",
        created_at=datetime.now(timezone.utc),
        chunk_count=1,
    )
    await doc_store.save_document(doc)
    assert await doc_store.get_document_by_source("/some/path.txt", "economics") is None
    assert await doc_store.get_document_by_source("/some/path.txt", "climate") == doc.id


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ctx():
    """Return a mock MCP Context whose lifespan_context is a partially-mocked engine."""
    from aa_mcp.engine import ArgumentAnalysisEngine
    from aa_mcp.config import ServerConfig
    from aa_mcp.models.rag import Chunk

    fake_chunk = Chunk(
        id=str(uuid.uuid4()), doc_id="x", collection="default",
        text="chunk text", index=0, start_char=0, end_char=10,
    )

    engine = ArgumentAnalysisEngine(ServerConfig())
    engine.chunker = MagicMock()
    engine.chunker.chunk = MagicMock(return_value=[fake_chunk])
    engine.embedder = MagicMock()
    engine.embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    engine.backend = MagicMock()
    engine.backend.add_embeddings = AsyncMock()
    engine.doc_store = MagicMock()
    engine.doc_store.get_document_by_source = AsyncMock(return_value=None)
    engine.doc_store.save_document = AsyncMock()
    engine.doc_store.save_chunks = AsyncMock()

    ctx = MagicMock()
    ctx.request_context.lifespan_context = engine
    ctx.report_progress = AsyncMock()
    # No MCP sampling — keeps _llm(ctx) returning None → no summary embeddings
    ctx.session.client_params.capabilities.sampling = None
    return ctx


# ── ingest_file_list ──────────────────────────────────────────────────────────

async def test_ingest_file_list_empty():
    from aa_mcp.tools.rag_ingest import ingest_file_list
    ctx = _make_ctx()
    result = await ingest_file_list(paths=[], collection="default", ctx=ctx)
    assert result["ingested"] == 0
    assert result["skipped"] == 0
    assert "Empty path list" in result["message"]


async def test_ingest_file_list_missing_file():
    from aa_mcp.tools.rag_ingest import ingest_file_list
    ctx = _make_ctx()
    result = await ingest_file_list(paths=["/nonexistent/file.txt"], collection="default", ctx=ctx)
    assert result["ingested"] == 0
    assert len(result["errors"]) == 1
    assert result["errors"][0]["error"] == "file_not_found"


async def test_ingest_file_list_skips_existing(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_file_list
    f = tmp_path / "doc.txt"
    f.write_text("some content")

    ctx = _make_ctx()
    ctx.request_context.lifespan_context.doc_store.get_document_by_source = AsyncMock(
        return_value="existing-doc-id"
    )

    result = await ingest_file_list(paths=[str(f)], collection="default", skip_existing=True, ctx=ctx)
    assert result["skipped"] == 1
    assert result["ingested"] == 0
    assert result["results"][0]["status"] == "skipped"


async def test_ingest_file_list_ingests_text_file(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_file_list
    f = tmp_path / "paper.txt"
    f.write_text("This is a research paper about climate change.")

    ctx = _make_ctx()
    result = await ingest_file_list(paths=[str(f)], collection="default", skip_existing=False, ctx=ctx)
    assert result["ingested"] == 1
    assert result["errors"] == []
    assert result["results"][0]["status"] == "ingested"


async def test_ingest_file_list_continues_after_error(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_file_list
    good = tmp_path / "good.txt"
    good.write_text("Good content here.")

    ctx = _make_ctx()
    paths = ["/does/not/exist.txt", str(good)]
    result = await ingest_file_list(paths=paths, collection="default", skip_existing=False, ctx=ctx)
    assert result["total"] == 2
    assert result["ingested"] == 1
    assert len(result["errors"]) == 1


# ── ingest_directory ──────────────────────────────────────────────────────────

async def test_ingest_directory_not_found():
    from aa_mcp.tools.rag_ingest import ingest_directory
    ctx = _make_ctx()
    result = await ingest_directory("/nonexistent/dir", ctx=ctx)
    assert result["error"] == "directory_not_found"


async def test_ingest_directory_not_a_directory(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_directory
    f = tmp_path / "file.txt"
    f.write_text("content")
    ctx = _make_ctx()
    result = await ingest_directory(str(f), ctx=ctx)
    assert result["error"] == "not_a_directory"


async def test_ingest_directory_no_matching_files(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_directory
    (tmp_path / "data.xyz").write_text("content")
    ctx = _make_ctx()
    result = await ingest_directory(str(tmp_path), pattern="*.xyz", ctx=ctx)
    assert result["ingested"] == 0
    assert "No matching files" in result["message"]


async def test_ingest_directory_ingests_matching_files(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_directory
    (tmp_path / "a.txt").write_text("Document A content.")
    (tmp_path / "b.txt").write_text("Document B content.")
    (tmp_path / "ignore.xyz").write_text("Not ingested.")

    ctx = _make_ctx()
    result = await ingest_directory(str(tmp_path), pattern="*.txt", ctx=ctx)
    assert result["ingested"] == 2
    assert result["errors"] == []


async def test_ingest_directory_recursive(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_directory
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "top.md").write_text("Top level.")
    (sub / "nested.md").write_text("Nested level.")

    ctx = _make_ctx()
    result = await ingest_directory(str(tmp_path), pattern="*.md", recursive=True, ctx=ctx)
    assert result["ingested"] == 2


async def test_ingest_directory_non_recursive_ignores_subdirs(tmp_path):
    from aa_mcp.tools.rag_ingest import ingest_directory
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "top.md").write_text("Top level.")
    (sub / "nested.md").write_text("Nested level.")

    ctx = _make_ctx()
    result = await ingest_directory(str(tmp_path), pattern="*.md", recursive=False, ctx=ctx)
    assert result["ingested"] == 1
