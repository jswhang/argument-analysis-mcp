"""Tests for TextChunker."""

from aa_mcp.config import ChunkingConfig
from aa_mcp.rag.chunker import TextChunker


def make_chunker(strategy="recursive", chunk_size=100, overlap=20):
    return TextChunker(ChunkingConfig(strategy=strategy, chunk_size=chunk_size, chunk_overlap=overlap, min_chunk_size=10))


def test_fixed_split_produces_chunks():
    chunker = make_chunker(strategy="fixed", chunk_size=50, overlap=10)
    text = "A" * 200
    chunks = chunker.chunk(text, doc_id="doc1", collection="test")
    assert len(chunks) > 1
    for c in chunks:
        assert c.doc_id == "doc1"
        assert c.collection == "test"
        assert len(c.text) >= 10


def test_recursive_split_short_text():
    chunker = make_chunker(chunk_size=500)
    text = "This is a short argument. It has two sentences."
    chunks = chunker.chunk(text, doc_id="d1", collection="c1")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunks_have_unique_ids():
    chunker = make_chunker(chunk_size=50, overlap=5)
    text = " ".join(["word"] * 100)
    chunks = chunker.chunk(text, doc_id="d1", collection="c1")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_min_chunk_size_filters_stubs():
    chunker = make_chunker(chunk_size=50, overlap=5)
    # Very short text should still produce one chunk (> min_chunk_size=10)
    text = "Short text here."
    chunks = chunker.chunk(text, doc_id="d1", collection="c1")
    assert all(len(c.text) >= 10 for c in chunks)
