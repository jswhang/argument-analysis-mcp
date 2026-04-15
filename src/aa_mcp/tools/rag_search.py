"""MCP tool wrappers for RAG search — thin adapters over ArgumentAnalysisEngine."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from aa_mcp.analysis.llm_caller import has_mcp_sampling, mcp_session_caller
from aa_mcp.engine import ArgumentAnalysisEngine


def _engine(ctx: Context) -> ArgumentAnalysisEngine:
    return ctx.request_context.lifespan_context


def _llm(ctx: Context):
    if has_mcp_sampling(ctx.session):
        return mcp_session_caller(ctx.session)
    return None


async def search(
    query: str,
    collection: str = "default",
    top_k: int = 5,
    ctx: Context = None,
) -> list[dict[str, Any]]:
    """Search the RAG knowledge base. Returns ranked chunks with relevance scores."""
    results = await _engine(ctx).search(query, collection, top_k=top_k)
    return [
        {
            "rank": r.rank,
            "score": round(r.score, 4),
            "text": r.chunk.text,
            "doc_id": r.chunk.doc_id,
            "chunk_id": r.chunk.id,
            "metadata": r.chunk.metadata,
        }
        for r in results
    ]


async def get_document(
    doc_id: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Retrieve full article text for a document by its ID."""
    doc = await _engine(ctx).get_document(doc_id)
    if not doc:
        return {"error": "not_found", "doc_id": doc_id}
    return {
        "id": doc.id,
        "title": doc.title,
        "source": doc.source,
        "collection": doc.collection,
        "full_text": doc.full_text,
        "metadata": doc.metadata,
        "chunk_count": doc.chunk_count,
        "created_at": doc.created_at.isoformat(),
    }


async def synthesize_answer(
    query: str,
    collection: str = "default",
    top_k: int = 5,
    ctx: Context = None,
) -> str:
    """Search RAG store and synthesize a cited prose answer via LLM sampling.
    Falls back to returning formatted chunks if sampling is unavailable.
    """
    return await _engine(ctx).synthesize_answer(query, collection, top_k=top_k, llm=_llm(ctx))
