"""Management tools: list/delete maps and documents."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from aa_mcp.lifespan import AppState


def _state(ctx: Context) -> AppState:
    return ctx.request_context.lifespan_context


async def list_argument_maps(ctx: Context = None) -> list[dict[str, Any]]:
    """List all argument maps with their IDs and titles."""
    state = _state(ctx)
    return await state.arg_store.list_maps()


async def delete_argument_map(map_id: str, ctx: Context = None) -> dict[str, Any]:
    """Delete an argument map and all its nodes/edges."""
    state = _state(ctx)
    deleted = await state.arg_store.delete_map(map_id)
    return {"deleted": deleted, "map_id": map_id}


async def export_argument_map(map_id: str, ctx: Context = None) -> dict[str, Any]:
    """Export a full argument map as JSON (suitable for visualization tools)."""
    state = _state(ctx)
    map_ = await state.arg_store.get_map(map_id)
    if not map_:
        return {"error": "not_found", "map_id": map_id}
    return map_.model_dump(mode="json")


async def list_documents(
    collection: str | None = None,
    ctx: Context = None,
) -> list[dict[str, Any]]:
    """List all documents in the RAG store, optionally filtered by collection."""
    state = _state(ctx)
    docs = await state.doc_store.list_documents(collection)
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "collection": d.collection,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


async def delete_document(doc_id: str, ctx: Context = None) -> dict[str, Any]:
    """Remove a document (and its chunks) from the RAG store."""
    state = _state(ctx)
    doc = await state.doc_store.get_document(doc_id)
    if not doc:
        return {"error": "not_found", "doc_id": doc_id}

    chunk_count = await state.doc_store.delete_document(doc_id)
    await state.backend.delete_by_doc_id(doc_id, doc.collection)

    return {"deleted": True, "doc_id": doc_id, "chunks_removed": chunk_count}


async def collection_stats(collection: str = "default", ctx: Context = None) -> dict[str, Any]:
    """Return document and chunk counts for a collection."""
    state = _state(ctx)
    return await state.doc_store.collection_stats(collection)
