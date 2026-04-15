"""MCP tool wrappers for argument mapping — thin adapters over ArgumentAnalysisEngine."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from aa_mcp.analysis.llm_caller import has_mcp_sampling, mcp_session_caller
from aa_mcp.engine import ArgumentAnalysisEngine
from aa_mcp.models.mapping import RelationshipType


def _engine(ctx: Context) -> ArgumentAnalysisEngine:
    return ctx.request_context.lifespan_context


def _llm(ctx: Context):
    if has_mcp_sampling(ctx.session):
        return mcp_session_caller(ctx.session)
    return None


async def create_argument_map(
    title: str,
    description: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a new argument map. Returns the map ID."""
    map_ = await _engine(ctx).create_argument_map(title, description)
    return {"id": map_.id, "title": map_.title}


async def add_argument_node(
    map_id: str,
    text: str,
    node_type: str = "claim",
    ctx: Context = None,
) -> dict[str, Any]:
    """Add an argument node to a map.
    node_type: claim | premise | conclusion | evidence | rebuttal
    """
    valid_types = {"claim", "premise", "conclusion", "evidence", "rebuttal"}
    if node_type not in valid_types:
        return {"error": f"node_type must be one of {sorted(valid_types)}"}

    node = await _engine(ctx).add_argument_node(map_id, text, node_type)
    return {"id": node.id, "map_id": map_id, "node_type": node_type, "text": text}


async def link_arguments(
    map_id: str,
    source_id: str,
    target_id: str,
    relationship: str,
    explanation: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a directed edge between two argument nodes.
    relationship: supports | attacks | qualifies | elaborates
    """
    valid_rels = {r.value for r in RelationshipType}
    if relationship not in valid_rels:
        return {"error": f"relationship must be one of {sorted(valid_rels)}"}

    edge = await _engine(ctx).link_arguments(map_id, source_id, target_id, relationship, explanation)
    return {"id": edge.id, "source_id": source_id, "target_id": target_id, "relationship": relationship}


async def get_argument_map(
    map_id: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Retrieve a full argument map with all nodes and edges."""
    map_ = await _engine(ctx).get_argument_map(map_id)
    if not map_:
        return {"error": "not_found", "map_id": map_id}
    return map_.model_dump(mode="json")


async def auto_map_argument(
    text: str,
    map_id: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Full pipeline: decompose argument → create nodes → infer relationships.
    Creates a new map if map_id is not provided. Requires sampling capability.
    """
    llm = _llm(ctx)
    if llm is None:
        return {"error": "sampling_not_supported", "message": "This tool requires sampling/createMessage."}

    async def _on_progress(pct, msg):
        await ctx.report_progress(pct, 100, msg)

    return await _engine(ctx).auto_map_argument(text, map_id=map_id, llm=llm, on_progress=_on_progress)
