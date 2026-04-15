"""MCP tool wrappers for argument analysis — thin adapters over ArgumentAnalysisEngine."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from aa_mcp.analysis.llm_caller import has_mcp_sampling, mcp_session_caller
from aa_mcp.engine import ArgumentAnalysisEngine


def _engine(ctx: Context) -> ArgumentAnalysisEngine:
    return ctx.request_context.lifespan_context


def _llm(ctx: Context):
    """Return an LLMCaller from the MCP session, or None if unsupported."""
    if has_mcp_sampling(ctx.session):
        return mcp_session_caller(ctx.session)
    return None


async def _on_progress(ctx: Context, pct: int, msg: str) -> None:
    await ctx.report_progress(pct, 100, msg)


async def assess_argument(
    text: str,
    collection: str = "default",
    store_result: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Analyze an argument for logical quality, reasoning errors, and fallacies.
    Returns an overall validity score, detected weaknesses, and a prose summary.

    If reference documents have been added to the knowledge base (via ingest_file,
    ingest_url, or ingest_text), claims will also be cross-checked against those
    sources. The logical analysis runs regardless of whether a knowledge base exists.

    Requires sampling capability (the MCP client must support sampling/createMessage).
    """
    llm = _llm(ctx)
    if llm is None:
        return {
            "error": "sampling_not_supported",
            "message": "This tool requires an MCP client that supports sampling/createMessage (e.g. Claude Desktop).",
        }

    engine = _engine(ctx)

    async def _progress(pct, msg):
        await _on_progress(ctx, pct, msg)

    result = await engine.assess_argument(text, collection, llm=llm, store_result=store_result, on_progress=_progress)
    return _format_assessment(result)


async def analyze_argument_structure(
    text: str,
    store_result: bool = False,
    ctx: Context = None,
) -> dict[str, Any]:
    """Break down an argument into its structural components: the main claim,
    supporting evidence, the underlying reasoning connecting evidence to claim,
    any stated limitations, and anticipated counterarguments.

    No prior knowledge of argument analysis frameworks required.
    Requires sampling capability.
    """
    llm = _llm(ctx)
    if llm is None:
        return {"error": "sampling_not_supported", "message": "This tool requires sampling/createMessage."}

    toulmin = await _engine(ctx).decompose_argument(text, llm)
    return toulmin.model_dump(by_alias=True)


async def detect_fallacies(
    text: str,
    ctx: Context = None,
) -> list[dict[str, Any]]:
    """Identify logical fallacies. Returns list of {fallacy_type, excerpt, explanation, severity}.
    Requires sampling capability.
    """
    llm = _llm(ctx)
    if llm is None:
        return [{"error": "sampling_not_supported", "message": "This tool requires sampling/createMessage."}]

    fallacies = await _engine(ctx).detect_fallacies(text, llm)
    return [f.model_dump() for f in fallacies]


async def get_assessment(
    assessment_id: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Retrieve a previously stored ArgumentAssessment by ID."""
    assessment = await _engine(ctx).get_assessment(assessment_id)
    if not assessment:
        return {"error": "not_found", "id": assessment_id}
    return _format_assessment(assessment)


def _format_assessment(a) -> dict[str, Any]:
    result = {
        "id": a.id,
        "overall_validity_score": round(a.overall_validity_score, 3),
        "summary": a.summary,
        "argument_structure": a.toulmin.model_dump(by_alias=True),
        "fallacies": [f.model_dump() for f in a.fallacies],
        "component_assessments": [ca.model_dump() for ca in a.component_assessments],
        "created_at": a.created_at.isoformat(),
    }
    if a.overall_truthfulness_score is not None:
        result["overall_knowledge_base_score"] = round(a.overall_truthfulness_score, 3)
    if a.rag_collection_used:
        result["rag_collection_used"] = a.rag_collection_used
    if a.critical_weaknesses:
        result["critical_weaknesses"] = [
            {
                "component": ca.component_type,
                "text": ca.text[:200],
                "weakness": ca.weakness_explanation,
                "validity_score": round(ca.logical_validity_score, 3),
                **({"knowledge_base_score": round(ca.knowledge_base_score, 3)} if ca.knowledge_base_score is not None else {}),
                "contradicted_by": ca.contradicting_evidence,
            }
            for ca in a.critical_weaknesses
        ]
    return result
