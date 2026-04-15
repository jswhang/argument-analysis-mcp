"""Argument Analysis MCP Server — also usable as a standalone library.

Quickstart (library mode)::

    from aa_mcp import ArgumentAnalysisEngine, LLMCaller
    import asyncio

    async def my_llm(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        # plug in any LLM here
        ...

    async def main():
        async with ArgumentAnalysisEngine() as engine:
            await engine.ingest_file("/path/to/paper.pdf", collection="domain")
            assessment = await engine.assess_argument(
                "Your LLM output text here...",
                collection="domain",
                llm=my_llm,
            )
            print(assessment.summary)
            for w in assessment.critical_weaknesses:
                print("WEAK:", w.component_type, "-", w.weakness_explanation)

    asyncio.run(main())
"""

__version__ = "0.1.0"

from aa_mcp.analysis.llm_caller import LLMCaller, mcp_session_caller, has_mcp_sampling
from aa_mcp.config import ServerConfig
from aa_mcp.engine import ArgumentAnalysisEngine, ProgressCallback
from aa_mcp.models.argument import (
    ArgumentAssessment,
    ComponentAssessment,
    FallacyDetection,
    ToulminComponents,
)
from aa_mcp.models.mapping import ArgumentEdge, ArgumentMap, ArgumentNode, RelationshipType
from aa_mcp.models.rag import Chunk, Document, SearchResult

__all__ = [
    # Engine
    "ArgumentAnalysisEngine",
    "ProgressCallback",
    # LLM caller
    "LLMCaller",
    "mcp_session_caller",
    "has_mcp_sampling",
    # Config
    "ServerConfig",
    # Models
    "ArgumentAssessment",
    "ComponentAssessment",
    "FallacyDetection",
    "ToulminComponents",
    "ArgumentEdge",
    "ArgumentMap",
    "ArgumentNode",
    "RelationshipType",
    "Chunk",
    "Document",
    "SearchResult",
]
