"""Tests for ArgumentAnalysisEngine used as a standalone library (no MCP context)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aa_mcp.config import ServerConfig
from aa_mcp.engine import ArgumentAnalysisEngine
from aa_mcp.models.rag import Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_engine() -> ArgumentAnalysisEngine:
    """Engine with mocked subsystems — no real disk/network IO."""
    fake_chunk = Chunk(
        id=str(uuid.uuid4()), doc_id="d1", collection="default",
        text="The greenhouse effect traps heat.", index=0, start_char=0, end_char=32,
    )

    engine = ArgumentAnalysisEngine(ServerConfig())
    engine.chunker = MagicMock()
    engine.chunker.chunk = MagicMock(return_value=[fake_chunk])
    engine.embedder = MagicMock()
    engine.embedder.embed = AsyncMock(return_value=[0.1] * 384)
    engine.embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    engine.backend = MagicMock()
    engine.backend.add_embeddings = AsyncMock()
    engine.backend.search = AsyncMock(return_value=[])
    engine.doc_store = MagicMock()
    engine.doc_store.save_document = AsyncMock()
    engine.doc_store.save_chunks = AsyncMock()
    engine.doc_store.get_document_by_source = AsyncMock(return_value=None)
    engine.doc_store.get_chunks_by_ids = AsyncMock(return_value=[fake_chunk])
    engine.arg_store = MagicMock()
    engine.arg_store.save_assessment = AsyncMock()
    engine.arg_store.get_assessment = AsyncMock(return_value=None)
    engine.arg_store.create_map = AsyncMock()
    engine.arg_store.add_node = AsyncMock()
    engine.arg_store.add_edge = AsyncMock()
    engine.arg_store.get_map = AsyncMock(return_value=None)
    engine.arg_store.list_maps = AsyncMock(return_value=[])
    engine.arg_store.delete_map = AsyncMock(return_value=True)

    from aa_mcp.rag.pipeline import RAGPipeline
    engine.pipeline = RAGPipeline(engine.backend, engine.embedder, engine.doc_store, engine.config)

    return engine


def _make_llm(responses: list[str]):
    """LLMCaller that returns successive fixed responses."""
    it = iter(responses)

    async def _caller(prompt, system, max_tokens, temperature):
        return next(it, "{}")

    return _caller


# ── LLMCaller protocol ────────────────────────────────────────────────────────

async def test_any_async_callable_works_as_llm():
    """Any async (prompt, system, max_tokens, temperature) -> str works as an LLMCaller."""
    from aa_mcp.analysis.sampling import SamplingHelper
    from aa_mcp.config import SamplingConfig

    call_log = []

    async def my_llm(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        call_log.append({"prompt": prompt, "system": system})
        return json.dumps({"claim": "X", "data": [], "warrant": "Y"})

    sampler = SamplingHelper(my_llm, SamplingConfig())
    result = await sampler.decompose("X because Y.")
    assert result.claim == "X"
    assert len(call_log) == 1


# ── assess_argument ───────────────────────────────────────────────────────────

async def test_assess_argument_requires_llm():
    engine = _make_engine()
    with pytest.raises(ValueError, match="LLMCaller"):
        await engine.assess_argument("Some argument.", llm=None)


async def test_assess_argument_raises_on_bad_llm_json():
    engine = _make_engine()

    async def bad_llm(prompt, system, max_tokens, temperature):
        return "not json at all"

    with pytest.raises(ValueError, match="unparseable JSON"):
        await engine.assess_argument("X because Y.", llm=bad_llm)


async def test_assess_argument_returns_assessment():
    engine = _make_engine()

    toulmin_resp = json.dumps({
        "claim": "Climate change is real",
        "data": ["CO2 levels are rising"],
        "warrant": "Greenhouse gases trap heat",
    })
    component_resp = json.dumps({
        "logical_validity_score": 0.9,
        "knowledge_base_score": 0.85,
        "supporting_evidence": ["CO2 data supports this"],
        "contradicting_evidence": [],
        "weakness_flag": False,
        "weakness_explanation": None,
    })
    fallacy_resp = "[]"
    synthesis_resp = json.dumps({
        "overall_validity_score": 0.88,
        "overall_truthfulness_score": 0.83,
        "summary": "This is a well-supported argument.",
    })

    # decompose + (1 component: claim) + (1 datum) + warrant + fallacies + synthesis
    llm = _make_llm([toulmin_resp, component_resp, component_resp, component_resp, fallacy_resp, synthesis_resp])

    from aa_mcp.models.argument import ArgumentAssessment
    result = await engine.assess_argument("Climate change is real because CO2 is rising.", llm=llm)

    assert isinstance(result, ArgumentAssessment)
    assert result.overall_validity_score == pytest.approx(0.88)
    assert result.summary == "This is a well-supported argument."
    engine.arg_store.save_assessment.assert_awaited_once()


async def test_assess_argument_store_result_false():
    engine = _make_engine()
    # claim="X", data=[], warrant="Y" → 2 component assessments (claim + warrant)
    toulmin_resp = json.dumps({"claim": "X", "data": [], "warrant": "Y"})
    component_resp = json.dumps({
        "logical_validity_score": 0.8, "knowledge_base_score": 0.8,
        "supporting_evidence": [], "contradicting_evidence": [],
        "weakness_flag": False, "weakness_explanation": None,
    })
    synthesis_resp = json.dumps({
        "overall_validity_score": 0.8, "overall_truthfulness_score": 0.8, "summary": "OK.",
    })
    llm = _make_llm([toulmin_resp, component_resp, component_resp, "[]", synthesis_resp])
    await engine.assess_argument("X because Y.", llm=llm, store_result=False)
    engine.arg_store.save_assessment.assert_not_awaited()


async def test_assess_argument_flags_critical_weaknesses():
    engine = _make_engine()
    toulmin_resp = json.dumps({"claim": "X", "data": ["weak datum"], "warrant": "Y"})
    component_resp_weak = json.dumps({
        "logical_validity_score": 0.3, "knowledge_base_score": 0.2,
        "supporting_evidence": [], "contradicting_evidence": ["contradicts X"],
        "weakness_flag": True, "weakness_explanation": "Directly contradicted by evidence",
    })
    component_resp_ok = json.dumps({
        "logical_validity_score": 0.8, "knowledge_base_score": 0.8,
        "supporting_evidence": [], "contradicting_evidence": [],
        "weakness_flag": False, "weakness_explanation": None,
    })
    synthesis_resp = json.dumps({
        "overall_validity_score": 0.4, "overall_truthfulness_score": 0.3,
        "summary": "Weak argument with contradicted components.",
    })
    llm = _make_llm([toulmin_resp, component_resp_weak, component_resp_weak, component_resp_ok, "[]", synthesis_resp])
    result = await engine.assess_argument("X because weak datum.", llm=llm)
    assert len(result.critical_weaknesses) > 0
    assert result.critical_weaknesses[0].weakness_flag is True


async def test_assess_argument_progress_callback():
    engine = _make_engine()
    toulmin_resp = json.dumps({"claim": "X", "data": [], "warrant": "Y"})
    component_resp = json.dumps({
        "logical_validity_score": 0.8, "knowledge_base_score": 0.8,
        "supporting_evidence": [], "contradicting_evidence": [],
        "weakness_flag": False, "weakness_explanation": None,
    })
    synthesis_resp = json.dumps({
        "overall_validity_score": 0.8, "overall_truthfulness_score": 0.8, "summary": "OK.",
    })
    llm = _make_llm([toulmin_resp, component_resp, component_resp, "[]", synthesis_resp])

    progress_events: list[tuple[int, str]] = []

    async def _on_progress(pct, msg):
        progress_events.append((pct, msg))

    await engine.assess_argument("X.", llm=llm, on_progress=_on_progress)
    assert len(progress_events) > 0
    assert progress_events[-1][0] == 100


# ── decompose_argument ────────────────────────────────────────────────────────

async def test_decompose_argument_standalone():
    engine = _make_engine()
    llm = _make_llm([json.dumps({"claim": "X", "data": ["A"], "warrant": "B"})])
    result = await engine.decompose_argument("X because A.", llm)
    assert result.claim == "X"
    assert result.data == ["A"]


# ── detect_fallacies ──────────────────────────────────────────────────────────

async def test_detect_fallacies_standalone():
    engine = _make_engine()
    payload = [{"fallacy_type": "ad_hominem", "excerpt": "you're wrong",
                "explanation": "Attacks the person.", "severity": "medium"}]
    llm = _make_llm([json.dumps(payload)])
    result = await engine.detect_fallacies("You're wrong so your claim is wrong.", llm)
    assert len(result) == 1
    assert result[0].fallacy_type == "ad_hominem"


# ── ingest_text ───────────────────────────────────────────────────────────────

async def test_ingest_text_standalone():
    engine = _make_engine()
    result = await engine.ingest_text("Research findings...", "My Paper", collection="science")
    assert result["collection"] == "science"
    assert result["chunks"] >= 1
    engine.doc_store.save_document.assert_awaited_once()
    engine.doc_store.save_chunks.assert_awaited_once()


# ── synthesize_answer ─────────────────────────────────────────────────────────

async def test_synthesize_answer_without_llm_returns_formatted_chunks():
    """Graceful degradation: returns formatted chunks when no LLM provided."""
    engine = _make_engine()
    # Make the pipeline return a result
    from aa_mcp.models.rag import SearchResult
    fake_chunk = Chunk(id="c1", doc_id="d1", collection="default",
                       text="CO2 levels have risen by 50% since industrialisation.", index=0,
                       start_char=0, end_char=52)
    engine.backend.search = AsyncMock(return_value=[("c1", 0.9)])
    engine.doc_store.get_chunks_by_ids = AsyncMock(return_value=[fake_chunk])

    result = await engine.synthesize_answer("What is happening to CO2?", llm=None)
    assert "CO2 levels" in result
    assert isinstance(result, str)


async def test_synthesize_answer_with_llm():
    engine = _make_engine()
    fake_chunk = Chunk(id="c1", doc_id="d1", collection="default",
                       text="CO2 levels have risen.", index=0, start_char=0, end_char=20)
    engine.backend.search = AsyncMock(return_value=[("c1", 0.9)])
    engine.doc_store.get_chunks_by_ids = AsyncMock(return_value=[fake_chunk])

    llm = _make_llm(["CO2 levels have risen significantly [1]."])
    result = await engine.synthesize_answer("CO2 trend?", llm=llm)
    assert "CO2" in result


# ── Argument mapping ──────────────────────────────────────────────────────────

async def test_create_and_link_map():
    engine = _make_engine()
    map_ = await engine.create_argument_map("Test Map", "A test")
    assert map_.title == "Test Map"
    engine.arg_store.create_map.assert_awaited_once()

    node = await engine.add_argument_node(map_.id, "Main claim", "claim")
    assert node.node_type == "claim"

    support = await engine.add_argument_node(map_.id, "Evidence", "premise")
    edge = await engine.link_arguments(map_.id, support.id, node.id, "supports", "Direct support")
    assert edge.relationship.value == "supports"
