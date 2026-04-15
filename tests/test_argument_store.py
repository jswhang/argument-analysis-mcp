"""Tests for ArgumentStore and DocumentStore."""

from __future__ import annotations

import pytest

from aa_mcp.models.argument import ArgumentAssessment, ComponentAssessment, FallacyDetection, ToulminComponents
from aa_mcp.models.mapping import ArgumentEdge, ArgumentMap, ArgumentNode, RelationshipType
from aa_mcp.models.rag import Chunk, Document
from aa_mcp.store.argument_store import ArgumentStore
from aa_mcp.store.document_store import DocumentStore
import uuid
from datetime import datetime, timezone


@pytest.fixture
async def doc_store(tmp_db):
    store = DocumentStore(tmp_db)
    await store.initialize()
    return store


@pytest.fixture
async def arg_store(tmp_db):
    store = ArgumentStore(tmp_db)
    await store.initialize()
    return store


async def test_document_round_trip(doc_store):
    doc = Document(
        id=str(uuid.uuid4()),
        title="Test Article",
        source="/tmp/test.txt",
        collection="default",
        full_text="This is a test document with some content.",
        created_at=datetime.now(timezone.utc),
        chunk_count=1,
    )
    await doc_store.save_document(doc)
    retrieved = await doc_store.get_document(doc.id)
    assert retrieved is not None
    assert retrieved.title == "Test Article"
    assert retrieved.full_text == doc.full_text


async def test_chunk_round_trip(doc_store):
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id, title="Doc", source="x", collection="default",
        full_text="text", created_at=datetime.now(timezone.utc), chunk_count=2
    )
    await doc_store.save_document(doc)

    chunks = [
        Chunk(id=str(uuid.uuid4()), doc_id=doc_id, collection="default", text=f"chunk {i}", index=i, start_char=0, end_char=10)
        for i in range(2)
    ]
    await doc_store.save_chunks(chunks)

    ids = [c.id for c in chunks]
    retrieved = await doc_store.get_chunks_by_ids(ids)
    assert len(retrieved) == 2
    texts = {c.text for c in retrieved}
    assert "chunk 0" in texts
    assert "chunk 1" in texts


async def test_assessment_round_trip(arg_store):
    assessment = ArgumentAssessment(
        id=str(uuid.uuid4()),
        source_text="Climate change is real because scientists say so.",
        toulmin=ToulminComponents(claim="Climate change is real", data=["scientists say so"], warrant="expert consensus"),
        component_assessments=[
            ComponentAssessment(
                component_type="claim",
                text="Climate change is real",
                logical_validity_score=0.9,
                knowledge_base_score=0.95,
                weakness_flag=False,
            )
        ],
        fallacies=[FallacyDetection(fallacy_type="appeal_to_authority", excerpt="scientists say so", explanation="Appeal to authority without evidence", severity="low")],
        overall_validity_score=0.85,
        overall_truthfulness_score=0.90,
        summary="This is a solid argument.",
        rag_collection_used="default",
    )
    await arg_store.save_assessment(assessment)
    retrieved = await arg_store.get_assessment(assessment.id)
    assert retrieved is not None
    assert retrieved.overall_validity_score == pytest.approx(0.85)
    assert len(retrieved.fallacies) == 1
    assert retrieved.fallacies[0].fallacy_type == "appeal_to_authority"


async def test_argument_map_round_trip(arg_store):
    now = datetime.now(timezone.utc)
    map_ = ArgumentMap(id=str(uuid.uuid4()), title="Test Map", created_at=now, updated_at=now)
    await arg_store.create_map(map_)

    node1 = ArgumentNode(id=str(uuid.uuid4()), map_id=map_.id, text="Main claim", node_type="claim")
    node2 = ArgumentNode(id=str(uuid.uuid4()), map_id=map_.id, text="Supporting premise", node_type="premise")
    await arg_store.add_node(node1)
    await arg_store.add_node(node2)

    edge = ArgumentEdge(
        id=str(uuid.uuid4()), map_id=map_.id,
        source_id=node2.id, target_id=node1.id,
        relationship=RelationshipType.SUPPORTS, explanation="Directly supports"
    )
    await arg_store.add_edge(edge)

    retrieved = await arg_store.get_map(map_.id)
    assert retrieved is not None
    assert len(retrieved.nodes) == 2
    assert len(retrieved.edges) == 1
    assert retrieved.edges[0].relationship == RelationshipType.SUPPORTS
