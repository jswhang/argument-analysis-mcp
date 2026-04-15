"""SQLite store for argument analyses, assessments, and argument maps."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from aa_mcp.models.argument import ArgumentAssessment, ComponentAssessment, FallacyDetection, ToulminComponents
from aa_mcp.models.mapping import ArgumentEdge, ArgumentMap, ArgumentNode, RelationshipType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id                          TEXT PRIMARY KEY,
    source_text                 TEXT NOT NULL,
    toulmin_json                TEXT NOT NULL,
    component_assessments_json  TEXT NOT NULL DEFAULT '[]',
    fallacies_json              TEXT NOT NULL DEFAULT '[]',
    overall_validity_score      REAL NOT NULL DEFAULT 0.0,
    overall_truthfulness_score  REAL,
    summary                     TEXT NOT NULL DEFAULT '',
    rag_collection_used         TEXT,
    metadata_json               TEXT NOT NULL DEFAULT '{}',
    created_at                  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS argument_maps (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS argument_nodes (
    id            TEXT PRIMARY KEY,
    map_id        TEXT NOT NULL REFERENCES argument_maps(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    node_type     TEXT NOT NULL,
    assessment_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS argument_edges (
    id           TEXT PRIMARY KEY,
    map_id       TEXT NOT NULL REFERENCES argument_maps(id) ON DELETE CASCADE,
    source_id    TEXT NOT NULL REFERENCES argument_nodes(id) ON DELETE CASCADE,
    target_id    TEXT NOT NULL REFERENCES argument_nodes(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    explanation  TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_map_id ON argument_nodes(map_id);
CREATE INDEX IF NOT EXISTS idx_edges_map_id ON argument_edges(map_id);
"""


class ArgumentStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    # ── Assessments ────────────────────────────────────────────────────────

    async def save_assessment(self, assessment: ArgumentAssessment) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO assessments
                    (id, source_text, toulmin_json, component_assessments_json, fallacies_json,
                     overall_validity_score, overall_truthfulness_score, summary,
                     rag_collection_used, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment.id,
                    assessment.source_text,
                    assessment.toulmin.model_dump_json(),
                    json.dumps([ca.model_dump() for ca in assessment.component_assessments]),
                    json.dumps([f.model_dump() for f in assessment.fallacies]),
                    assessment.overall_validity_score,
                    assessment.overall_truthfulness_score,
                    assessment.summary,
                    assessment.rag_collection_used,
                    json.dumps(assessment.metadata),
                    assessment.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_assessment(self, assessment_id: str) -> ArgumentAssessment | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM assessments WHERE id = ?", (assessment_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return _row_to_assessment(row) if row else None

    # ── Argument Maps ───────────────────────────────────────────────────────

    async def create_map(self, map_: ArgumentMap) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO argument_maps (id, title, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (map_.id, map_.title, map_.description, map_.created_at.isoformat(), map_.updated_at.isoformat()),
            )
            await db.commit()

    async def get_map(self, map_id: str) -> ArgumentMap | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM argument_maps WHERE id = ?", (map_id,)) as cursor:
                map_row = await cursor.fetchone()
            if not map_row:
                return None
            async with db.execute(
                "SELECT * FROM argument_nodes WHERE map_id = ?", (map_id,)
            ) as cursor:
                node_rows = await cursor.fetchall()
            async with db.execute(
                "SELECT * FROM argument_edges WHERE map_id = ?", (map_id,)
            ) as cursor:
                edge_rows = await cursor.fetchall()

        return ArgumentMap(
            id=map_row["id"],
            title=map_row["title"],
            description=map_row["description"],
            nodes=[_row_to_node(r) for r in node_rows],
            edges=[_row_to_edge(r) for r in edge_rows],
        )

    async def list_maps(self) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, title, description, created_at FROM argument_maps ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_node(self, node: ArgumentNode) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO argument_nodes (id, map_id, text, node_type, assessment_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (node.id, node.map_id, node.text, node.node_type, node.assessment_id, json.dumps(node.metadata)),
            )
            await db.execute(
                "UPDATE argument_maps SET updated_at = datetime('now') WHERE id = ?", (node.map_id,)
            )
            await db.commit()

    async def add_edge(self, edge: ArgumentEdge) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO argument_edges (id, map_id, source_id, target_id, relationship, explanation) VALUES (?, ?, ?, ?, ?, ?)",
                (edge.id, edge.map_id, edge.source_id, edge.target_id, edge.relationship.value, edge.explanation),
            )
            await db.execute(
                "UPDATE argument_maps SET updated_at = datetime('now') WHERE id = ?", (edge.map_id,)
            )
            await db.commit()

    async def delete_map(self, map_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id FROM argument_maps WHERE id = ?", (map_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    return False
            await db.execute("DELETE FROM argument_maps WHERE id = ?", (map_id,))
            await db.commit()
        return True


def _row_to_assessment(row: aiosqlite.Row) -> ArgumentAssessment:
    from datetime import datetime

    components_data = json.loads(row["component_assessments_json"])
    components = [ComponentAssessment(**ca) for ca in components_data]
    fallacies = [FallacyDetection(**f) for f in json.loads(row["fallacies_json"])]

    return ArgumentAssessment(
        id=row["id"],
        source_text=row["source_text"],
        toulmin=ToulminComponents.model_validate_json(row["toulmin_json"]),
        component_assessments=components,
        fallacies=fallacies,
        overall_validity_score=row["overall_validity_score"],
        overall_truthfulness_score=row["overall_truthfulness_score"],
        critical_weaknesses=[c for c in components if c.weakness_flag],
        summary=row["summary"],
        rag_collection_used=row["rag_collection_used"],
        metadata=json.loads(row["metadata_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_node(row: aiosqlite.Row) -> ArgumentNode:
    return ArgumentNode(
        id=row["id"],
        map_id=row["map_id"],
        text=row["text"],
        node_type=row["node_type"],
        assessment_id=row["assessment_id"],
        metadata=json.loads(row["metadata_json"]),
    )


def _row_to_edge(row: aiosqlite.Row) -> ArgumentEdge:
    return ArgumentEdge(
        id=row["id"],
        map_id=row["map_id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        relationship=RelationshipType(row["relationship"]),
        explanation=row["explanation"],
    )
