"""Data models for argument maps (graph of argument nodes and edges)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    QUALIFIES = "qualifies"
    ELABORATES = "elaborates"


class ArgumentNode(BaseModel):
    id: str
    map_id: str
    text: str
    node_type: Literal["claim", "premise", "conclusion", "evidence", "rebuttal"]
    assessment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArgumentEdge(BaseModel):
    id: str
    map_id: str
    source_id: str
    target_id: str
    relationship: RelationshipType
    explanation: str | None = None


class ArgumentMap(BaseModel):
    id: str
    title: str
    description: str | None = None
    nodes: list[ArgumentNode] = Field(default_factory=list)
    edges: list[ArgumentEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
