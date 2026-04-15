"""Data models for argument analysis and assessment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToulminComponents(BaseModel):
    """Internal argument structure used by the analysis pipeline."""

    model_config = ConfigDict(populate_by_name=True)

    claim: str
    data: list[str] = Field(default_factory=list, serialization_alias="evidence")
    warrant: str = Field(default="", serialization_alias="reasoning")
    backing: str | None = Field(default=None, serialization_alias="support")
    qualifier: str | None = Field(default=None, serialization_alias="limitations")
    rebuttal: str | None = Field(default=None, serialization_alias="counterarguments")


class FallacyDetection(BaseModel):
    """A detected logical fallacy."""

    fallacy_type: str
    excerpt: str
    explanation: str
    severity: Literal["low", "medium", "high"]


class ComponentAssessment(BaseModel):
    """Assessment of a single argument component."""

    component_type: Literal["claim", "evidence", "reasoning", "support", "limitations", "counterarguments"]
    text: str
    logical_validity_score: float = Field(ge=0.0, le=1.0)
    knowledge_base_score: float | None = None  # None = no KB evidence was available
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    weakness_flag: bool = False
    weakness_explanation: str | None = None


class ArgumentAssessment(BaseModel):
    """Full assessment result for an argument."""

    id: str
    source_text: str
    toulmin: ToulminComponents
    component_assessments: list[ComponentAssessment] = Field(default_factory=list)
    fallacies: list[FallacyDetection] = Field(default_factory=list)
    overall_validity_score: float = Field(ge=0.0, le=1.0, default=0.0)
    overall_truthfulness_score: float | None = Field(default=None)  # None when no KB evidence available
    critical_weaknesses: list[ComponentAssessment] = Field(default_factory=list)
    summary: str = ""
    rag_collection_used: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
