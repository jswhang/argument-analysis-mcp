"""Tests for SamplingHelper.decompose() and _parse_json()."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aa_mcp.analysis.sampling import SamplingHelper, _parse_json
from aa_mcp.config import SamplingConfig
from aa_mcp.models.argument import ToulminComponents


def _make_sampler(response_text: str) -> SamplingHelper:
    """Build a SamplingHelper backed by a fixed-response LLMCaller."""
    async def _caller(prompt, system, max_tokens, temperature):
        return response_text

    return SamplingHelper(_caller, SamplingConfig())


# ── _parse_json ──────────────────────────────────────────────────────────────

def test_parse_json_plain():
    raw = '{"claim": "X", "data": ["Y"]}'
    assert _parse_json(raw) == {"claim": "X", "data": ["Y"]}


def test_parse_json_strips_markdown_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_strips_plain_fences():
    raw = "```\n[1, 2]\n```"
    assert _parse_json(raw) == [1, 2]


def test_parse_json_prefix_text():
    raw = "Here is the JSON:\n{\"key\": \"value\"}"
    assert _parse_json(raw) == {"key": "value"}


def test_parse_json_array():
    raw = '[{"fallacy_type": "ad_hominem"}]'
    result = _parse_json(raw)
    assert isinstance(result, list)
    assert result[0]["fallacy_type"] == "ad_hominem"


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError, match="unparseable JSON"):
        _parse_json("this is not json at all")


# ── SamplingHelper.decompose ─────────────────────────────────────────────────

async def test_decompose_full_components():
    payload = {
        "claim": "Climate change is real",
        "data": ["CO2 is rising", "Temperatures are up"],
        "warrant": "Greenhouse gases trap heat",
        "backing": "IPCC reports",
        "qualifier": "with high confidence",
        "rebuttal": "Natural variability is possible but insufficient",
    }
    sampler = _make_sampler(json.dumps(payload))
    result = await sampler.decompose("Climate change is real because CO2 is rising.")
    assert isinstance(result, ToulminComponents)
    assert result.claim == "Climate change is real"
    assert len(result.data) == 2
    assert result.warrant == "Greenhouse gases trap heat"
    assert result.backing == "IPCC reports"
    assert result.qualifier == "with high confidence"
    assert result.rebuttal is not None


async def test_decompose_minimal_response():
    """Only required fields — optional fields should default to None/empty."""
    payload = {"claim": "The sky is blue", "data": [], "warrant": "Wavelength of light"}
    sampler = _make_sampler(json.dumps(payload))
    result = await sampler.decompose("The sky is blue.")
    assert result.claim == "The sky is blue"
    assert result.data == []
    assert result.backing is None
    assert result.qualifier is None
    assert result.rebuttal is None


async def test_decompose_strips_markdown_fences():
    payload = {"claim": "X", "data": ["A"], "warrant": "B"}
    raw = f"```json\n{json.dumps(payload)}\n```"
    sampler = _make_sampler(raw)
    result = await sampler.decompose("X because A.")
    assert result.claim == "X"


async def test_decompose_caller_invoked():
    """The LLMCaller must be called exactly once per decompose() call."""
    call_count = 0

    async def _counting_caller(prompt, system, max_tokens, temperature):
        nonlocal call_count
        call_count += 1
        return json.dumps({"claim": "X", "data": [], "warrant": "Y"})

    sampler = SamplingHelper(_counting_caller, SamplingConfig())
    await sampler.decompose("X.")
    assert call_count == 1
