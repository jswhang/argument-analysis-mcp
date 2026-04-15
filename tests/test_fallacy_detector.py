"""Tests for SamplingHelper.detect_fallacies()."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aa_mcp.analysis.sampling import SamplingHelper
from aa_mcp.config import SamplingConfig
from aa_mcp.models.argument import FallacyDetection


def _make_sampler(response_text: str) -> SamplingHelper:
    async def _caller(prompt, system, max_tokens, temperature):
        return response_text

    return SamplingHelper(_caller, SamplingConfig())


async def test_detect_fallacies_returns_list():
    payload = [
        {
            "fallacy_type": "ad_hominem",
            "excerpt": "you're biased",
            "explanation": "Attacks the person, not the argument.",
            "severity": "medium",
        }
    ]
    sampler = _make_sampler(json.dumps(payload))
    result = await sampler.detect_fallacies("You're biased so your argument is wrong.")
    assert len(result) == 1
    assert isinstance(result[0], FallacyDetection)
    assert result[0].fallacy_type == "ad_hominem"
    assert result[0].severity == "medium"


async def test_detect_fallacies_empty_when_none_found():
    sampler = _make_sampler("[]")
    result = await sampler.detect_fallacies("This is a well-reasoned argument.")
    assert result == []


async def test_detect_fallacies_multiple():
    payload = [
        {
            "fallacy_type": "straw_man",
            "excerpt": "nobody wants that",
            "explanation": "Misrepresents the opposing position.",
            "severity": "high",
        },
        {
            "fallacy_type": "false_dichotomy",
            "excerpt": "either you agree or you're wrong",
            "explanation": "Presents only two options when more exist.",
            "severity": "medium",
        },
    ]
    sampler = _make_sampler(json.dumps(payload))
    result = await sampler.detect_fallacies("Some argument text.")
    assert len(result) == 2
    types = {f.fallacy_type for f in result}
    assert "straw_man" in types
    assert "false_dichotomy" in types


async def test_detect_fallacies_filters_incomplete_items():
    """Items missing required keys should be silently dropped."""
    payload = [
        {
            "fallacy_type": "ad_hominem",
            "excerpt": "you're biased",
            "explanation": "Attacks the person.",
            "severity": "low",
        },
        {
            "fallacy_type": "incomplete_item",
            # missing excerpt, explanation, severity
        },
    ]
    sampler = _make_sampler(json.dumps(payload))
    result = await sampler.detect_fallacies("Some text.")
    assert len(result) == 1
    assert result[0].fallacy_type == "ad_hominem"


async def test_detect_fallacies_non_list_response_returns_empty():
    """If LLM returns an object instead of array, return empty list gracefully."""
    sampler = _make_sampler('{"error": "unexpected"}')
    result = await sampler.detect_fallacies("Some text.")
    assert result == []


async def test_detect_fallacies_strips_markdown_fences():
    payload = [
        {
            "fallacy_type": "slippery_slope",
            "excerpt": "if we allow X",
            "explanation": "Assumes a chain of events without justification.",
            "severity": "low",
        }
    ]
    raw = f"```json\n{json.dumps(payload)}\n```"
    sampler = _make_sampler(raw)
    result = await sampler.detect_fallacies("If we allow X, everything will collapse.")
    assert len(result) == 1
    assert result[0].fallacy_type == "slippery_slope"
