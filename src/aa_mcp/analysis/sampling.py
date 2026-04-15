"""SamplingHelper — all LLM call-back patterns via an LLMCaller."""

from __future__ import annotations

import json
import re

from aa_mcp.analysis.llm_caller import LLMCaller
from aa_mcp.config import SamplingConfig
from aa_mcp.models.argument import ComponentAssessment, FallacyDetection, ToulminComponents
from aa_mcp.models.rag import SearchResult


class SamplingHelper:
    """Wraps an LLMCaller with typed methods for each analysis task.

    ``caller`` is any async callable ``(prompt, system, max_tokens, temperature) -> str``.
    In the MCP server this is created via ``mcp_session_caller(session)``.
    In standalone use any conforming async function works.
    """

    def __init__(self, caller: LLMCaller, config: SamplingConfig) -> None:
        self._caller = caller
        self._cfg = config

    async def _call(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float = 0.1) -> str:
        return await self._caller(prompt, system_prompt, max_tokens, temperature)

    async def decompose(self, text: str) -> ToulminComponents:
        """Extract argument components from text."""
        system = (
            "You are an expert in argument analysis and critical reasoning. "
            "Always respond with valid JSON only — no markdown, no explanation."
        )
        prompt = f"""Analyze the following text and identify its argument components.

Return a JSON object with these keys:
- "claim": the main assertion or conclusion being argued for (string)
- "data": list of evidence, facts, or premises offered in support (array of strings)
- "warrant": the principle or reasoning that connects the evidence to the claim (string, may be implicit — infer it if unstated)
- "backing": additional grounding for the reasoning, such as established principles or cited authority (string or null)
- "qualifier": any stated limitations, degrees of certainty, or scope restrictions (string or null)
- "rebuttal": any counter-arguments or exceptions the author acknowledges (string or null)

Text to analyze:
\"\"\"
{text}
\"\"\"

JSON response:"""

        raw = await self._call(prompt, system, self._cfg.max_tokens_decompose, self._cfg.temperature_analysis)
        return ToulminComponents.model_validate(_parse_json(raw))

    async def assess_component(
        self,
        component_type: str,
        component_text: str,
        evidence_chunks: list[SearchResult],
    ) -> ComponentAssessment:
        """Assess a single argument component for logical quality, with KB evidence as supplementary context."""
        system = (
            "You are an expert in critical thinking, logic, and argumentation. "
            "Always respond with valid JSON only — no markdown, no explanation."
        )

        evidence_block = _format_evidence(evidence_chunks)
        has_evidence = bool(evidence_chunks)

        prompt = f"""Assess the following argument component.

Component role: {component_type}
Component text: "{component_text}"

{"Knowledge base evidence:" if has_evidence else "No knowledge base evidence available for this component."}
{evidence_block if has_evidence else ""}

Return a JSON object with:
- "logical_validity_score": float 0.0-1.0. PRIMARY SCORE. Evaluate logical soundness on its own merits — is the reasoning valid? Is the evidence relevant and sufficient? Score based on logic alone, independent of the knowledge base.
- "knowledge_base_score": float 0.0-1.0 or null. SECONDARY SCORE. How well is this component supported by the provided knowledge base evidence? Use null if no knowledge base evidence was provided — do NOT default to 0.5. Use 0.0-0.4 if evidence actively contradicts this component; 0.5-0.7 if neutral or mixed; 0.8-1.0 if strongly supported.
- "supporting_evidence": array of short quote strings from the knowledge base that support this component (empty array if none)
- "contradicting_evidence": array of short quote strings from the knowledge base that contradict this component (empty array if none)
- "weakness_flag": boolean — true if logical_validity_score is below 0.4, OR if knowledge base evidence actively contradicts this component
- "weakness_explanation": string explaining the weakness (null if weakness_flag is false)

JSON response:"""

        raw = await self._call(prompt, system, self._cfg.max_tokens_assess, self._cfg.temperature_analysis)
        data = _parse_json(raw)
        kb_raw = data.get("knowledge_base_score")
        return ComponentAssessment(
            component_type=component_type,
            text=component_text,
            logical_validity_score=float(data.get("logical_validity_score", 0.5)),
            knowledge_base_score=float(kb_raw) if kb_raw is not None else None,
            supporting_evidence=data.get("supporting_evidence", []),
            contradicting_evidence=data.get("contradicting_evidence", []),
            weakness_flag=bool(data.get("weakness_flag", False)),
            weakness_explanation=data.get("weakness_explanation"),
        )

    async def detect_fallacies(self, text: str) -> list[FallacyDetection]:
        """Identify logical fallacies in the full argument text."""
        system = (
            "You are an expert in logic and informal fallacies. "
            "Always respond with valid JSON only — no markdown, no explanation."
        )
        prompt = f"""Identify any logical fallacies in the following text.

Return a JSON array. Each element should have:
- "fallacy_type": snake_case name (e.g. "ad_hominem", "straw_man", "false_dichotomy", "appeal_to_authority", "slippery_slope", "hasty_generalization", "circular_reasoning", "red_herring")
- "excerpt": the specific passage containing the fallacy (short quote)
- "explanation": why this is a fallacy and how it weakens the argument
- "severity": "low" | "medium" | "high"

If no fallacies are found, return an empty array [].

Text:
\"\"\"
{text}
\"\"\"

JSON response:"""

        raw = await self._call(prompt, system, self._cfg.max_tokens_fallacy, self._cfg.temperature_analysis)
        data = _parse_json(raw)
        if not isinstance(data, list):
            return []
        return [FallacyDetection(**f) for f in data if _valid_fallacy(f)]

    async def synthesize_assessment(
        self,
        source_text: str,
        toulmin: ToulminComponents,
        component_assessments: list[ComponentAssessment],
        fallacies: list[FallacyDetection],
    ) -> tuple[str, float, float | None]:
        """Produce prose summary and overall scores. Returns (summary, validity_score, truthfulness_score)."""
        system = (
            "You are an expert argument analyst. "
            "Provide a clear, direct assessment. "
            "Always respond with valid JSON only — no markdown fences, no explanation outside JSON."
        )

        weak_components = [ca for ca in component_assessments if ca.weakness_flag]
        weak_block = ""
        if weak_components:
            weak_block = "\n\nCRITICAL WEAKNESSES IDENTIFIED:\n" + "\n".join(
                f"- [{ca.component_type.upper()}] {ca.text[:200]}: {ca.weakness_explanation}"
                for ca in weak_components
            )

        fallacy_block = ""
        if fallacies:
            fallacy_block = "\n\nFALLACIES DETECTED:\n" + "\n".join(
                f"- {f.fallacy_type} ({f.severity}): {f.explanation}"
                for f in fallacies
            )

        def _kb_part(ca: ComponentAssessment) -> str:
            return f", kb={ca.knowledge_base_score:.2f}" if ca.knowledge_base_score is not None else ""

        comp_block = "\n".join(
            f"- {ca.component_type}: logic={ca.logical_validity_score:.2f}{_kb_part(ca)}"
            for ca in component_assessments
        )

        has_kb_evidence = any(ca.knowledge_base_score is not None for ca in component_assessments)

        prompt = f"""You have analyzed an argument. Produce a final assessment report.

ORIGINAL ARGUMENT:
\"\"\"{source_text[:500]}\"\"\"

CLAIM: {toulmin.claim}
COMPONENT SCORES:
{comp_block}
{weak_block}
{fallacy_block}

Return a JSON object with:
- "overall_validity_score": float 0.0-1.0. Overall logical soundness — weight based on logic scores across components and any detected fallacies.
- "overall_truthfulness_score": float 0.0-1.0 or null. Overall alignment with knowledge base sources. {"Score based on kb scores above." if has_kb_evidence else "Set to null — no knowledge base evidence was available."}
- "summary": string — 2-4 sentence prose assessment. Lead with logical quality. If knowledge base evidence was checked, note factual findings. Explicitly call out any critical weaknesses by name.

JSON response:"""

        raw = await self._call(prompt, system, self._cfg.max_tokens_synthesize, self._cfg.temperature_synthesis)
        data = _parse_json(raw)
        summary = str(data.get("summary", "Assessment complete."))
        validity = float(data.get("overall_validity_score", 0.5))
        truthfulness_raw = data.get("overall_truthfulness_score")
        truthfulness = float(truthfulness_raw) if truthfulness_raw is not None else None
        return summary, validity, truthfulness

    async def synthesize_rag_answer(self, query: str, results: list[SearchResult]) -> str:
        """Synthesize an answer from RAG results."""
        system = "You are a helpful assistant. Answer using only the provided context. Cite sources with [N]."
        context = _format_evidence(results)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return await self._call(prompt, system, self._cfg.max_tokens_synthesize, self._cfg.temperature_synthesis)

    async def infer_relationships(self, nodes: list[dict]) -> list[dict]:
        """Suggest edges between argument nodes."""
        system = "You are an expert in argument structure. Respond with valid JSON only."
        nodes_text = "\n".join(f"[{n['id'][:8]}] ({n['node_type']}) {n['text']}" for n in nodes)
        prompt = f"""Given these argument nodes, identify meaningful relationships between them.

Nodes:
{nodes_text}

Return a JSON array of edges. Each edge:
- "source_id": full ID of source node
- "target_id": full ID of target node
- "relationship": "supports" | "attacks" | "qualifies" | "elaborates"
- "explanation": brief explanation (1 sentence)

Only include relationships you are confident about. Return [] if none are clear.

JSON response:"""

        raw = await self._call(prompt, system, 1024, self._cfg.temperature_analysis)
        data = _parse_json(raw)
        return data if isinstance(data, list) else []


def _format_evidence(results: list[SearchResult]) -> str:
    if not results:
        return "(no evidence retrieved)"
    return "\n\n".join(
        f"[{i + 1}] (score={r.score:.3f}) {r.chunk.text[:400]}"
        for i, r in enumerate(results)
    )


def _parse_json(raw: str) -> dict | list:
    """Extract and parse JSON from LLM response, stripping any markdown fences."""
    raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"[\[{]", raw)
    if match:
        raw = raw[match.start():]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned unparseable JSON: {exc}. Raw response: {raw[:200]!r}") from exc


def _valid_fallacy(f: dict) -> bool:
    return all(k in f for k in ("fallacy_type", "excerpt", "explanation", "severity"))
