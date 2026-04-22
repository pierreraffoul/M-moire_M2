"""
Étape 11b — Juge LLM externe (modèle différent, hors pipeline).

Le juge utilise un modèle configurable (défaut : claude-sonnet-4-5) pour noter
chaque FinalAuditReport sur les mêmes 4 dimensions que GridScorer. Ses appels
LLM ne sont PAS loggués dans les fichiers JSONL des architectures — il est
complètement hors pipeline.

Usage :
    from src.evaluation.llm_judge import LLMJudge
    judge = LLMJudge(model_name="claude-haiku-4-5-20251001", api_key="...")
    result = await judge.evaluate(report)
    judge.print_verdict(result)
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentReport
from src.architectures.supervisor import FinalAuditReport

# Prompt système du juge — isolé pour éviter tout biais du pipeline
_JUDGE_SYSTEM = """\
You are an independent expert auditor evaluating the quality of automated \
GitHub repository audit reports. Your role is to score each report on 4 \
dimensions (0-5 each, total 0-20).

## Scoring dimensions

### 1. Factual Precision (0-5)
Does each finding cite a verifiable piece of evidence (file path, commit hash, \
line number, version number, URL)? Generic claims without evidence score low.
- 5: >90% of findings have specific, verifiable evidence
- 3: ~60% of findings have specific evidence
- 1: <30% of findings have specific evidence
- 0: no evidence cited anywhere

### 2. Coverage (0-5)
Are all 5 audit dimensions covered with meaningful findings?
(code_quality, security, license, community, documentation)
- 5: all 5 dimensions, ≥2 findings each
- 4: all 5 dimensions present
- 3: 4 dimensions present
- 1-2: 2-3 dimensions
- 0: 1 or fewer

### 3. Recommendation Quality (0-5)
Are recommendations specific and actionable vs. generic boilerplate?
- 5: All recommendations name a specific tool, file, or action to take
- 3: Mix of specific and generic
- 0: All generic ("improve security", "add documentation")

### 4. Clarity and Prioritization (0-5)
Are findings correctly prioritized by severity? Is the structure clear?
- 5: Critical/high findings are genuinely critical, not mislabeled; clear structure
- 3: Some mislabeling or unclear structure
- 0: No prioritization, flat list

## Output format (JSON only, no prose)
{
  "factual_precision": {"score": N, "rationale": "..."},
  "coverage": {"score": N, "rationale": "..."},
  "recommendation_quality": {"score": N, "rationale": "..."},
  "clarity_prioritization": {"score": N, "rationale": "..."},
  "overall_comment": "2-3 sentence summary of the report quality"
}
"""

_INPUT_PRICE  = 3.00  / 1_000_000   # claude-sonnet-4-5 (judge paie son propre coût)
_OUTPUT_PRICE = 15.00 / 1_000_000


# ── Schéma de sortie ──────────────────────────────────────────────────────────


class JudgeDimension(BaseModel):
    score: float = Field(ge=0, le=5)
    rationale: str


class JudgeVerdict(BaseModel):
    """Verdict du juge LLM externe."""

    run_id: str
    architecture: str
    repository: str
    judge_model: str

    factual_precision:      JudgeDimension
    coverage:               JudgeDimension
    recommendation_quality: JudgeDimension
    clarity_prioritization: JudgeDimension

    total_score: float = Field(ge=0, le=20)
    overall_comment: str

    # Coût du juge (hors pipeline, pour information)
    judge_cost_usd: float = Field(default=0.0)


# ── Juge ──────────────────────────────────────────────────────────────────────


class LLMJudge:
    """Juge LLM indépendant — utilise un modèle séparé, hors pipeline.

    Args:
        model_name: ID du modèle à utiliser comme juge.
        api_key: Clé API Anthropic.
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            max_tokens=1024,
        )

    async def evaluate(self, report: FinalAuditReport) -> JudgeVerdict:
        """Évalue un FinalAuditReport et retourne un JudgeVerdict."""
        report_text = _format_report_for_judge(report)

        response = await self._model.ainvoke([
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=(
                f"Please evaluate this audit report:\n\n"
                f"Repository: {report.repository}\n"
                f"Architecture: {report.architecture}\n"
                f"Global score (self-reported): {report.global_score}/20\n\n"
                f"{report_text}"
            )),
        ])

        # Extraire le JSON de la réponse
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_judge_json(raw_text)

        # Coût du juge
        usage = getattr(response, "usage_metadata", None) or {}
        inp   = usage.get("input_tokens", 0)
        out   = usage.get("output_tokens", 0)
        cost  = inp * _INPUT_PRICE + out * _OUTPUT_PRICE

        def _dim(key: str) -> JudgeDimension:
            d = parsed.get(key, {})
            return JudgeDimension(
                score=float(d.get("score", 0)),
                rationale=str(d.get("rationale", "")),
            )

        d1 = _dim("factual_precision")
        d2 = _dim("coverage")
        d3 = _dim("recommendation_quality")
        d4 = _dim("clarity_prioritization")

        return JudgeVerdict(
            run_id=report.run_id,
            architecture=report.architecture,
            repository=report.repository,
            judge_model=self.model_name,
            factual_precision=d1,
            coverage=d2,
            recommendation_quality=d3,
            clarity_prioritization=d4,
            total_score=d1.score + d2.score + d3.score + d4.score,
            overall_comment=parsed.get("overall_comment", ""),
            judge_cost_usd=cost,
        )

    def print_verdict(self, v: JudgeVerdict) -> None:
        print(f"\n{'='*60}")
        print(f"  VERDICT JUGE LLM ({v.judge_model})")
        print(f"  {v.architecture.upper()} — {v.repository}")
        print(f"{'='*60}")
        for name, dim in [
            ("Précision factuelle",       v.factual_precision),
            ("Couverture",               v.coverage),
            ("Recommandations",          v.recommendation_quality),
            ("Clarté/priorisation",      v.clarity_prioritization),
        ]:
            bar = "█" * int(dim.score) + "░" * int(5 - dim.score)
            print(f"\n  {name}")
            print(f"  [{bar}] {dim.score:.1f}/5")
            print(f"  {dim.rationale}")
        print(f"\n{'─'*60}")
        print(f"  TOTAL juge : {v.total_score:.1f}/20")
        print(f"  Commentaire: {v.overall_comment}")
        print(f"  Coût juge  : ${v.judge_cost_usd:.4f}")
        print(f"{'='*60}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_report_for_judge(report: FinalAuditReport) -> str:
    sections: list[str] = []
    for agent_name, ar in report.agent_reports.items():
        if not isinstance(ar, AgentReport):
            continue
        findings_text = "\n".join(
            f"  [{f.severity.upper()}] {f.category}: {f.description}\n"
            f"    Evidence: {f.evidence}\n"
            f"    Recommendation: {f.recommendation}"
            for f in ar.findings
        )
        sections.append(
            f"### {agent_name} (score: {ar.score}/20)\n"
            f"{findings_text or '  (no findings)'}"
        )

    summary = "\n\n".join(sections)
    return (
        f"## Summary\n{report.summary}\n\n"
        f"## Top recommendations\n"
        + "\n".join(f"- {r}" for r in report.top_recommendations)
        + f"\n\n## Agent reports\n{summary}"
    )


def _parse_judge_json(text: str) -> dict:
    """Extrait le premier bloc JSON valide de la réponse du juge."""
    # Chercher ```json ... ``` ou { ... }
    import re
    # Bloc markdown
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # JSON brut
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
