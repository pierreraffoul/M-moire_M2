"""
Étape 11a — Notation par grille de critères (sans LLM).

Grille 4 dimensions × 5 points = 20 points max :

  1. Précision factuelle (0-5)
     Part de findings avec une evidence non-vide et spécifique
     (contient un chemin, un nom de fichier, un hash, une URL, un nombre…)

  2. Couverture (0-5)
     Les 5 dimensions d'audit sont-elles présentes ?
     (code_quality, security, license, community, documentation)

  3. Pertinence des recommandations (0-5)
     Proportion de recommandations spécifiques vs génériques
     (spécifique = contient un verbe d'action + un nom concret)

  4. Clarté et priorisation (0-5)
     Findings correctement priorisés (critical/high > medium > low/info),
     avec severity explicite, bien structurés

Usage :
    from src.evaluation.grid_scorer import GridScorer
    from src.architectures.supervisor import FinalAuditReport
    scorer = GridScorer()
    result = scorer.score(report)
    scorer.print_scorecard(result)
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base_agent import AgentReport, Finding
from src.architectures.supervisor import FinalAuditReport

# Catégories couvrant les 5 dimensions attendues
_COVERAGE_DIMENSIONS: list[str] = [
    "code_quality",
    "security",
    "license",
    "community",
    "documentation",
]

# Regex : evidence spécifique si elle contient l'un de ces patterns
_SPECIFIC_EVIDENCE_PATTERN = re.compile(
    r"[./\\]|"           # chemin de fichier
    r"\b[0-9a-f]{7,}\b|" # hash git / sha
    r"https?://|"        # URL
    r"\bv?\d+\.\d+|"     # version sémantique
    r"\bline\s+\d+|"     # numéro de ligne
    r"\b\d{4}-\d{2}-\d{2}",  # date
    re.IGNORECASE,
)

# Patterns pour recommandation spécifique (verbe d'action + substantif concret)
_ACTION_VERBS = re.compile(
    r"\b(add|enable|configure|update|remove|replace|set|create|fix|migrate|"
    r"pin|bump|restrict|enforce|document|publish|define|ajouter|activer|"
    r"configurer|mettre à jour|supprimer|corriger|définir)\b",
    re.IGNORECASE,
)
_CONCRETE_NOUN = re.compile(
    r"\b(workflow|action|dependabot|codeql|badge|license|readme|changelog|"
    r"pyproject|setup\.py|requirements|tox|pre-commit|ci|cd|secret|token|"
    r"branch|pr|issue|release|version|tag|hook|test|coverage|lint|type|"
    r"mypy|ruff|flake8|black|bandit|snyk|sast|sbom|cve|cwe|ossf|scorecard)\b",
    re.IGNORECASE,
)

_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


# ── Schéma de sortie ──────────────────────────────────────────────────────────


class DimensionScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=5)
    max_score: float = 5.0
    rationale: str


class GridScore(BaseModel):
    """Résultat de notation par grille."""

    run_id: str
    architecture: str
    repository: str

    factual_precision:     DimensionScore
    coverage:              DimensionScore
    recommendation_quality: DimensionScore
    clarity_prioritization: DimensionScore

    total_score: float = Field(ge=0, le=20, description="Somme des 4 dimensions.")

    # Statistiques utiles
    total_findings: int
    agents_present: list[str]


# ── Scorer ────────────────────────────────────────────────────────────────────


class GridScorer:
    """Évalue un FinalAuditReport sur 4 dimensions sans appel LLM."""

    def score(self, report: FinalAuditReport) -> GridScore:
        all_findings: list[Finding] = []
        for ar in report.agent_reports.values():
            if isinstance(ar, AgentReport):
                all_findings.extend(ar.findings)

        d1 = self._score_factual_precision(all_findings)
        d2 = self._score_coverage(report.agent_reports)
        d3 = self._score_recommendations(all_findings)
        d4 = self._score_clarity(all_findings)

        return GridScore(
            run_id=report.run_id,
            architecture=report.architecture,
            repository=report.repository,
            factual_precision=d1,
            coverage=d2,
            recommendation_quality=d3,
            clarity_prioritization=d4,
            total_score=d1.score + d2.score + d3.score + d4.score,
            total_findings=len(all_findings),
            agents_present=list(report.agent_reports.keys()),
        )

    # ── Dimension 1 : Précision factuelle ─────────────────────────────────────

    def _score_factual_precision(self, findings: list[Finding]) -> DimensionScore:
        if not findings:
            return DimensionScore(
                name="Précision factuelle",
                score=0.0,
                rationale="Aucun finding → score nul.",
            )
        specific = sum(
            1 for f in findings
            if f.evidence and bool(_SPECIFIC_EVIDENCE_PATTERN.search(f.evidence))
        )
        rate = specific / len(findings)
        score = round(rate * 5, 1)
        return DimensionScore(
            name="Précision factuelle",
            score=score,
            rationale=(
                f"{specific}/{len(findings)} findings avec evidence spécifique "
                f"({rate:.0%}) → {score}/5"
            ),
        )

    # ── Dimension 2 : Couverture ──────────────────────────────────────────────

    def _score_coverage(self, agent_reports: dict) -> DimensionScore:
        present = [dim for dim in _COVERAGE_DIMENSIONS if dim in agent_reports]
        rate = len(present) / len(_COVERAGE_DIMENSIONS)
        score = round(rate * 5, 1)
        missing = [d for d in _COVERAGE_DIMENSIONS if d not in present]
        return DimensionScore(
            name="Couverture",
            score=score,
            rationale=(
                f"{len(present)}/{len(_COVERAGE_DIMENSIONS)} dimensions présentes. "
                + (f"Manquantes: {missing}." if missing else "Couverture complète.")
            ),
        )

    # ── Dimension 3 : Pertinence des recommandations ──────────────────────────

    def _score_recommendations(self, findings: list[Finding]) -> DimensionScore:
        if not findings:
            return DimensionScore(
                name="Pertinence des recommandations",
                score=0.0,
                rationale="Aucun finding.",
            )
        specific = sum(
            1 for f in findings
            if (
                bool(_ACTION_VERBS.search(f.recommendation))
                and bool(_CONCRETE_NOUN.search(f.recommendation))
            )
        )
        rate = specific / len(findings)
        score = round(rate * 5, 1)
        return DimensionScore(
            name="Pertinence des recommandations",
            score=score,
            rationale=(
                f"{specific}/{len(findings)} recommandations spécifiques "
                f"({rate:.0%}) → {score}/5"
            ),
        )

    # ── Dimension 4 : Clarté et priorisation ─────────────────────────────────

    def _score_clarity(self, findings: list[Finding]) -> DimensionScore:
        if not findings:
            return DimensionScore(
                name="Clarté et priorisation",
                score=0.0,
                rationale="Aucun finding.",
            )

        # Vérifier que la severity est toujours valide
        valid_severity = sum(
            1 for f in findings if f.severity in _SEVERITY_ORDER
        )
        sev_rate = valid_severity / len(findings)

        # Vérifier que les findings sont bien priorisés (au moins 1 high/critical
        # et distribution non triviale)
        severity_counts = {}
        for f in findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        has_high = any(s in severity_counts for s in ("critical", "high"))
        has_variety = len(severity_counts) >= 3

        # Vérifier que category et description sont non-vides
        structured = sum(
            1 for f in findings
            if f.category and len(f.description) > 20
        )
        struct_rate = structured / len(findings)

        # Score composé
        base = (sev_rate * 0.4 + struct_rate * 0.4 + (0.1 if has_high else 0) + (0.1 if has_variety else 0))
        score = round(min(base * 5, 5.0), 1)

        return DimensionScore(
            name="Clarté et priorisation",
            score=score,
            rationale=(
                f"Severity valide: {valid_severity}/{len(findings)} ({sev_rate:.0%}), "
                f"Structurés: {structured}/{len(findings)} ({struct_rate:.0%}), "
                f"High/critical: {'oui' if has_high else 'non'}, "
                f"Variété: {'oui' if has_variety else 'non'} "
                f"→ {score}/5"
            ),
        )

    # ── Affichage ──────────────────────────────────────────────────────────────

    def print_scorecard(self, gs: GridScore) -> None:
        print(f"\n{'='*60}")
        print(f"  GRILLE DE NOTATION — {gs.architecture.upper()}")
        print(f"  {gs.repository}  |  run_id: {gs.run_id}")
        print(f"{'='*60}")
        for dim in [
            gs.factual_precision,
            gs.coverage,
            gs.recommendation_quality,
            gs.clarity_prioritization,
        ]:
            bar = "█" * int(dim.score) + "░" * int(5 - dim.score)
            print(f"\n  {dim.name}")
            print(f"  [{bar}] {dim.score:.1f}/5")
            print(f"  {dim.rationale}")
        print(f"\n{'─'*60}")
        print(f"  TOTAL : {gs.total_score:.1f} / 20")
        print(f"  (LLM self-score dans le rapport : comparer pour biais)")
        print(f"{'='*60}\n")
