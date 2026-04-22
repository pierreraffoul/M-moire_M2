"""
Étape 11c — Vérification factuelle des findings via l'API GitHub REST.

Échantillonne 3-5 findings par rapport et vérifie que l'evidence citée
correspond à une ressource réellement existante dans le dépôt. Utilise
l'API GitHub REST directe (pas via MCP, pas via les agents pipelines) pour
ne pas polluer les métriques du benchmark.

Stratégie de vérification :
  - Evidence contenant un chemin de fichier → check via GET /repos/{owner}/{repo}/contents/{path}
  - Evidence contenant un workflow → check via GET /repos/{owner}/{repo}/actions/workflows
  - Evidence contenant une URL GitHub → HTTP HEAD sur l'URL
  - Autres → marqué "unverifiable" (ne compte pas comme faux)

Usage :
    from src.evaluation.factual_checker import FactualChecker
    checker = FactualChecker(github_token="ghp_...")
    result = await checker.check(report, sample_size=5)
    print(f"Factual accuracy: {result.factual_accuracy_rate:.0%}")
"""

from __future__ import annotations

import random
import re
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentReport, Finding
from src.architectures.supervisor import FinalAuditReport

_GITHUB_API = "https://api.github.com"

# Regex pour extraire des chemins depuis l'evidence
_FILE_PATH_RE   = re.compile(r"[`'\"]?([\w./\-]+\.\w+)[`'\"]?")
_WORKFLOW_RE    = re.compile(r"\.github/workflows/[\w.\-]+\.ya?ml", re.IGNORECASE)
_GITHUB_URL_RE  = re.compile(r"https://github\.com/[\w.\-/]+")
_VERSION_RE     = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")


# ── Schémas de sortie ─────────────────────────────────────────────────────────


VerificationStatus = Literal["verified", "not_found", "unverifiable", "error"]


class VerifiedFinding(BaseModel):
    agent_name: str
    severity: str
    category: str
    description: str
    evidence: str
    status: VerificationStatus
    verification_note: str = ""


class FactualCheckReport(BaseModel):
    """Résultat de vérification factuelle sur un échantillon de findings."""

    run_id: str
    architecture: str
    repository: str

    sampled: int         = Field(description="Nombre de findings échantillonnés.")
    verified: int        = Field(description="Findings dont l'evidence a été confirmée.")
    not_found: int       = Field(description="Findings dont l'evidence est introuvable.")
    unverifiable: int    = Field(description="Findings dont l'evidence ne peut pas être vérifiée automatiquement.")
    errors: int          = Field(description="Erreurs lors de la vérification.")

    factual_accuracy_rate: float = Field(
        description="verified / (verified + not_found). Les unverifiable sont exclus."
    )

    details: list[VerifiedFinding]


# ── Checker ───────────────────────────────────────────────────────────────────


class FactualChecker:
    """Vérifie les findings via l'API GitHub REST directe.

    Args:
        github_token: Token GitHub avec scope repo:read.
        sample_size: Nombre de findings à vérifier par rapport.
        timeout_s: Timeout HTTP en secondes.
    """

    def __init__(
        self,
        github_token: str,
        sample_size: int = 5,
        timeout_s: float = 10.0,
    ) -> None:
        self.token       = github_token
        self.sample_size = sample_size
        self._headers    = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._timeout = httpx.Timeout(timeout_s)

    async def check(
        self,
        report: FinalAuditReport,
        sample_size: int | None = None,
    ) -> FactualCheckReport:
        """Échantillonne et vérifie des findings du rapport.

        Args:
            report: FinalAuditReport à vérifier.
            sample_size: Surcharge la valeur par défaut.

        Returns:
            FactualCheckReport avec taux de précision factuelle.
        """
        n = sample_size or self.sample_size
        owner, repo = report.repository.split("/", 1)

        # Collecter tous les findings avec leur agent d'origine
        all_pairs: list[tuple[str, Finding]] = []
        for agent_name, ar in report.agent_reports.items():
            if isinstance(ar, AgentReport):
                for f in ar.findings:
                    all_pairs.append((agent_name, f))

        # Échantillon aléatoire reproductible (seed = run_id)
        rng = random.Random(report.run_id)
        sample = rng.sample(all_pairs, min(n, len(all_pairs)))

        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
            details: list[VerifiedFinding] = []
            for agent_name, finding in sample:
                status, note = await self._verify_finding(client, owner, repo, finding)
                details.append(VerifiedFinding(
                    agent_name=agent_name,
                    severity=finding.severity,
                    category=finding.category,
                    description=finding.description[:120],
                    evidence=finding.evidence[:200],
                    status=status,
                    verification_note=note,
                ))

        cnt_verified     = sum(1 for d in details if d.status == "verified")
        cnt_not_found    = sum(1 for d in details if d.status == "not_found")
        cnt_unverifiable = sum(1 for d in details if d.status == "unverifiable")
        cnt_errors       = sum(1 for d in details if d.status == "error")

        verifiable = cnt_verified + cnt_not_found
        accuracy   = cnt_verified / verifiable if verifiable else 1.0

        return FactualCheckReport(
            run_id=report.run_id,
            architecture=report.architecture,
            repository=report.repository,
            sampled=len(details),
            verified=cnt_verified,
            not_found=cnt_not_found,
            unverifiable=cnt_unverifiable,
            errors=cnt_errors,
            factual_accuracy_rate=accuracy,
            details=details,
        )

    # ── Vérification par type d'evidence ──────────────────────────────────────

    async def _verify_finding(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        finding: Finding,
    ) -> tuple[VerificationStatus, str]:
        evidence = finding.evidence

        # 1. Workflow (.github/workflows/*.yml)
        wf_match = _WORKFLOW_RE.search(evidence)
        if wf_match:
            return await self._check_file_exists(client, owner, repo, wf_match.group(0))

        # 2. Chemin de fichier générique
        path_match = _FILE_PATH_RE.search(evidence)
        if path_match:
            candidate = path_match.group(1)
            if not candidate.startswith("http") and "." in candidate:
                return await self._check_file_exists(client, owner, repo, candidate)

        # 3. URL GitHub directe
        url_match = _GITHUB_URL_RE.search(evidence)
        if url_match:
            return await self._check_url(client, url_match.group(0))

        return "unverifiable", "No verifiable reference found in evidence."

    async def _check_file_exists(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        path: str,
    ) -> tuple[VerificationStatus, str]:
        url = f"{_GITHUB_API}/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
        try:
            r = await client.get(url)
            if r.status_code == 200:
                return "verified", f"File exists: {path}"
            elif r.status_code == 404:
                return "not_found", f"File not found via API: {path}"
            else:
                return "error", f"HTTP {r.status_code} for {path}"
        except httpx.TimeoutException:
            return "error", f"Timeout checking {path}"
        except Exception as exc:
            return "error", str(exc)

    async def _check_url(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[VerificationStatus, str]:
        try:
            r = await client.head(url, follow_redirects=True)
            if r.status_code < 400:
                return "verified", f"URL accessible: {url}"
            else:
                return "not_found", f"HTTP {r.status_code} for {url}"
        except httpx.TimeoutException:
            return "error", f"Timeout checking {url}"
        except Exception as exc:
            return "error", str(exc)

    def print_report(self, r: FactualCheckReport) -> None:
        print(f"\n{'='*60}")
        print(f"  VÉRIFICATION FACTUELLE — {r.architecture.upper()}")
        print(f"  {r.repository}  |  {r.sampled} findings échantillonnés")
        print(f"{'='*60}")
        for d in r.details:
            icon = {"verified": "✓", "not_found": "✗", "unverifiable": "?", "error": "!"}.get(d.status, "?")
            print(f"\n  [{icon}] [{d.severity.upper()}] {d.category}")
            print(f"      {d.description[:80]}")
            print(f"      Evidence : {d.evidence[:80]}")
            print(f"      → {d.verification_note}")
        print(f"\n{'─'*60}")
        print(f"  Verified     : {r.verified}")
        print(f"  Not found    : {r.not_found}")
        print(f"  Unverifiable : {r.unverifiable}")
        print(f"  Errors       : {r.errors}")
        print(f"  Accuracy     : {r.factual_accuracy_rate:.0%}  (verified / verifiable)")
        print(f"{'='*60}\n")
