"""
Étape 9 — Calcul des métriques de run depuis les fichiers JSONL.

Usage :
    from src.instrumentation.metrics import compute_metrics, RunMetrics
    m = compute_metrics(Path("results/logs/comparison-supervisor-001.jsonl"))
    print(m.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

# ── Tarifs claude-sonnet-4-5 ($/token) ────────────────────────────────────────
INPUT_PRICE_PER_TOKEN  = 3.00  / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 15.00 / 1_000_000
CACHE_READ_PER_TOKEN   = 0.30  / 1_000_000
CACHE_CREATE_PER_TOKEN = 3.75  / 1_000_000

# Noms d'agents considérés comme overhead d'orchestration
_OVERHEAD_AGENTS: frozenset[str] = frozenset({
    "supervisor", "synthesizer",
    "tech_supervisor", "community_supervisor", "top_supervisor",
    "tech_synthesizer", "community_synthesizer",
})


# ── Schéma de sortie ──────────────────────────────────────────────────────────


class RunMetrics(BaseModel):
    """Métriques agrégées d'un run d'architecture."""

    run_id: str
    architecture: str
    repository: str

    # MCP
    total_mcp_calls: int = Field(description="Nombre total d'appels MCP du run.")
    redundant_mcp_calls: int = Field(description="Appels MCP dont le hash a déjà été vu dans le même run.")
    redundancy_rate: float = Field(description="redundant / total, en [0, 1].")
    mcp_data_volume_bytes: int = Field(description="Somme des response_size_bytes de tous les appels MCP.")

    # LLM tokens
    total_tokens_input: int
    total_tokens_output: int
    total_tokens_cache_read: int
    total_tokens_cache_creation: int

    # Coûts
    total_cost_usd: float = Field(description="Coût total du run en dollars.")
    overhead_orchestration_cost_usd: float = Field(
        description="Coût des appels LLM supervisor/synthesizer uniquement."
    )

    # Durée
    wall_clock_duration_seconds: float = Field(
        description=(
            "Durée wall-clock mesurée à l'extérieur. "
            "Si non fournie, estimée depuis les timestamps JSONL."
        )
    )


# ── Fonction principale ────────────────────────────────────────────────────────


def compute_metrics(
    log_file: Path,
    wall_clock_duration_seconds: float | None = None,
) -> RunMetrics:
    """Lit un fichier JSONL et produit un RunMetrics.

    Args:
        log_file: Chemin vers le fichier .jsonl du run.
        wall_clock_duration_seconds: Durée wall-clock mesurée en externe.
            Si None, estimée depuis les timestamps du JSONL.

    Returns:
        RunMetrics calculé.
    """
    entries = _load_jsonl(log_file)

    # ── Identifiants du run ────────────────────────────────────────────────────
    run_id, architecture, repository = _extract_run_identity(entries)

    # ── MCP ───────────────────────────────────────────────────────────────────
    mcp_entries = [e for e in entries if e.get("event") == "mcp_call"]
    total_mcp   = len(mcp_entries)

    seen_hashes: set[str] = set()
    redundant   = 0
    data_volume = 0
    for e in mcp_entries:
        h = e.get("mcp_params_hash", "")
        if h and h in seen_hashes:
            redundant += 1
        elif h:
            seen_hashes.add(h)
        data_volume += e.get("response_size_bytes", 0)

    redundancy_rate = redundant / total_mcp if total_mcp else 0.0

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_entries = [e for e in entries if e.get("event") == "llm_call"]

    tok_in      = sum(e.get("input_tokens", 0)            for e in llm_entries)
    tok_out     = sum(e.get("output_tokens", 0)           for e in llm_entries)
    tok_cr      = sum(e.get("cache_read_tokens", 0)       for e in llm_entries)
    tok_cc      = sum(e.get("cache_creation_tokens", 0)   for e in llm_entries)

    total_cost    = _cost(llm_entries)
    overhead_cost = _cost([e for e in llm_entries if e.get("agent") in _OVERHEAD_AGENTS])

    # ── Durée ─────────────────────────────────────────────────────────────────
    if wall_clock_duration_seconds is not None:
        duration = wall_clock_duration_seconds
    else:
        duration = _estimate_duration(entries)

    return RunMetrics(
        run_id=run_id,
        architecture=architecture,
        repository=repository,
        total_mcp_calls=total_mcp,
        redundant_mcp_calls=redundant,
        redundancy_rate=redundancy_rate,
        mcp_data_volume_bytes=data_volume,
        total_tokens_input=tok_in,
        total_tokens_output=tok_out,
        total_tokens_cache_read=tok_cr,
        total_tokens_cache_creation=tok_cc,
        total_cost_usd=total_cost,
        overhead_orchestration_cost_usd=overhead_cost,
        wall_clock_duration_seconds=duration,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def reconstruct_final_report(log_file: Path) -> "FinalAuditReport | None":
    """Reconstruit un FinalAuditReport depuis un fichier JSONL (si findings persistés).

    Fonctionne avec les logs produits après l'ajout de findings dans agent_done.
    Retourne None si les findings ne sont pas dans le log.

    Args:
        log_file: Chemin vers le .jsonl du run.

    Returns:
        FinalAuditReport reconstruit, ou None si impossible.
    """
    # Import local pour éviter la dépendance circulaire au chargement du module
    from src.agents.base_agent import AgentReport, Finding
    from src.architectures.supervisor import FinalAuditReport

    entries = _load_jsonl(log_file)
    run_id, architecture, repository = _extract_run_identity(entries)

    # Collecter les agent_done avec findings
    agent_reports: dict[str, AgentReport] = {}
    for e in entries:
        if e.get("event") != "agent_done":
            continue
        agent_name = e.get("agent") or e.get("agent_name")
        score      = e.get("score", 0.0)
        raw_findings = e.get("findings")
        if raw_findings is None:
            continue  # log ancien sans findings sérialisés
        try:
            findings = [Finding(**f) for f in raw_findings]
        except Exception:
            findings = []
        agent_reports[agent_name] = AgentReport(
            agent_name=agent_name,
            repository=repository,
            score=score,
            findings=findings,
            raw_data={},  # non persisté
        )

    if not agent_reports:
        return None

    # Tenter de récupérer la synthèse finale
    summary = ""
    top_recs: list[str] = []
    global_score = 0.0
    for e in entries:
        if e.get("event") in ("supervisor_done", "hierarchical_done", "decentralized_done"):
            global_score = e.get("global_score", 0.0)
            break

    return FinalAuditReport(
        repository=repository,
        architecture=architecture,
        run_id=run_id,
        global_score=global_score,
        summary=summary or "Reconstructed from log.",
        top_recommendations=top_recs,
        agent_reports=agent_reports,
        supervisor_iterations=0,
        total_mcp_calls=0,
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _extract_run_identity(entries: list[dict]) -> tuple[str, str, str]:
    """Extrait run_id, architecture, repository depuis les premières entrées."""
    for e in entries:
        run_id      = e.get("run_id", "")
        architecture = e.get("architecture", "")
        repository  = e.get("repository", "")
        if run_id and architecture and repository:
            return run_id, architecture, repository
    return "unknown", "unknown", "unknown"


def _cost(llm_entries: list[dict]) -> float:
    total = 0.0
    for e in llm_entries:
        cc  = e.get("cache_creation_tokens", 0)
        cr  = e.get("cache_read_tokens", 0)
        inp = e.get("input_tokens", 0)
        out = e.get("output_tokens", 0)
        total += (
            (inp - cc - cr) * INPUT_PRICE_PER_TOKEN
            + cc * CACHE_CREATE_PER_TOKEN
            + cr * CACHE_READ_PER_TOKEN
            + out * OUTPUT_PRICE_PER_TOKEN
        )
    return total


def _estimate_duration(entries: list[dict]) -> float:
    """Estime la durée depuis les timestamps ISO des entrées JSONL."""
    from datetime import datetime, timezone

    timestamps: list[datetime] = []
    for e in entries:
        ts_str = e.get("timestamp") or e.get("timestamp_start")
        if not ts_str:
            continue
        try:
            # Supporte les formats avec ou sans fuseau horaire
            ts_str = ts_str.rstrip("Z")
            if "+" in ts_str:
                ts_str = ts_str.split("+")[0]
            dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            timestamps.append(dt)
        except ValueError:
            continue

    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds()
