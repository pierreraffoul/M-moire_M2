"""
Étape 10 — Détection et analyse des redondances MCP inter-agents.

La figure centrale : la matrice paire-à-paire qui montre combien d'appels MCP
identiques sont partagés entre chaque couple d'agents (ex: code_quality ×
security = 3 doublons). C'est la contribution la plus originale du mémoire.

Usage :
    from src.instrumentation.redundancy_detector import analyze_redundancy
    report = analyze_redundancy(Path("results/logs/comparison-supervisor-001.jsonl"))
    print(report.model_dump_json(indent=2))
    report.print_matrix()   # affichage ASCII de la matrice
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field


# ── Schémas de sortie ─────────────────────────────────────────────────────────


class RedundantCall(BaseModel):
    """Un appel MCP détecté comme redondant avec un appel précédent."""

    hash: str = Field(description="mcp_params_hash identifiant la requête.")
    tool: str = Field(description="Nom de l'outil MCP appelé.")
    params: dict = Field(description="Paramètres de l'appel.")
    first_agent: str = Field(description="Agent qui a effectué l'appel en premier.")
    duplicate_agent: str = Field(description="Agent qui a réémis la même requête.")
    first_timestamp: str
    duplicate_timestamp: str


class RedundancyReport(BaseModel):
    """Rapport complet de redondance pour un run."""

    run_id: str
    architecture: str
    repository: str

    total_mcp_calls: int
    unique_mcp_calls: int
    redundant_mcp_calls: int
    redundancy_rate: float = Field(description="(total - unique) / total, en [0,1].")

    # Matrice paire-à-paire : "agent_A×agent_B" → nombre de doublons
    redundancy_matrix: dict[str, int] = Field(
        description=(
            "Nombre d'appels MCP redondants entre chaque paire d'agents. "
            "Clé: 'agent_A×agent_B' (alphabétique), valeur: nb de doublons."
        )
    )

    # Outils les plus redondants : tool_name → nb d'appels redondants
    redundant_tools: dict[str, int] = Field(
        description="Classement des outils MCP par nombre d'appels redondants."
    )

    # Détail des doublons
    redundancy_details: list[RedundantCall] = Field(
        description="Liste exhaustive de tous les appels redondants détectés."
    )

    def print_matrix(self) -> None:
        """Affiche la matrice paire-à-paire en ASCII."""
        if not self.redundancy_matrix:
            print("  (aucune redondance inter-agents détectée)")
            return

        # Collecter tous les agents impliqués
        agents: set[str] = set()
        for key in self.redundancy_matrix:
            a, b = key.split("×")
            agents.add(a)
            agents.add(b)
        agents_sorted = sorted(agents)

        col_w = max(len(a) for a in agents_sorted) + 2

        # En-tête
        header = f"{'':>{col_w}}"
        for a in agents_sorted:
            header += f"  {a:>{col_w}}"
        print(header)
        print("-" * len(header))

        # Lignes
        for row_agent in agents_sorted:
            line = f"{row_agent:>{col_w}}"
            for col_agent in agents_sorted:
                if row_agent == col_agent:
                    line += f"  {'—':>{col_w}}"
                else:
                    key = "×".join(sorted([row_agent, col_agent]))
                    val = self.redundancy_matrix.get(key, 0)
                    line += f"  {val:>{col_w}}"
            print(line)

        print()
        print("  Légende : nombre d'appels MCP identiques partagés entre 2 agents")


# ── Fonction principale ────────────────────────────────────────────────────────


def analyze_redundancy(log_file: Path) -> RedundancyReport:
    """Analyse les redondances MCP dans un fichier JSONL.

    Args:
        log_file: Chemin vers le .jsonl du run.

    Returns:
        RedundancyReport avec matrice paire-à-paire et détails.
    """
    entries = [
        json.loads(l)
        for l in log_file.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    # Identité du run
    run_id, architecture, repository = _extract_identity(entries)

    mcp_entries = [e for e in entries if e.get("event") == "mcp_call"]
    total = len(mcp_entries)

    # hash → (agent, timestamp) du PREMIER appel
    first_seen: dict[str, tuple[str, str]] = {}

    matrix: dict[str, int]      = defaultdict(int)   # "A×B" → nb doublons
    tool_counts: dict[str, int] = defaultdict(int)   # tool → nb doublons
    details: list[RedundantCall] = []

    for e in mcp_entries:
        h         = e.get("mcp_params_hash", "")
        agent     = e.get("agent_name", "unknown")
        tool      = e.get("mcp_tool", "unknown")
        params    = e.get("mcp_params", {})
        timestamp = e.get("timestamp", "")

        if not h:
            continue

        if h not in first_seen:
            first_seen[h] = (agent, timestamp)
        else:
            first_agent, first_ts = first_seen[h]
            # Clé symétrique ordonnée alphabétiquement
            pair_key = "×".join(sorted([first_agent, agent]))
            matrix[pair_key] += 1
            tool_counts[tool] += 1
            details.append(RedundantCall(
                hash=h,
                tool=tool,
                params=params,
                first_agent=first_agent,
                duplicate_agent=agent,
                first_timestamp=first_ts,
                duplicate_timestamp=timestamp,
            ))

    unique    = len(first_seen)
    redundant = total - unique
    rate      = redundant / total if total else 0.0

    # Trier les outils par nb de doublons décroissant
    sorted_tools = dict(
        sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    )

    return RedundancyReport(
        run_id=run_id,
        architecture=architecture,
        repository=repository,
        total_mcp_calls=total,
        unique_mcp_calls=unique,
        redundant_mcp_calls=redundant,
        redundancy_rate=rate,
        redundancy_matrix=dict(matrix),
        redundant_tools=sorted_tools,
        redundancy_details=details,
    )


# ── Helper ────────────────────────────────────────────────────────────────────


def _extract_identity(entries: list[dict]) -> tuple[str, str, str]:
    for e in entries:
        run_id       = e.get("run_id", "")
        architecture = e.get("architecture", "")
        repository   = e.get("repository", "")
        if run_id and architecture and repository:
            return run_id, architecture, repository
    return "unknown", "unknown", "unknown"
