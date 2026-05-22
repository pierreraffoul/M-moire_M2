#!/usr/bin/env python3
"""
Generate LangGraph diagrams for the mémoire.
No LLM or MCP calls — only graph structure (nodes + edges) is needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DIAGRAMS_DIR = ROOT / "results" / "diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

# ── Mock objects (aucun appel réseau) ─────────────────────────────────────────

mock_model = MagicMock()
mock_model.bind_tools.return_value = mock_model
mock_model.with_structured_output.return_value = MagicMock()
mock_model.kwargs = {"tools": []}

mock_mcp = MagicMock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_diagram(name: str, graph, xray: bool = False, description: str = "") -> None:
    """Sauvegarde le code Mermaid (.md) et l'image (.png) d'un graphe compilé."""
    drawable = graph.get_graph(xray=xray)
    mermaid_code = drawable.draw_mermaid()

    # Fichier .md avec code Mermaid prêt pour mermaid.live
    md_content = f"# {description or name}\n\n"
    md_content += "> Généré par `experiments/generate_diagrams.py`\n\n"
    md_content += f"```mermaid\n{mermaid_code}\n```\n"
    md_path = DIAGRAMS_DIR / f"{name}.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  ✓ {md_path.name}")

    # Fichier .png via mermaid.ink (HTTP, pas LLM)
    try:
        png_data = drawable.draw_mermaid_png()
        png_path = DIAGRAMS_DIR / f"{name}.png"
        png_path.write_bytes(png_data)
        print(f"  ✓ {png_path.name}")
    except Exception as exc:
        print(f"  ✗ PNG non généré ({exc})")
        print(f"    → Collez le code Mermaid sur https://mermaid.live")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Boucle ReAct (BaseAuditAgent)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Boucle ReAct (BaseAuditAgent)...")

from langchain_core.tools import tool
from src.agents.base_agent import BaseAuditAgent

@tool
def get_repository(owner: str, repo: str) -> str:
    """Get repository metadata."""
    return ""

@tool
def get_file_contents(owner: str, repo: str, path: str) -> str:
    """Get file contents from a repository."""
    return ""

@tool
def search_code(query: str) -> str:
    """Search code in a repository."""
    return ""

react_agent = BaseAuditAgent(
    name="code_quality",
    system_prompt="Audit agent.",
    allowed_mcp_tools=["get_repository", "get_file_contents", "search_code"],
    model=mock_model,
    mcp_client=mock_mcp,
)
react_graph = react_agent._build_react_graph(mock_model, [get_repository, get_file_contents, search_code])

save_diagram(
    "graph_react_agent",
    react_graph,
    description="Boucle ReAct — BaseAuditAgent (partagée par les 5 agents spécialisés)",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture Supervisor
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Architecture Supervisor...")

from src.architectures.supervisor import SupervisorOrchestrator

supervisor_orch = SupervisorOrchestrator(model=mock_model, mcp_client=mock_mcp)
supervisor_graph = supervisor_orch._build_graph()

save_diagram(
    "graph_supervisor",
    supervisor_graph,
    description="Architecture Supervisor — LLM orchestrateur séquentiel (1 superviseur LLM)",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Architecture Hierarchical — graphe parent
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Architecture Hierarchical...")

from src.architectures.hierarchical import (
    HierarchicalOrchestrator,
    _TECH_AGENTS,
    _COMMUNITY_AGENTS,
)

hier_orch = HierarchicalOrchestrator(model=mock_model, mcp_client=mock_mcp)

tech_subgraph      = hier_orch._build_team_subgraph("tech",      _TECH_AGENTS)
community_subgraph = hier_orch._build_team_subgraph("community", _COMMUNITY_AGENTS)
parent_graph       = hier_orch._build_parent_graph(tech_subgraph, community_subgraph)

# Vue parent (tech_team et community_team = boîtes noires)
save_diagram(
    "graph_hierarchical",
    parent_graph,
    xray=False,
    description="Architecture Hierarchical — graphe parent (3 niveaux de supervision LLM)",
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Architecture Decentralized
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Architecture Decentralized...")

from src.architectures.decentralized import DecentralizedOrchestrator

dec_orch = DecentralizedOrchestrator(model=mock_model, mcp_client=mock_mcp)
dec_graph = dec_orch._build_graph()

save_diagram(
    "graph_decentralized",
    dec_graph,
    description="Architecture Decentralized — handoffs Command(goto=...), 0 superviseur LLM",
)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nDiagrammes sauvegardés dans : {DIAGRAMS_DIR}")
print("Pour les PNG manquants → coller le code .md sur https://mermaid.live")
