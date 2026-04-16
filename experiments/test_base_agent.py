"""
Test de l'agent CodeQuality sur pallets/flask (étape 4).

Vérifie que :
- La boucle ReAct fonctionne (model → tools → model → ... → synthesize)
- L'AgentReport est structuré et cohérent (findings, score, raw_data)
- Les logs MCP sont produits et persistés en JSONL

Usage :
    uv run python experiments/test_base_agent.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path (nécessaire car le .pth d'install éditable
# ne se propage pas toujours avec les chemins Unicode sous macOS)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_anthropic import ChatAnthropic

from src.agents.code_quality import CodeQualityAgent
from src.instrumentation.logger import setup_logging
from src.mcp.github_client import build_github_mcp_client

LOG_FILE = "results/logs/test-base-agent-001.jsonl"
REPO = "pallets/flask"
RUN_ID = "test-base-agent-001"


async def main() -> None:
    token = os.environ["GITHUB_TOKEN"]
    api_key = os.environ["ANTHROPIC_API_KEY"]
    binary = Path(__file__).parent.parent / "github-mcp-server"

    setup_logging(log_file=LOG_FILE)

    print("=" * 60)
    print("TEST BASE AGENT — CodeQuality sur pallets/flask")
    print(f"Log file : {LOG_FILE}")
    print("=" * 60)

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=4096,
    )
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)
    agent = CodeQualityAgent(model=model, mcp_client=mcp_client)

    print(f"\nLancement de l'analyse sur {REPO} ...")
    report = await agent.analyze(repo=REPO, run_id=RUN_ID, architecture="supervisor")

    # ── Affichage du rapport ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAPPORT D'AUDIT")
    print("=" * 60)
    print(f"Agent     : {report.agent_name}")
    print(f"Repository: {report.repository}")
    print(f"Score     : {report.score}/20")
    print(f"Findings  : {len(report.findings)}")
    print(f"Raw data  : {len(report.raw_data)} appels MCP capturés")

    print("\nFindings :")
    for i, f in enumerate(report.findings, 1):
        print(f"  {i}. [{f.severity.upper()}] {f.category}: {f.description[:80]}...")

    # ── Vérifications minimales ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VÉRIFICATIONS")
    print("=" * 60)

    assert report.agent_name == "code_quality", f"agent_name incorrect: {report.agent_name}"
    print("  ✓ agent_name == 'code_quality'")

    assert report.repository == REPO, f"repository incorrect: {report.repository}"
    print(f"  ✓ repository == '{REPO}'")

    assert 0 <= report.score <= 20, f"score hors limites: {report.score}"
    print(f"  ✓ score dans [0, 20] : {report.score}")

    assert len(report.findings) > 0, "Aucun finding — le LLM n'a rien détecté"
    print(f"  ✓ {len(report.findings)} finding(s) présents")

    assert len(report.raw_data) > 0, "raw_data vide — aucun appel MCP loggé dans les messages"
    print(f"  ✓ raw_data non vide ({len(report.raw_data)} entrées)")

    # ── Vérification du fichier JSONL ─────────────────────────────────────────
    log_path = Path(__file__).parent.parent / LOG_FILE
    assert log_path.exists(), f"Fichier JSONL introuvable : {log_path}"

    mcp_entries = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "mcp_call":
                mcp_entries.append(entry)

    assert len(mcp_entries) > 0, "Aucune entrée mcp_call dans le JSONL"
    print(f"  ✓ {len(mcp_entries)} appels MCP loggés dans {LOG_FILE}")

    tools_used = {e["mcp_tool"] for e in mcp_entries}
    print(f"  ✓ Outils utilisés : {sorted(tools_used)}")

    # Vérifier que tous les champs requis sont présents
    required = [
        "run_id", "architecture", "repository", "agent_name",
        "mcp_tool", "mcp_params_hash", "response_size_bytes",
        "timestamp_start", "duration_ms", "success",
    ]
    for entry in mcp_entries:
        missing = [f for f in required if f not in entry]
        assert not missing, f"Champs manquants dans mcp_call: {missing}"
    print(f"  ✓ Tous les champs requis présents dans les {len(mcp_entries)} entrées")

    print("\n" + "=" * 60)
    print("Étape 4 validée. BaseAuditAgent opérationnel.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
