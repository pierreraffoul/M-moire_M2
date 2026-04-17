"""
Test séquentiel des 5 agents sur pallets/flask (étape 5).

Vérifie que :
- Les 5 agents produisent chacun un AgentReport valide
- SecurityAgent gère gracieusement les 403 sur les alertes (sans crasher)
- Chaque rapport contient au moins 2 findings
- Le récap final affiche : scores, findings, appels MCP, coût total

Usage :
    uv run python experiments/test_all_agents.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_anthropic import ChatAnthropic

from src.agents.code_quality import CodeQualityAgent
from src.agents.community import CommunityAgent
from src.agents.documentation import DocumentationAgent
from src.agents.license import LicenseAgent
from src.agents.security import SecurityAgent
from src.instrumentation.logger import setup_logging
from src.mcp.github_client import build_github_mcp_client

LOG_FILE = "results/logs/test-all-agents-001.jsonl"
REPO = "pallets/flask"
RUN_ID = "test-all-agents-001"

# Tarifs claude-sonnet-4-5 ($/M tokens)
INPUT_PRICE  = 3.00 / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000
CACHE_READ   = 0.30 / 1_000_000
CACHE_CREATE = 3.75 / 1_000_000


def compute_cost(llm_entries: list[dict]) -> float:
    total = 0.0
    for e in llm_entries:
        details = e.get("input_token_details", {})
        cc = e.get("cache_creation_tokens", 0)
        cr = e.get("cache_read_tokens", 0)
        inp = e.get("input_tokens", 0)
        out = e.get("output_tokens", 0)
        total += (inp - cc - cr) * INPUT_PRICE + cc * CACHE_CREATE + cr * CACHE_READ + out * OUTPUT_PRICE
    return total


async def main() -> None:
    token = os.environ["GITHUB_TOKEN"]
    api_key = os.environ["ANTHROPIC_API_KEY"]
    binary = Path(__file__).parent.parent / "github-mcp-server"

    setup_logging(log_file=LOG_FILE)

    print("=" * 70)
    print("TEST 5 AGENTS — pallets/flask")
    print(f"Log file : {LOG_FILE}")
    print("=" * 70)

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=4096,
    )
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)

    agents = [
        CodeQualityAgent(model=model, mcp_client=mcp_client),
        CommunityAgent(model=model, mcp_client=mcp_client),
        SecurityAgent(model=model, mcp_client=mcp_client),
        DocumentationAgent(model=model, mcp_client=mcp_client),
        LicenseAgent(model=model, mcp_client=mcp_client),
    ]

    results = []
    wall_start = time.perf_counter()

    for agent in agents:
        print(f"\n[{agent.name.upper()}] Analyse en cours...")
        t0 = time.perf_counter()
        report = await agent.analyze(repo=REPO, run_id=RUN_ID, architecture="supervisor")
        elapsed = time.perf_counter() - t0
        results.append((agent.name, report, elapsed))
        print(f"  → score={report.score}/20  findings={len(report.findings)}  mcp_calls={len(report.raw_data)}  {elapsed:.1f}s")

    wall_total = time.perf_counter() - wall_start

    # ── Lecture du JSONL pour coût ─────────────────────────────────────────────
    log_path = Path(__file__).parent.parent / LOG_FILE
    all_entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    mcp_entries = [e for e in all_entries if e.get("event") == "mcp_call"]
    llm_entries = [e for e in all_entries if e.get("event") == "llm_call"]

    total_cost = compute_cost(llm_entries)
    total_input  = sum(e.get("input_tokens", 0) for e in llm_entries)
    total_output = sum(e.get("output_tokens", 0) for e in llm_entries)
    total_cr     = sum(e.get("cache_read_tokens", 0) for e in llm_entries)

    # ── Récap ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉCAP — 5 AGENTS sur pallets/flask")
    print("=" * 70)
    print(f"{'Agent':<18} {'Score':>5}  {'Findings':>8}  {'MCP calls':>9}  {'Durée':>7}")
    print("-" * 70)
    for name, report, elapsed in results:
        print(f"  {name:<16} {report.score:>5.1f}  {len(report.findings):>8}  {len(report.raw_data):>9}  {elapsed:>6.1f}s")
    print("-" * 70)
    total_findings = sum(len(r.findings) for _, r, _ in results)
    total_mcp = sum(len(r.raw_data) for _, r, _ in results)
    print(f"  {'TOTAL':<16} {'':>5}  {total_findings:>8}  {total_mcp:>9}  {wall_total:>6.1f}s")

    print(f"\nTokens : input={total_input:,}  output={total_output:,}  cache_read={total_cr:,}")
    print(f"Coût total     : ${total_cost:.4f}")
    print(f"Coût moyen/agent : ${total_cost/5:.4f}")

    # ── Vérifications ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VÉRIFICATIONS")
    print("=" * 70)

    all_ok = True

    for name, report, _ in results:
        ok = True
        if report.agent_name != name:
            print(f"  ✗ {name}: agent_name incorrect ({report.agent_name})")
            ok = False
        if not (0 <= report.score <= 20):
            print(f"  ✗ {name}: score hors limites ({report.score})")
            ok = False
        if len(report.findings) < 2:
            print(f"  ✗ {name}: moins de 2 findings ({len(report.findings)})")
            ok = False
        if ok:
            print(f"  ✓ {name}: score={report.score}/20, {len(report.findings)} findings, {len(report.raw_data)} MCP calls")
        else:
            all_ok = False

    # Vérification spéciale SecurityAgent : doit avoir géré les 403
    security_report = next(r for n, r, _ in results if n == "security")
    security_mcp = [e for e in mcp_entries if e.get("agent_name") == "security"]
    alert_tools = ["list_dependabot_alerts", "list_secret_scanning_alerts", "list_code_scanning_alerts"]
    alert_attempts = [e for e in security_mcp if e.get("mcp_tool") in alert_tools]
    if alert_attempts:
        failed = [e for e in alert_attempts if not e.get("success", True)]
        print(f"  ✓ security: {len(alert_attempts)} appels d'alertes tentés, {len(failed)} échecs gracieusement gérés")
    else:
        print(f"  ℹ security: aucun appel d'alertes (LLM a directement utilisé les signaux indirects)")

    assert all_ok, "Certaines vérifications ont échoué"
    print("\n" + "=" * 70)
    print("Étape 5 validée. Les 5 agents sont opérationnels.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
