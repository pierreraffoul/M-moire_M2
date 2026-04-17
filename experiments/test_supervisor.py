"""
Test de l'architecture Supervisor sur pallets/flask (étape 6).

Compare avec la baseline étape 5 (5 agents séquentiels indépendants) :
  Baseline : 42 appels MCP, 41 findings, $1.56, ~579s

Usage :
    uv run python experiments/test_supervisor.py
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

from src.architectures.supervisor import FinalAuditReport, SupervisorOrchestrator
from src.instrumentation.logger import setup_logging
from src.mcp.github_client import build_github_mcp_client

LOG_FILE = "results/logs/test-supervisor-001.jsonl"
REPO = "pallets/flask"
RUN_ID = "test-supervisor-001"

# Baseline étape 5
BASELINE_MCP_CALLS = 42
BASELINE_FINDINGS  = 41
BASELINE_COST      = 1.56
BASELINE_DURATION  = 579.0

INPUT_PRICE  = 3.00 / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000
CACHE_READ   = 0.30 / 1_000_000
CACHE_CREATE = 3.75 / 1_000_000


def compute_cost(llm_entries: list[dict]) -> float:
    total = 0.0
    for e in llm_entries:
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
    print("TEST SUPERVISOR — pallets/flask")
    print(f"Log file : {LOG_FILE}")
    print("=" * 70)

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=4096,
    )
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)
    orchestrator = SupervisorOrchestrator(model=model, mcp_client=mcp_client)

    print(f"\nLancement du supervisor sur {REPO} ...")
    wall_start = time.perf_counter()
    report: FinalAuditReport = await orchestrator.run(repo=REPO, run_id=RUN_ID)
    wall_total = time.perf_counter() - wall_start

    # ── Lecture JSONL ─────────────────────────────────────────────────────────
    log_path = Path(__file__).parent.parent / LOG_FILE
    all_entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    mcp_entries = [e for e in all_entries if e.get("event") == "mcp_call"]
    llm_entries = [e for e in all_entries if e.get("event") == "llm_call"]

    total_cost   = compute_cost(llm_entries)
    total_input  = sum(e.get("input_tokens", 0) for e in llm_entries)
    total_output = sum(e.get("output_tokens", 0) for e in llm_entries)
    total_cr     = sum(e.get("cache_read_tokens", 0) for e in llm_entries)

    # Séparer les appels LLM : agents vs supervisor/synthesizer
    agent_llm   = [e for e in llm_entries if e.get("agent") not in ("supervisor", "synthesizer")]
    overhead_llm = [e for e in llm_entries if e.get("agent") in ("supervisor", "synthesizer")]
    overhead_cost = compute_cost(overhead_llm)

    total_findings = sum(len(r.findings) for r in report.agent_reports.values())

    # ── Rapport ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RAPPORT SUPERVISOR")
    print("=" * 70)
    print(f"Score global    : {report.global_score}/20")
    print(f"Summary         : {report.summary[:120]}...")
    print("\nTop recommandations :")
    for i, rec in enumerate(report.top_recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\nAgents exécutés : {list(report.agent_reports.keys())}")
    print(f"Ordre réel      : {list(report.agent_reports.keys())} (supervisor iterations={report.supervisor_iterations})")
    print()

    for name, r in report.agent_reports.items():
        print(f"  {name:<18} score={r.score}/20  findings={len(r.findings):<3} mcp={len(r.raw_data)}")

    # ── Comparaison avec baseline ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARAISON AVEC BASELINE (étape 5 — 5 agents indépendants)")
    print("=" * 70)
    print(f"{'Métrique':<30} {'Baseline':>12} {'Supervisor':>12} {'Delta':>10}")
    print("-" * 68)

    def delta(base, val, fmt="{:+.0f}"):
        d = val - base
        pct = d / base * 100 if base else 0
        return fmt.format(d) + f" ({pct:+.1f}%)"

    print(f"  {'Appels MCP totaux':<28} {BASELINE_MCP_CALLS:>12} {report.total_mcp_calls:>12} {delta(BASELINE_MCP_CALLS, report.total_mcp_calls):>10}")
    print(f"  {'Findings totaux':<28} {BASELINE_FINDINGS:>12} {total_findings:>12} {delta(BASELINE_FINDINGS, total_findings):>10}")
    print(f"  {'Durée (s)':<28} {BASELINE_DURATION:>12.0f} {wall_total:>12.0f} {delta(BASELINE_DURATION, wall_total):>10}")
    print(f"  {'Coût total ($)':<28} {BASELINE_COST:>12.4f} {total_cost:>12.4f} {delta(BASELINE_COST, total_cost, '{:+.4f}'):>10}")
    print(f"  {'Overhead orches. ($)':<28} {'N/A':>12} {overhead_cost:>12.4f}")
    print(f"  {'LLM calls agents':<28} {'—':>12} {len(agent_llm):>12}")
    print(f"  {'LLM calls supervisor':<28} {'—':>12} {len(overhead_llm):>12}")

    print(f"\nTokens total   : input={total_input:,}  output={total_output:,}  cache_read={total_cr:,}")

    # ── Vérifications ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VÉRIFICATIONS")
    print("=" * 70)

    assert len(report.agent_reports) == 5, f"Attendu 5 rapports agents, obtenu {len(report.agent_reports)}"
    print(f"  ✓ 5 rapports agents présents")

    assert 0 <= report.global_score <= 20
    print(f"  ✓ global_score={report.global_score} dans [0, 20]")

    assert total_findings >= 10, f"Trop peu de findings ({total_findings})"
    print(f"  ✓ {total_findings} findings totaux")

    assert len(overhead_llm) >= 2, "Pas d'appels LLM supervisor/synthesizer loggés"
    print(f"  ✓ {len(overhead_llm)} appels LLM overhead (supervisor={sum(1 for e in overhead_llm if e.get('agent')=='supervisor')}, synthesizer={sum(1 for e in overhead_llm if e.get('agent')=='synthesizer')})")

    print("\n" + "=" * 70)
    print("Étape 6 validée. Architecture Supervisor opérationnelle.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
