"""
Tableau comparatif des 3 architectures LangGraph sur pallets/flask.

Compare :
  - Supervisor    : LLM orchestrateur séquentiel (étape 6)
  - Hierarchical  : 2 sous-graphes + 3 superviseurs LLM (étape 7)
  - Decentralized : Command handoffs, 0 superviseur LLM (étape 8)

Métriques mesurées (tableau central du mémoire) :
  MCP calls totaux | MCP calls redondants | Findings | Score global
  Coût total ($)   | Overhead orchestration ($) | Durée (s)
  LLM calls agents | LLM calls superviseurs

Usage :
    uv run python experiments/test_all_architectures.py

Résultats écrits dans :
    results/logs/comparison-supervisor-001.jsonl
    results/logs/comparison-hierarchical-001.jsonl
    results/logs/comparison-decentralized-001.jsonl
    results/comparison_table.txt
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_anthropic import ChatAnthropic

from src.architectures.decentralized import DecentralizedOrchestrator
from src.architectures.hierarchical import HierarchicalOrchestrator
from src.architectures.supervisor import FinalAuditReport, SupervisorOrchestrator
from src.instrumentation.logger import setup_logging
from src.mcp.github_client import build_github_mcp_client

REPO   = "pallets/flask"
RUN_ID_SUP  = "comparison-supervisor-001"
RUN_ID_HIE  = "comparison-hierarchical-001"
RUN_ID_DEC  = "comparison-decentralized-001"

LOG_SUP = f"results/logs/{RUN_ID_SUP}.jsonl"
LOG_HIE = f"results/logs/{RUN_ID_HIE}.jsonl"
LOG_DEC = f"results/logs/{RUN_ID_DEC}.jsonl"

# Baseline étape 5 (5 agents séquentiels indépendants)
BASELINE = {
    "name": "Baseline (seq.)",
    "mcp_calls":    42,
    "mcp_redundant": 0,
    "findings":     41,
    "score":        14.6,
    "cost":         1.56,
    "overhead_cost": 0.0,
    "duration":     579.0,
    "llm_agents":   5,
    "llm_overhead": 0,
}

INPUT_PRICE  = 3.00 / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000
CACHE_READ   = 0.30  / 1_000_000
CACHE_CREATE = 3.75  / 1_000_000

# Noms d'agents considérés comme overhead d'orchestration
_OVERHEAD_AGENT_NAMES = {
    "supervisor", "synthesizer",
    "tech_supervisor", "community_supervisor", "top_supervisor",
    "tech_synthesizer", "community_synthesizer",
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_log(log_file: str) -> list[dict]:
    path = Path(__file__).parent.parent / log_file
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def compute_cost(llm_entries: list[dict]) -> float:
    total = 0.0
    for e in llm_entries:
        cc  = e.get("cache_creation_tokens", 0)
        cr  = e.get("cache_read_tokens", 0)
        inp = e.get("input_tokens", 0)
        out = e.get("output_tokens", 0)
        total += (inp - cc - cr) * INPUT_PRICE + cc * CACHE_CREATE + cr * CACHE_READ + out * OUTPUT_PRICE
    return total


def count_redundant_mcp(mcp_entries: list[dict]) -> int:
    """Compte les appels MCP dont les paramètres ont déjà été vus (hash identique)."""
    seen: set[str] = set()
    redundant = 0
    for e in mcp_entries:
        h = e.get("mcp_params_hash", "")
        if h and h in seen:
            redundant += 1
        elif h:
            seen.add(h)
    return redundant


def parse_metrics(log_file: str, report: FinalAuditReport, wall_time: float) -> dict:
    entries     = load_log(log_file)
    mcp_entries = [e for e in entries if e.get("event") == "mcp_call"]
    llm_entries = [e for e in entries if e.get("event") == "llm_call"]

    agent_llm    = [e for e in llm_entries if e.get("agent") not in _OVERHEAD_AGENT_NAMES]
    overhead_llm = [e for e in llm_entries if e.get("agent") in _OVERHEAD_AGENT_NAMES]

    total_findings = sum(len(r.findings) for r in report.agent_reports.values())

    return {
        "mcp_calls":     len(mcp_entries),
        "mcp_redundant": count_redundant_mcp(mcp_entries),
        "findings":      total_findings,
        "score":         report.global_score,
        "cost":          compute_cost(llm_entries),
        "overhead_cost": compute_cost(overhead_llm),
        "duration":      wall_time,
        "llm_agents":    len(agent_llm),
        "llm_overhead":  len(overhead_llm),
        "total_input":   sum(e.get("input_tokens", 0) for e in llm_entries),
        "total_output":  sum(e.get("output_tokens", 0) for e in llm_entries),
        "cache_read":    sum(e.get("cache_read_tokens", 0) for e in llm_entries),
    }


# ── Runner ─────────────────────────────────────────────────────────────────────


async def run_architecture(name: str, orchestrator, run_id: str, log_file: str, repo: str):
    setup_logging(log_file=log_file)
    print(f"\n{'='*70}")
    print(f"  {name.upper()} — {repo}")
    print(f"  Log : {log_file}")
    print(f"{'='*70}")

    wall_start = time.perf_counter()
    report: FinalAuditReport = await orchestrator.run(repo=repo, run_id=run_id)
    wall_time = time.perf_counter() - wall_start

    metrics = parse_metrics(log_file, report, wall_time)
    metrics["name"] = name
    metrics["report"] = report

    print(f"  Score : {report.global_score}/20  |  Findings : {metrics['findings']}")
    print(f"  MCP   : {metrics['mcp_calls']} total, {metrics['mcp_redundant']} redondants")
    print(f"  Coût  : ${metrics['cost']:.4f} (overhead: ${metrics['overhead_cost']:.4f})")
    print(f"  Durée : {wall_time:.0f}s")

    return metrics


# ── Tableau ────────────────────────────────────────────────────────────────────


def print_comparison_table(results: list[dict]) -> str:
    """Affiche et retourne le tableau comparatif."""
    all_rows = [BASELINE] + results

    col_w = 22
    header_cols = ["Métrique"] + [r["name"] for r in all_rows]
    sep = "+" + "+".join("-" * (col_w + 2) for _ in header_cols) + "+"

    def row(label, values, fmt=None):
        cells = [f" {label:<{col_w}} "]
        for v in values:
            if fmt:
                cells.append(f" {fmt(v):>{col_w}} ")
            else:
                cells.append(f" {str(v):>{col_w}} ")
        return "|" + "|".join(cells) + "|"

    def delta(base, val):
        if base == 0:
            return "N/A"
        d = val - base
        pct = d / base * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{pct:.1f}%"

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  TABLEAU COMPARATIF — 3 ARCHITECTURES LANGGRAPH")
    lines.append(f"  Repo : {REPO}  |  Date : 2026-04-16")
    lines.append("=" * 80)
    lines.append("")

    # En-têtes
    lines.append(sep)
    lines.append(row("Métrique", [r["name"] for r in all_rows]))
    lines.append(sep)

    def add_row(label, key, fmt_fn=None, base_key=None):
        vals = [r.get(key, "?") for r in all_rows]
        if fmt_fn:
            vals_str = [fmt_fn(v) for v in vals]
        else:
            vals_str = [str(v) for v in vals]
        lines.append(row(label, vals_str))

    add_row("MCP calls totaux",      "mcp_calls")
    add_row("MCP calls redondants",  "mcp_redundant")
    add_row("Findings totaux",       "findings")
    add_row("Score global (/20)",    "score",        lambda v: f"{v:.1f}" if isinstance(v, float) else str(v))
    add_row("Coût total ($)",        "cost",         lambda v: f"${v:.4f}" if isinstance(v, float) else str(v))
    add_row("Overhead orchestr. ($)","overhead_cost",lambda v: f"${v:.4f}" if isinstance(v, float) else str(v))
    add_row("Durée (s)",             "duration",     lambda v: f"{v:.0f}" if isinstance(v, float) else str(v))
    add_row("LLM calls agents",      "llm_agents")
    add_row("LLM calls orchestr.",   "llm_overhead")

    lines.append(sep)

    # Deltas vs baseline
    lines.append("")
    lines.append("  Deltas par rapport à la baseline (étape 5 — séquentiel indépendant)")
    lines.append("")
    base = BASELINE
    for r in results:
        lines.append(f"  [{r['name']}]")
        for key, label in [
            ("mcp_calls",    "MCP calls"),
            ("findings",     "Findings"),
            ("cost",         "Coût"),
            ("duration",     "Durée"),
            ("llm_overhead", "LLM overhead"),
        ]:
            bv = base.get(key, 0)
            rv = r.get(key, 0)
            rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
            lines.append(f"    {label:<22} baseline={bv}  val={rv_str}  delta={delta(bv if bv else 1, rv)}")
        lines.append("")

    lines.append("=" * 80)

    table_str = "\n".join(lines)
    print(table_str)
    return table_str


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    token   = os.environ["GITHUB_TOKEN"]
    api_key = os.environ["ANTHROPIC_API_KEY"]
    binary  = Path(__file__).parent.parent / "github-mcp-server"

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=4096,
    )

    results: list[dict] = []

    # ── Supervisor ────────────────────────────────────────────────────────────
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)
    orchestrator = SupervisorOrchestrator(model=model, mcp_client=mcp_client)
    metrics_sup = await run_architecture(
        "Supervisor", orchestrator, RUN_ID_SUP, LOG_SUP, REPO
    )
    results.append(metrics_sup)

    # ── Hierarchical ──────────────────────────────────────────────────────────
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)
    orchestrator = HierarchicalOrchestrator(model=model, mcp_client=mcp_client)
    metrics_hie = await run_architecture(
        "Hierarchical", orchestrator, RUN_ID_HIE, LOG_HIE, REPO
    )
    results.append(metrics_hie)

    # ── Decentralized ─────────────────────────────────────────────────────────
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)
    orchestrator = DecentralizedOrchestrator(model=model, mcp_client=mcp_client)
    metrics_dec = await run_architecture(
        "Decentralized", orchestrator, RUN_ID_DEC, LOG_DEC, REPO
    )
    results.append(metrics_dec)

    # ── Tableau comparatif ────────────────────────────────────────────────────
    table_str = print_comparison_table(results)

    # Sauvegarde du tableau
    out_path = Path(__file__).parent.parent / "results" / "comparison_table.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_str, encoding="utf-8")
    print(f"\nTableau sauvegardé : {out_path}")

    # ── Assertions minimales ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VÉRIFICATIONS")
    print("=" * 70)

    for m in results:
        name = m["name"]
        assert len(m["report"].agent_reports) == 5, f"{name}: attendu 5 rapports"
        print(f"  ✓ {name}: 5 rapports agents")
        assert m["findings"] >= 10, f"{name}: trop peu de findings ({m['findings']})"
        print(f"  ✓ {name}: {m['findings']} findings")
        assert 0 <= m["score"] <= 20, f"{name}: score hors bornes"
        print(f"  ✓ {name}: score={m['score']}/20")

    # Decentralized : 0 LLM overhead (sauf synthesizer = 1 appel)
    assert metrics_dec["llm_overhead"] <= 1, (
        f"Decentralized: overhead LLM inattendu = {metrics_dec['llm_overhead']} "
        f"(attendu ≤ 1 pour le seul synthesizer)"
    )
    print(f"  ✓ Decentralized: overhead LLM = {metrics_dec['llm_overhead']} (synthesizer uniquement)")

    print("\n" + "=" * 70)
    print("Étapes 7+8 validées. Les 3 architectures sont opérationnelles.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
