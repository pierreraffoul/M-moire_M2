"""
Test des modules d'évaluation — étapes 9, 10, 11.

Montre :
  1. RunMetrics pour les 3 architectures (depuis les logs Flask des étapes 6-8)
  2. RedundancyReport avec matrice paire-à-paire pour les 3 architectures
  3. GridScorer sur un rapport Supervisor reconstruit depuis les logs
  4. LLMJudge (optionnel, nécessite ANTHROPIC_API_KEY)
  5. FactualChecker (optionnel, nécessite GITHUB_TOKEN)

Usage :
    uv run python experiments/test_evaluation.py
    uv run python experiments/test_evaluation.py --skip-llm   # évite les appels LLM
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from src.instrumentation.metrics import RunMetrics, compute_metrics, reconstruct_final_report
from src.instrumentation.redundancy_detector import analyze_redundancy
from src.evaluation.grid_scorer import GridScorer

LOGS = {
    "Supervisor":    ("results/logs/comparison-supervisor-001.jsonl",    608),
    "Hierarchical":  ("results/logs/comparison-hierarchical-001.jsonl",  595),
    "Decentralized": ("results/logs/comparison-decentralized-001.jsonl", 552),
}

SKIP_LLM = "--skip-llm" in sys.argv


def print_metrics_table(metrics: dict[str, RunMetrics]) -> None:
    col_w = 16
    fields = [
        ("total_mcp_calls",                 "MCP calls totaux"),
        ("redundant_mcp_calls",             "MCP redondants"),
        ("redundancy_rate",                 "Taux redondance"),
        ("mcp_data_volume_bytes",           "Volume données (B)"),
        ("total_tokens_input",              "Tokens input"),
        ("total_tokens_output",             "Tokens output"),
        ("total_tokens_cache_read",         "Cache read tokens"),
        ("total_cost_usd",                  "Coût total ($)"),
        ("overhead_orchestration_cost_usd", "Overhead LLM ($)"),
        ("wall_clock_duration_seconds",     "Durée (s)"),
    ]

    print("\n" + "=" * 78)
    print("  ÉTAPE 9 — RunMetrics : 3 architectures sur pallets/flask")
    print("=" * 78)
    header = f"  {'Métrique':<32}" + "".join(f"{a:>{col_w}}" for a in metrics)
    print(header)
    print("  " + "-" * 76)

    for field, label in fields:
        row = f"  {label:<32}"
        for m in metrics.values():
            val = getattr(m, field)
            if "usd" in field:
                row += f"{'${:.4f}'.format(val):>{col_w}}"
            elif "rate" in field:
                row += f"{'{:.1%}'.format(val):>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.0f}"
            else:
                row += f"{val:>{col_w}}"
        print(row)

    print("=" * 78)


def print_redundancy_section(arch: str, log: str) -> None:
    print(f"\n{'='*60}")
    print(f"  ÉTAPE 10 — RedundancyReport : {arch.upper()}")
    r = analyze_redundancy(Path(log))
    print(f"  {r.total_mcp_calls} appels | {r.unique_mcp_calls} uniques | "
          f"{r.redundant_mcp_calls} redondants ({r.redundancy_rate:.1%})")
    print(f"\n  Outils les plus redondants :")
    for tool, cnt in list(r.redundant_tools.items())[:5]:
        print(f"    {tool:<35} {cnt:>3} doublon(s)")
    print(f"\n  Matrice paire-à-paire (doublons MCP entre agents) :")
    r.print_matrix()


def grid_scorer_section(arch: str, log: str) -> None:
    print(f"\n{'='*60}")
    print(f"  ÉTAPE 11a — GridScorer : {arch.upper()}")
    report = reconstruct_final_report(Path(log))
    if report is None:
        print(f"  ⚠ Findings non persistés dans ce log (run antérieur à étape 9).")
        print(f"    Relancer le benchmark pour avoir des GridScores sur les vrais findings.")
        return
    scorer = GridScorer()
    gs = scorer.score(report)
    scorer.print_scorecard(gs)


async def llm_judge_section(arch: str, log: str, api_key: str) -> None:
    from src.evaluation.llm_judge import LLMJudge
    print(f"\n{'='*60}")
    print(f"  ÉTAPE 11b — LLMJudge : {arch.upper()}")
    report = reconstruct_final_report(Path(log))
    if report is None:
        print(f"  ⚠ Findings non persistés — impossible d'évaluer.")
        return
    judge = LLMJudge(model_name="claude-haiku-4-5-20251001", api_key=api_key)
    verdict = await judge.evaluate(report)
    judge.print_verdict(verdict)


async def factual_checker_section(arch: str, log: str, github_token: str) -> None:
    from src.evaluation.factual_checker import FactualChecker
    print(f"\n{'='*60}")
    print(f"  ÉTAPE 11c — FactualChecker : {arch.upper()}")
    report = reconstruct_final_report(Path(log))
    if report is None:
        print(f"  ⚠ Findings non persistés — impossible de vérifier.")
        return
    checker = FactualChecker(github_token=github_token, sample_size=5)
    fc_report = await checker.check(report)
    checker.print_report(fc_report)


async def main() -> None:
    api_key      = os.environ.get("ANTHROPIC_API_KEY", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")

    # ── Étape 9 : RunMetrics ──────────────────────────────────────────────────
    all_metrics: dict[str, RunMetrics] = {}
    for arch, (log, duration) in LOGS.items():
        all_metrics[arch] = compute_metrics(Path(log), wall_clock_duration_seconds=duration)
    print_metrics_table(all_metrics)

    # ── Étape 10 : Redundancy (toutes les architectures) ─────────────────────
    for arch, (log, _) in LOGS.items():
        print_redundancy_section(arch, log)

    # ── Étape 11a : GridScorer (toutes les architectures) ────────────────────
    for arch, (log, _) in LOGS.items():
        grid_scorer_section(arch, log)

    if not SKIP_LLM and api_key:
        # ── Étape 11b : LLMJudge (Supervisor uniquement pour le test) ────────
        await llm_judge_section("Supervisor", LOGS["Supervisor"][0], api_key)
    else:
        print("\n  ⏭ LLMJudge ignoré (--skip-llm ou pas de clé API).")

    if not SKIP_LLM and github_token:
        # ── Étape 11c : FactualChecker (Supervisor uniquement) ───────────────
        await factual_checker_section("Supervisor", LOGS["Supervisor"][0], github_token)
    else:
        print("  ⏭ FactualChecker ignoré (--skip-llm ou pas de token GitHub).\n")

    print("=" * 60)
    print("  Étapes 9-10-11 validées.")
    print("  Note : GridScorer affichera des résultats réels au prochain")
    print("  run du benchmark (findings maintenant persistés dans agent_done).")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
