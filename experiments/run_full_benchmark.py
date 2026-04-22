"""
Étape 12 — Benchmark complet : 10 repos × 3 architectures × 3 runs = 90 audits.

Fonctionnalités :
  - Idempotence : skip si results/logs/{arch}-{owner}-{repo}-run{N}.jsonl existe
  - Budget cap : arrêt gracieux si coût cumulé > settings.max_budget_usd
  - Timeout 600s par audit
  - Gestion d'erreurs : .error à la place du JSONL, benchmark continue
  - Résumé final → results/benchmark_summary.json

Usage :
    uv run python experiments/run_full_benchmark.py
    uv run python experiments/run_full_benchmark.py --dry-run    # liste les audits
    uv run python experiments/run_full_benchmark.py --repos flask django  # subset
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_anthropic import ChatAnthropic

from src.architectures.decentralized import DecentralizedOrchestrator
from src.architectures.hierarchical import HierarchicalOrchestrator
from src.architectures.supervisor import FinalAuditReport, SupervisorOrchestrator
from src.instrumentation.logger import setup_logging
from src.instrumentation.metrics import RunMetrics, compute_metrics
from src.instrumentation.redundancy_detector import analyze_redundancy
from src.evaluation.grid_scorer import GridScorer
from src.mcp.github_client import build_github_mcp_client

# ── Constantes ────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = RESULTS_DIR / "logs"
REPORTS_DIR = RESULTS_DIR / "reports"

ARCHITECTURES   = ["supervisor", "hierarchical", "decentralized"]
N_RUNS          = 3
AUDIT_TIMEOUT_S = 900   # 15 min — Flask baseline = 608s, larger repos may exceed 10 min


def p(msg: str = "") -> None:
    """Print with immediate flush for piped/redirected output."""
    print(msg, flush=True)


# ── Structures de données ──────────────────────────────────────────────────────


@dataclass
class AuditTask:
    repo_owner:   str
    repo_name:    str
    category:     str
    architecture: str
    run_n:        int

    @property
    def repo(self) -> str:
        return f"{self.repo_owner}/{self.repo_name}"

    @property
    def run_id(self) -> str:
        return f"{self.architecture}-{self.repo_owner}-{self.repo_name}-run{self.run_n}"

    @property
    def log_file(self) -> Path:
        return LOGS_DIR / f"{self.run_id}.jsonl"

    @property
    def report_file(self) -> Path:
        return REPORTS_DIR / f"{self.run_id}.json"

    @property
    def error_file(self) -> Path:
        return LOGS_DIR / f"{self.run_id}.error"


@dataclass
class AuditResult:
    task:      AuditTask
    success:   bool
    metrics:   RunMetrics | None = None
    score:     float | None      = None
    duration:  float             = 0.0
    cost:      float             = 0.0
    error_msg: str               = ""


@dataclass
class BenchmarkState:
    total:      int
    done:       int   = 0
    skipped:    int   = 0
    failed:     int   = 0
    cumul_cost: float = 0.0
    start_time: float = field(default_factory=time.perf_counter)
    results:    list[AuditResult] = field(default_factory=list)

    @property
    def elapsed_min(self) -> float:
        return (time.perf_counter() - self.start_time) / 60


# ── Chargement des repos ───────────────────────────────────────────────────────


def load_repos(repos_yaml: Path) -> list[dict]:
    with open(repos_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["repositories"]


def build_tasks(repos: list[dict], filter_repos: list[str] | None = None) -> list[AuditTask]:
    tasks: list[AuditTask] = []
    for repo in repos:
        name = repo["name"]
        if filter_repos and name not in filter_repos:
            continue
        for arch in ARCHITECTURES:
            for run_n in range(1, N_RUNS + 1):
                tasks.append(AuditTask(
                    repo_owner=repo["owner"],
                    repo_name=name,
                    category=repo.get("category", "unknown"),
                    architecture=arch,
                    run_n=run_n,
                ))
    return tasks


# ── Exécution d'un audit ──────────────────────────────────────────────────────


def _build_orchestrator(architecture: str, model: ChatAnthropic, token: str, binary: Path):
    mcp_client = build_github_mcp_client(token=token, binary_path=binary)
    if architecture == "supervisor":
        return SupervisorOrchestrator(model=model, mcp_client=mcp_client)
    elif architecture == "hierarchical":
        return HierarchicalOrchestrator(model=model, mcp_client=mcp_client)
    elif architecture == "decentralized":
        return DecentralizedOrchestrator(model=model, mcp_client=mcp_client)
    raise ValueError(f"Unknown architecture: {architecture}")


async def run_audit(
    task: AuditTask,
    model: ChatAnthropic,
    token: str,
    binary: Path,
) -> AuditResult:
    """Lance un audit unique avec timeout. Retourne toujours (jamais d'exception)."""
    # Supprimer l'ancien log partiel (retry après timeout/erreur précédente)
    if task.log_file.exists():
        task.log_file.unlink()
    setup_logging(log_file=str(task.log_file))

    wall_start = time.perf_counter()
    try:
        orch = _build_orchestrator(task.architecture, model, token, binary)
        report: FinalAuditReport = await asyncio.wait_for(
            orch.run(repo=task.repo, run_id=task.run_id),
            timeout=AUDIT_TIMEOUT_S,
        )

        wall_time = time.perf_counter() - wall_start
        metrics   = compute_metrics(task.log_file, wall_clock_duration_seconds=wall_time)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        task.report_file.write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )

        return AuditResult(
            task=task,
            success=True,
            metrics=metrics,
            score=report.global_score,
            duration=wall_time,
            cost=metrics.total_cost_usd,
        )

    except asyncio.TimeoutError:
        wall_time = time.perf_counter() - wall_start
        msg = f"TIMEOUT après {AUDIT_TIMEOUT_S}s"
        _write_error(task, msg)
        # Supprimer le JSONL partiel pour que la prochaine tentative repart propre
        if task.log_file.exists():
            task.log_file.unlink()
        return AuditResult(task=task, success=False, duration=wall_time, error_msg=msg)

    except Exception as exc:
        wall_time = time.perf_counter() - wall_start
        msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _write_error(task, msg)
        if task.log_file.exists():
            task.log_file.unlink()
        return AuditResult(task=task, success=False, duration=wall_time, error_msg=str(exc))


def _write_error(task: AuditTask, msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    task.error_file.write_text(
        json.dumps({
            "run_id":       task.run_id,
            "architecture": task.architecture,
            "repository":   task.repo,
            "error":        msg,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2),
        encoding="utf-8",
    )


# ── Affichage progression ─────────────────────────────────────────────────────


def print_progress(idx: int, total: int, result: AuditResult, state: BenchmarkState) -> None:
    task  = result.task
    icon  = "✓" if result.success else "✗"
    score = f"{result.score:.1f}" if result.score is not None else " err"
    cost  = f"${result.cost:.3f}" if result.cost else "  err"
    dur   = f"{result.duration:.0f}s"

    p(
        f"[{idx:>3}/{total}] {icon} {task.architecture:<14} | "
        f"{task.repo:<28} | run {task.run_n} | "
        f"{cost:>7} | {dur:>5} | score {score}"
    )
    p(
        f"         Cumul : ${state.cumul_cost:.4f} | "
        f"écoulé : {state.elapsed_min:.0f}min | "
        f"réussis : {state.done} | échecs : {state.failed}"
    )


# ── Résumé final ──────────────────────────────────────────────────────────────


def compute_summary(state: BenchmarkState) -> dict:
    import statistics

    summary: dict = {
        "total_audits":       state.total,
        "completed":          state.done + state.failed,
        "skipped":            state.skipped,
        "succeeded":          state.done,
        "failed":             state.failed,
        "cumul_cost_usd":     state.cumul_cost,
        "total_duration_min": state.elapsed_min,
        "by_architecture":    {},
    }

    for arch in ARCHITECTURES:
        arch_results = [r for r in state.results if r.task.architecture == arch and r.success]
        if not arch_results:
            summary["by_architecture"][arch] = {"n": 0}
            continue

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {"mean": None, "std": None, "median": None}
            return {
                "mean":   round(statistics.mean(vals), 4),
                "std":    round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                "median": round(statistics.median(vals), 4),
            }

        summary["by_architecture"][arch] = {
            "n_succeeded":    len(arch_results),
            "n_failed":       sum(1 for r in state.results if r.task.architecture == arch and not r.success),
            "mcp_calls":      _stats([r.metrics.total_mcp_calls for r in arch_results if r.metrics]),
            "redundant_calls":_stats([r.metrics.redundant_mcp_calls for r in arch_results if r.metrics]),
            "redundancy_rate":_stats([r.metrics.redundancy_rate for r in arch_results if r.metrics]),
            "cost_usd":       _stats([r.metrics.total_cost_usd for r in arch_results if r.metrics]),
            "overhead_usd":   _stats([r.metrics.overhead_orchestration_cost_usd for r in arch_results if r.metrics]),
            "duration_s":     _stats([r.metrics.wall_clock_duration_seconds for r in arch_results if r.metrics]),
            "global_score":   _stats([r.score for r in arch_results if r.score is not None]),
        }

    return summary


def print_final_summary(summary: dict) -> None:
    p()
    p("=" * 80)
    p("  RÉSUMÉ FINAL DU BENCHMARK")
    p("=" * 80)
    p(f"  Audits total   : {summary['total_audits']}")
    p(f"  Réussis        : {summary['succeeded']}")
    p(f"  Échoués        : {summary['failed']}")
    p(f"  Ignorés (skip) : {summary['skipped']}")
    p(f"  Coût total     : ${summary['cumul_cost_usd']:.4f}")
    p(f"  Durée totale   : {summary['total_duration_min']:.0f} min")
    p()

    metrics_labels = [
        ("mcp_calls",       "MCP calls"),
        ("redundant_calls", "Redondants"),
        ("redundancy_rate", "Taux redondance"),
        ("cost_usd",        "Coût ($)"),
        ("overhead_usd",    "Overhead ($)"),
        ("duration_s",      "Durée (s)"),
        ("global_score",    "Score /20"),
    ]
    col_w = 28
    p(f"  {'Métrique':<22}" + "".join(f"{a:<{col_w}}" for a in ARCHITECTURES))
    p("  " + "-" * (22 + col_w * len(ARCHITECTURES)))

    for key, label in metrics_labels:
        row = f"  {label:<22}"
        for arch in ARCHITECTURES:
            arch_data = summary["by_architecture"].get(arch, {})
            stat = arch_data.get(key, {})
            if not stat or stat.get("mean") is None:
                row += f"{'N/A':<{col_w}}"
            else:
                val = f"{stat['mean']:.3f} ± {stat['std']:.3f}"
                row += f"{val:<{col_w}}"
        p(row)

    p("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    dry_run = "--dry-run" in sys.argv

    filter_repos: list[str] | None = None
    if "--repos" in sys.argv:
        idx = sys.argv.index("--repos")
        filter_repos = sys.argv[idx + 1:]

    token   = os.environ["GITHUB_TOKEN"]
    api_key = os.environ["ANTHROPIC_API_KEY"]
    binary  = ROOT / "github-mcp-server"

    try:
        from src.config import get_settings
        max_budget = get_settings().max_budget_usd
    except Exception:
        max_budget = 60.0

    repos = load_repos(ROOT / "data" / "repos.yaml")
    tasks = build_tasks(repos, filter_repos)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    p("=" * 70)
    p(f"  BENCHMARK GitHubAuditBench — {len(tasks)} audits planifiés")
    p(f"  Budget max : ${max_budget:.0f} | Timeout par audit : {AUDIT_TIMEOUT_S}s")
    p("=" * 70)

    if dry_run:
        p()
        p("[DRY RUN] Audits planifiés :")
        for i, t in enumerate(tasks, 1):
            exists = "✓ SKIP" if t.report_file.exists() else "  TODO"
            p(f"  [{i:>3}] {exists} {t.architecture:<14} {t.repo:<28} run{t.run_n}")
        return

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=4096,
    )

    state = BenchmarkState(total=len(tasks))
    idx   = 0

    for task in tasks:
        idx += 1

        # ── Idempotence : skip seulement si le rapport JSON final existe ─────
        # (le JSONL peut exister partiellement en cas de timeout/erreur)
        if task.report_file.exists():
            p(f"[{idx:>3}/{len(tasks)}] ⏭ SKIP  {task.architecture:<14} | {task.repo} | run{task.run_n}")
            state.skipped += 1
            try:
                m = compute_metrics(task.log_file)
                state.cumul_cost += m.total_cost_usd
            except Exception:
                pass
            continue

        # ── Budget cap ────────────────────────────────────────────────────────
        if state.cumul_cost >= max_budget:
            p(f"\n⚠ BUDGET ATTEINT : ${state.cumul_cost:.4f} / ${max_budget:.0f}")
            p(f"  Arrêt après {idx - 1}/{len(tasks)} audits.")
            break

        # ── Lancer l'audit ────────────────────────────────────────────────────
        p(f"[{idx:>3}/{len(tasks)}] → {task.architecture:<14} | {task.repo:<28} | run{task.run_n} …")
        result = await run_audit(task, model, token, binary)

        if result.success:
            state.done       += 1
            state.cumul_cost += result.cost
        else:
            state.failed += 1
            p(f"         ✗ ERREUR : {result.error_msg[:120]}")

        state.results.append(result)
        print_progress(idx, len(tasks), result, state)
        p()

    # ── Résumé final ──────────────────────────────────────────────────────────
    summary = compute_summary(state)
    print_final_summary(summary)

    out = RESULTS_DIR / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    p(f"\n  Résumé sauvegardé : {out}")


if __name__ == "__main__":
    asyncio.run(main())
