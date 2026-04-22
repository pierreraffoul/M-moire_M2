"""
Analyse post-benchmark — lit tous les résultats et produit les tableaux du mémoire.

Sorties :
  results/analysis/metrics_table.csv      — tableau central (moyennes ± std)
  results/analysis/mann_whitney.json      — tests statistiques paire-à-paire
  results/analysis/by_category.json       — résultats par catégorie de repo
  results/analysis/redundancy_matrix.json — matrices paire-à-paire agrégées
  results/analysis/summary_report.txt     — rapport lisible terminal/mémoire

Usage :
    uv run python experiments/analyze_benchmark.py
    uv run python experiments/analyze_benchmark.py --arch supervisor   # filtre
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.instrumentation.metrics import RunMetrics, compute_metrics
from src.instrumentation.redundancy_detector import RedundancyReport, analyze_redundancy

ROOT         = Path(__file__).parent.parent
LOGS_DIR     = ROOT / "results" / "logs"
REPORTS_DIR  = ROOT / "results" / "reports"
ANALYSIS_DIR = ROOT / "results" / "analysis"

ARCHITECTURES = ["supervisor", "hierarchical", "decentralized"]

# Métriques à analyser
METRIC_KEYS = [
    "total_mcp_calls",
    "redundant_mcp_calls",
    "redundancy_rate",
    "mcp_data_volume_bytes",
    "total_cost_usd",
    "overhead_orchestration_cost_usd",
    "wall_clock_duration_seconds",
    "total_tokens_input",
    "total_tokens_output",
    "total_tokens_cache_read",
]

# Catégories des repos (depuis repos.yaml)
_CATEGORY_MAP: dict[str, str] = {}  # repo_full → category, rempli dynamiquement


# ── Chargement des données ────────────────────────────────────────────────────


def load_all_metrics(filter_arch: str | None = None) -> dict[str, list[RunMetrics]]:
    """Charge tous les RunMetrics depuis results/logs/*.jsonl."""
    by_arch: dict[str, list[RunMetrics]] = {a: [] for a in ARCHITECTURES}

    for log_file in sorted(LOGS_DIR.glob("*.jsonl")):
        stem = log_file.stem  # ex: supervisor-pallets-flask-run1
        parts = stem.split("-")
        if len(parts) < 4:
            continue
        arch = parts[0]
        if arch not in ARCHITECTURES:
            continue
        if filter_arch and arch != filter_arch:
            continue
        try:
            m = compute_metrics(log_file)
            by_arch[arch].append(m)
        except Exception as e:
            print(f"  ⚠ Erreur lecture {log_file.name}: {e}")

    return by_arch


def load_all_redundancy(filter_arch: str | None = None) -> dict[str, list[RedundancyReport]]:
    """Charge tous les RedundancyReports."""
    by_arch: dict[str, list[RedundancyReport]] = {a: [] for a in ARCHITECTURES}

    for log_file in sorted(LOGS_DIR.glob("*.jsonl")):
        stem  = log_file.stem
        parts = stem.split("-")
        if len(parts) < 4:
            continue
        arch = parts[0]
        if arch not in ARCHITECTURES:
            continue
        if filter_arch and arch != filter_arch:
            continue
        try:
            r = analyze_redundancy(log_file)
            by_arch[arch].append(r)
        except Exception as e:
            print(f"  ⚠ Erreur redondance {log_file.name}: {e}")

    return by_arch


def load_all_reports() -> dict[str, list[dict]]:
    """Charge tous les FinalAuditReport (JSON) depuis results/reports/."""
    by_arch: dict[str, list[dict]] = {a: [] for a in ARCHITECTURES}
    for report_file in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            arch = data.get("architecture", "")
            if arch in by_arch:
                by_arch[arch].append(data)
        except Exception as e:
            print(f"  ⚠ Erreur rapport {report_file.name}: {e}")
    return by_arch


def _load_category_map() -> None:
    import yaml
    repos_yaml = ROOT / "data" / "repos.yaml"
    if not repos_yaml.exists():
        return
    with open(repos_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for r in data.get("repositories", []):
        key = f"{r['owner']}/{r['name']}"
        _CATEGORY_MAP[key] = r.get("category", "unknown")


# ── Statistiques ──────────────────────────────────────────────────────────────


def _stats(vals: list[float]) -> dict:
    import statistics
    if not vals:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n":      len(vals),
        "mean":   round(statistics.mean(vals), 4),
        "std":    round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
        "median": round(statistics.median(vals), 4),
        "min":    round(min(vals), 4),
        "max":    round(max(vals), 4),
    }


def aggregate_by_arch(by_arch: dict[str, list[RunMetrics]]) -> dict[str, dict]:
    """Calcule les statistiques par architecture pour chaque métrique."""
    result: dict[str, dict] = {}
    for arch, metrics_list in by_arch.items():
        result[arch] = {}
        for key in METRIC_KEYS:
            vals = [getattr(m, key) for m in metrics_list if hasattr(m, key)]
            result[arch][key] = _stats(vals)
    return result


# ── Tests statistiques Mann-Whitney ──────────────────────────────────────────


def mann_whitney_tests(by_arch: dict[str, list[RunMetrics]]) -> dict:
    """Test Mann-Whitney U par paire d'architectures pour chaque métrique."""
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {"error": "scipy not installed — pip install scipy"}

    pairs = [
        ("supervisor",   "hierarchical"),
        ("supervisor",   "decentralized"),
        ("hierarchical", "decentralized"),
    ]
    results: dict = {}
    for key in METRIC_KEYS:
        results[key] = {}
        for a1, a2 in pairs:
            vals1 = [getattr(m, key) for m in by_arch.get(a1, []) if hasattr(m, key)]
            vals2 = [getattr(m, key) for m in by_arch.get(a2, []) if hasattr(m, key)]
            if len(vals1) < 3 or len(vals2) < 3:
                results[key][f"{a1}_vs_{a2}"] = {"n1": len(vals1), "n2": len(vals2), "p_value": None, "significant": None}
                continue
            stat, p = scipy_stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            results[key][f"{a1}_vs_{a2}"] = {
                "n1":          len(vals1),
                "n2":          len(vals2),
                "U_statistic": round(float(stat), 4),
                "p_value":     round(float(p), 6),
                "significant": bool(p < 0.05),
            }
    return results


# ── Analyse par catégorie ─────────────────────────────────────────────────────


def analyze_by_category(by_arch: dict[str, list[RunMetrics]]) -> dict:
    """Agrège les métriques par catégorie de repo."""
    _load_category_map()
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for arch, metrics_list in by_arch.items():
        for m in metrics_list:
            cat = _CATEGORY_MAP.get(m.repository, "unknown")
            for key in ["total_mcp_calls", "redundancy_rate", "total_cost_usd", "wall_clock_duration_seconds"]:
                by_cat[cat][key].append(getattr(m, key))

    return {
        cat: {key: _stats(vals) for key, vals in metrics.items()}
        for cat, metrics in by_cat.items()
    }


# ── Matrice de redondance agrégée ─────────────────────────────────────────────


def aggregate_redundancy_matrix(by_arch: dict[str, list[RedundancyReport]]) -> dict:
    """Somme les matrices paire-à-paire sur tous les runs par architecture."""
    result: dict[str, dict[str, dict]] = {}
    for arch, reports in by_arch.items():
        matrix_sum: dict[str, int] = defaultdict(int)
        tool_sum:   dict[str, int] = defaultdict(int)
        for r in reports:
            for pair, cnt in r.redundancy_matrix.items():
                matrix_sum[pair] += cnt
            for tool, cnt in r.redundant_tools.items():
                tool_sum[tool] += cnt
        # Normaliser par nombre de runs
        n = max(len(reports), 1)
        result[arch] = {
            "n_runs":          len(reports),
            "matrix_avg":      {k: round(v / n, 2) for k, v in matrix_sum.items()},
            "tools_avg":       dict(sorted(
                {k: round(v / n, 2) for k, v in tool_sum.items()}.items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
        }
    return result


# ── Export CSV du tableau central ─────────────────────────────────────────────


def export_metrics_csv(aggregated: dict[str, dict], out_path: Path) -> None:
    labels = {
        "total_mcp_calls":                   "MCP calls",
        "redundant_mcp_calls":               "Redondants",
        "redundancy_rate":                   "Taux redondance",
        "mcp_data_volume_bytes":             "Volume données (B)",
        "total_cost_usd":                    "Coût total ($)",
        "overhead_orchestration_cost_usd":   "Overhead LLM ($)",
        "wall_clock_duration_seconds":       "Durée (s)",
        "total_tokens_input":                "Tokens input",
        "total_tokens_output":               "Tokens output",
        "total_tokens_cache_read":           "Cache read",
    }
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Métrique"] + [
            col
            for a in ARCHITECTURES
            for col in (f"{a} mean", f"{a} std", f"{a} median")
        ]
        writer.writerow(header)
        for key, label in labels.items():
            row = [label]
            for arch in ARCHITECTURES:
                s = aggregated.get(arch, {}).get(key, {})
                row += [s.get("mean", ""), s.get("std", ""), s.get("median", "")]
            writer.writerow(row)
    print(f"  ✓ CSV : {out_path}")


# ── Rapport lisible ────────────────────────────────────────────────────────────


def print_and_save_summary(
    aggregated:     dict[str, dict],
    mw_tests:       dict,
    by_category:    dict,
    redund_matrix:  dict,
    out_path:       Path,
) -> None:
    lines: list[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def row_3(label: str, vals: list) -> None:
        lines.append(f"  {label:<35}" + "".join(f"{str(v):>17}" for v in vals))

    h("TABLEAU CENTRAL — Moyennes ± écarts-types (tous repos, tous runs)")
    lines.append(f"  {'Métrique':<35}" + "".join(f"{a:>17}" for a in ARCHITECTURES))
    lines.append("  " + "-" * (35 + 17 * len(ARCHITECTURES)))

    metric_labels = [
        ("total_mcp_calls",                 "MCP calls"),
        ("redundant_mcp_calls",             "Redondants"),
        ("redundancy_rate",                 "Taux redondance"),
        ("total_cost_usd",                  "Coût total ($)"),
        ("overhead_orchestration_cost_usd", "Overhead LLM ($)"),
        ("wall_clock_duration_seconds",     "Durée (s)"),
        ("total_tokens_input",              "Tokens input"),
        ("total_tokens_cache_read",         "Cache read"),
    ]
    for key, label in metric_labels:
        vals = []
        for arch in ARCHITECTURES:
            s = aggregated.get(arch, {}).get(key, {})
            m = s.get("mean")
            std = s.get("std")
            if m is None:
                vals.append("N/A")
            elif key in ("redundancy_rate",):
                vals.append(f"{m:.1%}±{std:.1%}")
            elif key in ("total_cost_usd", "overhead_orchestration_cost_usd"):
                vals.append(f"${m:.4f}±{std:.4f}")
            else:
                vals.append(f"{m:.1f}±{std:.1f}")
        row_3(label, vals)

    h("TESTS MANN-WHITNEY (p < 0.05 = différence significative)")
    if "error" in mw_tests:
        lines.append(f"  {mw_tests['error']}")
    else:
        for key in ["total_mcp_calls", "redundancy_rate", "total_cost_usd", "wall_clock_duration_seconds"]:
            lines.append(f"\n  {key}")
            for pair_key, result in mw_tests.get(key, {}).items():
                p = result.get("p_value")
                sig = "✓ sig." if result.get("significant") else "  n.s."
                p_str = f"{p:.4f}" if p is not None else "N/A"
                lines.append(f"    {pair_key:<40} p={p_str}  {sig}")

    h("RÉSULTATS PAR CATÉGORIE DE REPO")
    for cat, metrics in by_category.items():
        lines.append(f"\n  [{cat}]")
        for key in ["total_mcp_calls", "redundancy_rate", "total_cost_usd"]:
            s = metrics.get(key, {})
            m = s.get("mean")
            if m is not None:
                lines.append(f"    {key:<40} mean={m:.3f}")

    h("OUTILS MCP LES PLUS REDONDANTS (moyenne par run)")
    for arch, data in redund_matrix.items():
        lines.append(f"\n  [{arch}]  ({data['n_runs']} runs)")
        for tool, avg in list(data.get("tools_avg", {}).items())[:5]:
            lines.append(f"    {tool:<40} {avg:.1f} doublon(s)/run")

    report_str = "\n".join(lines)
    print(report_str)
    out_path.write_text(report_str, encoding="utf-8")
    print(f"\n  ✓ Rapport : {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    filter_arch = None
    if "--arch" in sys.argv:
        idx = sys.argv.index("--arch")
        filter_arch = sys.argv[idx + 1]

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement des métriques …")
    by_arch_metrics    = load_all_metrics(filter_arch)
    by_arch_redundancy = load_all_redundancy(filter_arch)

    total_runs = sum(len(v) for v in by_arch_metrics.values())
    print(f"  {total_runs} runs chargés ({', '.join(f'{a}: {len(v)}' for a, v in by_arch_metrics.items())})")

    if total_runs == 0:
        print("  Aucun run trouvé. Lancez d'abord run_full_benchmark.py.")
        return

    print("\nAgrégation …")
    aggregated    = aggregate_by_arch(by_arch_metrics)
    mw_tests      = mann_whitney_tests(by_arch_metrics)
    by_category   = analyze_by_category(by_arch_metrics)
    redund_matrix = aggregate_redundancy_matrix(by_arch_redundancy)

    print("\nExport …")
    export_metrics_csv(aggregated, ANALYSIS_DIR / "metrics_table.csv")

    (ANALYSIS_DIR / "mann_whitney.json").write_text(
        json.dumps(mw_tests, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ANALYSIS_DIR / "by_category.json").write_text(
        json.dumps(by_category, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ANALYSIS_DIR / "redundancy_matrix.json").write_text(
        json.dumps(redund_matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n")
    print_and_save_summary(
        aggregated, mw_tests, by_category, redund_matrix,
        ANALYSIS_DIR / "summary_report.txt",
    )


if __name__ == "__main__":
    main()
