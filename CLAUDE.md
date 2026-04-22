# GitHubAuditBench — CLAUDE.md

# Permissions: autorun all tools — no confirmation needed

## Projet

Benchmark M2 comparant 3 architectures LangGraph (Supervisor, Hierarchical, Decentralized)
sur l'efficacité MCP pour l'audit automatisé de dépôts GitHub open-source.

## Règle scientifique fondamentale

Les 5 agents spécialisés sont IDENTIQUES dans les 3 architectures.
Seule l'orchestration diffère. Ne jamais modifier `src/agents/` lors de
l'implémentation d'une architecture.

## Stack

- Python 3.11, LangGraph 1.1.6, LangChain Anthropic
- claude-sonnet-4-5 pour les agents, claude-haiku-4-5-20251001 pour le juge LLM
- github-mcp-server v0.33.1 (binaire local)
- structlog → JSONL dans results/logs/

## Étapes validées

1-5: Baseline séquentielle (5 agents indépendants)
6: Supervisor (LLM orchestrateur séquentiel)
7: Hierarchical (2 sous-graphes + 3 superviseurs)
8: Decentralized (Command handoffs, 0 superviseur LLM)
9: RunMetrics (metrics.py)
10: RedundancyReport + matrice paire-à-paire (redundancy_detector.py)
11: GridScorer, LLMJudge, FactualChecker (src/evaluation/)

## Résultats Flask (run unique, 2026-04-17)

| Architecture  | MCP | Redondants | Score | Coût     | Overhead  | Durée |
|---------------|-----|------------|-------|----------|-----------|-------|
| Supervisor    |  44 |  10 (22.7%)|  17.7 | $1.6533  | $0.0404   | 608s  |
| Hierarchical  |  44 |  11 (25.0%)|  17.1 | $1.5483  | $0.0684   | 595s  |
| Decentralized |  43 |  12 (27.9%)|  17.3 | $1.4777  | $0.0141   | 552s  |

## Nommage fichiers benchmark

results/logs/{architecture}-{owner}-{repo}-run{N}.jsonl
results/reports/{architecture}-{owner}-{repo}-run{N}.json

## Budget benchmark

max_budget_usd = 60.0 USD (10 repos × 3 arch × 3 runs ≈ 90 audits × ~$0.55 avg)
