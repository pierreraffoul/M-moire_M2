# GitHubAuditBench

Benchmark empirique comparant **3 architectures multi-agents LangGraph** (Superviseur, Hiérarchique, Décentralisé) sur leur **efficience d'utilisation du Model Context Protocol (MCP)** appliquée à l'audit automatisé de projets open-source GitHub.

## Contexte académique

Ce projet est le support expérimental d'un mémoire de M2 dont l'objectif est de mesurer l'**efficience MCP** (appels redondants, partage de contexte, coût par architecture) selon la topologie multi-agents — un sujet peu traité dans la littérature. Le papier le plus proche est CA-MCP (Jayanti & Han, arXiv:2601.11595, 2026) qui modifie le MCP ; ici nous gardons le MCP standard et comparons les architectures qui le consomment.

## Principe de comparaison valide

Les **agents sont strictement identiques** entre les 3 architectures (même classe de base, mêmes prompts, même modèle LLM, même client MCP). **Seule l'orchestration varie.**

## Stack technique

- **Python 3.11+**, **uv** pour la gestion de paquets
- **LangGraph natif** (`StateGraph`) — pas de `create_agent` ni `AgentExecutor`
- **langchain-anthropic** (`ChatAnthropic`) — interface LLM uniquement
- **langchain-mcp-adapters** (`MultiServerMCPClient`) — connexion au GitHub MCP
- **structlog** — logs JSON structurés pour l'instrumentation MCP
- **pydantic v2** — modèles de données

## Architecture des agents

Chaque agent implémente une **boucle ReAct LangGraph native** (2 nœuds : `model` + `tools` + conditional edge) via `BaseAuditAgent`. Les 5 agents spécialisés héritent de cette base :

| Agent | Outils MCP GitHub (whitelist) |
|---|---|
| `code_quality` | `search_code`, `list_workflows`, `get_file_contents`, `list_pull_requests` |
| `community` | `list_commits`, `list_contributors`, `list_issues`, `get_repository` |
| `security` | `list_alerts`, `list_dependabot_alerts`, `list_secret_scanning_alerts` |
| `documentation` | `get_file_contents` |
| `license` | `get_file_contents`, `get_repository` |

## 3 Architectures comparées

```
Superviseur          Hiérarchique           Décentralisé
─────────────        ─────────────────      ──────────────────
supervisor LLM       top_supervisor         entry_node
    │                 ┌──┴──┐               code_quality ──→ security
    ├── code_quality  tech   community       security ──→ license
    ├── security      team   team            license ──→ ...
    ├── license       │      │               ... ──→ synthesizer
    ├── community     ...    ...
    ├── documentation
    └── synthesizer  synthesizer            synthesizer
```

## 7 Métriques mesurées

1. `total_mcp_calls` — nombre total d'appels MCP
2. `redundant_mcp_calls` — appels avec même hash de paramètres
3. `mcp_data_volume_bytes` — volume de données reçues via MCP
4. `total_cost_usd` — coût tokens Anthropic
5. `total_tokens_input` + `total_tokens_output`
6. `wall_clock_duration_seconds`
7. `audit_quality_score` — /20 évalué par LLM juge

## Installation

```bash
# Prérequis : Python 3.11+, uv installé
uv sync

# Copier et remplir les secrets
cp .env.example .env
# Éditer .env : ANTHROPIC_API_KEY, GITHUB_TOKEN
```

## Utilisation

```bash
# Audit unique
uv run python experiments/run_single_audit.py \
    --repo facebook/react \
    --architecture supervisor \
    --run-id debug-1

# Benchmark complet (10 repos × 3 archis × 3 runs = 90 audits)
uv run python experiments/run_full_benchmark.py
```

## Structure du projet

```
github-audit-bench/
├── src/
│   ├── agents/          # BaseAuditAgent + 5 agents spécialisés
│   ├── architectures/   # supervisor, hierarchical, decentralized
│   ├── mcp/             # InstrumentedMCPClient + GitHub wrapper
│   ├── instrumentation/ # logger, metrics, redundancy_detector
│   └── evaluation/      # grid_scorer, llm_judge, factual_checker
├── prompts/             # Prompts système versionnés en markdown
├── data/repos.yaml      # 10 repos à auditer
├── experiments/         # Scripts CLI
└── results/             # Logs JSON + rapports (gitignored)
```

## Ordre de développement

1. [x] Setup : pyproject.toml, structure, README, .env.example
2. [ ] Exploration MCP : connexion GitHub MCP, appel `get_repository` direct
3. [ ] Instrumentation : `InstrumentedMCPClient` + `logger.py`
4. [ ] `BaseAuditAgent` : boucle ReAct LangGraph native
5. [ ] **Validation** avec toi avant de continuer
6. [ ] 4 agents spécialisés restants
7. [ ] `supervisor.py`
8. [ ] `hierarchical.py`
9. [ ] `decentralized.py`
10. [ ] `metrics.py` + `redundancy_detector.py`
11. [ ] `evaluation/`
12. [ ] `run_full_benchmark.py`
13. [ ] Tests unitaires

## Licence

Usage académique uniquement — mémoire M2 MIAGE.
