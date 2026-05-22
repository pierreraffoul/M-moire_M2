# Rapport technique — GitHubAuditBench

> Architecture LangGraph multi-agents pour l'audit automatisé de dépôts GitHub  
> Benchmark M2 — Pierre Raffoul — Avril 2026

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Agents spécialisés](#2-agents-spécialisés)
   - 2.1 [Boucle ReAct partagée](#21-boucle-react-partagée)
   - 2.2 [Les cinq agents](#22-les-cinq-agents)
3. [Orchestrations](#3-orchestrations)
   - 3.1 [Supervisor](#31-architecture-supervisor)
   - 3.2 [Hierarchical](#32-architecture-hierarchical)
   - 3.3 [Decentralized](#33-architecture-decentralized)
4. [Intégration MCP](#4-intégration-mcp)
5. [Instrumentation et métriques](#5-instrumentation-et-métriques)
6. [Pipeline d'évaluation](#6-pipeline-dévaluation)
7. [Schémas de données](#7-schémas-de-données)
8. [Flux d'exécution bout-en-bout](#8-flux-dexécution-bout-en-bout)

---

## 1. Vue d'ensemble

Le projet implémente un benchmark scientifique comparant trois stratégies d'orchestration multi-agents (Supervisor, Hierarchical, Decentralized) sur la tâche d'audit de dépôts GitHub open-source. La variable expérimentale isolée est l'orchestration : les cinq agents spécialisés sont **rigoureusement identiques** dans les trois architectures.

```
┌─────────────────────────────────────────────┐
│              Benchmark Runner               │
│   (10 repos × 3 architectures × 3 runs)     │
└──────────────────┬──────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  Supervisor  Hierarchical  Decentralized
       │           │           │
       └───────────┼───────────┘
                   │
         ┌─────────▼─────────┐
         │   5 Agents MCP    │
         │ code_quality      │
         │ security          │
         │ license           │
         │ community         │
         │ documentation     │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ github-mcp-server │
         │   (stdio, v0.33)  │
         └───────────────────┘
```

**Stack technique** : Python 3.11, LangGraph 1.1.6, LangChain Anthropic, `claude-sonnet-4-5` pour les agents, `claude-haiku-4-5-20251001` pour le juge LLM.

---

## 2. Agents spécialisés

### 2.1 Boucle ReAct partagée

Tous les agents héritent de `BaseAuditAgent` (`src/agents/base_agent.py`). Cette classe implémente une boucle **ReAct** (Reasoning + Acting) comme un `StateGraph` LangGraph à trois nœuds.

#### Schéma d'état interne

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # accumulateur avec réducteur
    repository: str
    iteration_count: int
    agent_report: AgentReport | None
    context_limit_reached: bool
```

Le réducteur `add_messages` garantit l'accumulation séquentielle de tous les échanges LLM/outil.

#### Graphe ReAct (3 nœuds)

```
START
  │
  ▼
model_node ──────────────────────────────────────────┐
  │                                                   │
  │  (outil demandé ?)                               │
  ├── OUI ──► tools_node ──► model_node (boucle)     │
  │                                                   │
  └── NON ──────────────────────────────────────────►│
                                                      ▼
                                               synthesize_node
                                                      │
                                                      ▼
                                                     END
```

**`model_node`** :  
- Invoque `claude-sonnet-4-5` avec le prompt système mis en cache (directive `cache_control: ephemeral`)
- Vérifie `CONTEXT_TOKEN_LIMIT = 150 000 tokens` (approximation chars/4) avant chaque appel
- Logue chaque appel LLM avec les compteurs de tokens : `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`
- Incrémente `iteration_count` à chaque passage

**`tools_node`** :  
- Utilise `ToolNode` de LangChain avec `handle_tool_errors=handler`
- Le handler convertit toute exception en `ToolMessage` d'erreur pour ne pas bloquer la boucle
- N'exécute que les outils MCP de la whitelist de l'agent

**`synthesize_node`** :  
- Appelle le LLM avec `structured_output(AgentReportOutput, include_raw=True)`
- Extrait les réponses MCP brutes dans `raw_data` (indexé par `tool_call_id`) pour vérification ultérieure
- Gère les échecs de synthèse avec `try/except` : retourne un rapport vide plutôt que crasher

#### Fonction de routage `should_continue`

```python
def should_continue(state: AgentState) -> str:
    if state["context_limit_reached"]:
        return "synthesize"
    if state["iteration_count"] >= MAX_ITERATIONS:  # MAX_ITERATIONS = 8
        return "synthesize"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "synthesize"
```

Trois conditions de sortie forcée : dépassement du contexte, dépassement de la limite d'itérations, ou absence d'appel outil dans la dernière réponse LLM.

#### Mise en cache des prompts

```python
def _make_cached_system_message(self) -> SystemMessage:
    return SystemMessage(content=[{
        "type": "text",
        "text": self.system_prompt,
        "cache_control": {"type": "ephemeral"}
    }])
```

Les définitions d'outils MCP sont également marquées `ephemeral` lors de leur chargement initial. Cela permet de bénéficier du cache de prompt Anthropic sur la durée de la session MCP ouverte.

#### Modèles de données

```python
class Finding(BaseModel):
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str
    description: str
    evidence: str      # citation vérifiable (chemin, hash, URL...)
    recommendation: str

class AgentReport(BaseModel):
    agent_name: str
    repository: str
    findings: list[Finding]
    score: float       # 0-20
    raw_data: dict     # réponses MCP brutes par tool_call_id
```

---

### 2.2 Les cinq agents

Chaque agent spécialisé définit uniquement : son prompt système et sa liste d'outils MCP autorisés. La boucle ReAct est intégralement héritée.

#### SecurityAgent (`src/agents/security.py`)

**Périmètre** : vulnérabilités, alertes Dependabot, secrets exposés, scans CodeQL, workflows CI.

**Outils MCP autorisés (13)** :
```
list_dependabot_alerts      list_secret_scanning_alerts
list_code_scanning_alerts   get_file_contents
search_code                 get_repository_tree
get_repository              list_workflows
list_pull_requests          list_issues
list_commits                get_issue
get_pull_request
```

**Particularité** : le prompt gère explicitement les erreurs HTTP 403 sur les dépôts publics (endpoints sécurité désactivés) et pivote vers des signaux indirects (patterns dans le code, configuration CI).

---

#### CodeQualityAgent (`src/agents/code_quality.py`)

**Périmètre** : intégration continue, hygiène des PR, structure du dépôt, gestion des dépendances, patterns de code.

**Outils MCP autorisés (8)** :
```
get_file_contents     search_code
list_commits          list_pull_requests
list_workflows        get_repository
get_repository_tree   list_branches
```

**Contraintes dans le prompt** : `perPage ≤ 10`, maximum 5 fichiers lus, maximum 8 appels MCP — pour contrôler le coût et éviter la pollution du contexte.

---

#### DocumentationAgent (`src/agents/documentation.py`)

**Périmètre** : qualité du README, documentation API, changelog, guide de contribution, exemples de code.

**Outils MCP autorisés (5)** :
```
get_file_contents   get_repository_tree
search_code         get_repository
list_commits
```

---

#### LicenseAgent (`src/agents/license.py`)

**Périmètre** : présence d'une licence, métadonnées SPDX, cohérence des en-têtes de fichiers, compatibilité de licence.

**Outils MCP autorisés (4)** :
```
get_file_contents   get_repository_tree
get_repository      search_code
```

---

#### CommunityAgent (`src/agents/community.py`)

**Périmètre** : gestion des issues, activité des contributeurs, gouvernance du projet, fréquence de release, discussions.

**Outils MCP autorisés (7)** :
```
list_issues         get_issue
list_commits        get_file_contents
get_repository      get_repository_tree
search_code
```

---

## 3. Orchestrations

### 3.1 Architecture Supervisor

**Fichier** : `src/architectures/supervisor.py`  
**Principe** : un LLM orchestrateur central décide séquentiellement quel agent appeler, puis synthétise les résultats.

#### Schéma d'état

```python
class SupervisorState(TypedDict):
    repository: str
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_reports: dict[str, AgentReport]
    visited_agents: list[str]
    next_agent: str | None
    iteration_count: int
    final_report: FinalAuditReport | None
```

#### Topologie du graphe

```
START
  │
  ▼
supervisor_node
  │
  ├─► code_quality_node ──┐
  ├─► security_node ──────┤
  ├─► license_node ───────┼──► supervisor_node (boucle)
  ├─► community_node ─────┤
  ├─► documentation_node ─┘
  │
  └─► synthesize_node
              │
              ▼
             END
```

#### Nœud superviseur (`supervisor_node`)

```python
async def supervisor_node(state: SupervisorState) -> dict:
    remaining = _AGENT_NAMES - set(state["visited_agents"])
    if not remaining:
        return {"next_agent": "synthesize"}
    
    # Décision LLM (structured output)
    decision = await _decision_model.ainvoke(
        [_make_cached_system_message(), *state["messages"]]
    )
    # Détection boucle : si agent déjà visité, override
    if decision.next_agent in state["visited_agents"]:
        decision.next_agent = next(iter(remaining))
    
    return {
        "messages": [...],
        "next_agent": decision.next_agent,
        "iteration_count": state["iteration_count"] + 1
    }
```

Le modèle de décision utilise `SupervisorDecision(BaseModel)` avec `next_agent: Literal["code_quality", "security", "license", "community", "documentation", "synthesize"]` et un champ `reasoning`.

#### Nœuds agents (factory pattern)

```python
def make_agent_node(agent_name: str):
    async def node(state: SupervisorState) -> dict:
        set_run_context(state["run_id"], "supervisor", state["repository"], agent_name)
        report = await agents[agent_name].analyze(
            state["repository"], state["run_id"], "supervisor"
        )
        return {
            "agent_reports": {agent_name: report},
            "visited_agents": [*state["visited_agents"], agent_name],
            "messages": [AIMessage(content=f"{agent_name}: score={report.score}")]
        }
    return node
```

#### Fonction de routage

```python
def route_after_supervisor(state: SupervisorState) -> str:
    if state["iteration_count"] >= MAX_SUPERVISOR_ITERATIONS:  # 12
        return "synthesize"
    if len(state["visited_agents"]) >= len(_AGENT_NAMES):  # 5
        return "synthesize"
    if state["next_agent"] in valid_nodes:
        return state["next_agent"]
    return "synthesize"
```

**Overhead LLM** : N appels superviseur (1 par agent + appels de réorientation éventuels) + 1 appel synthèse finale.

---

### 3.2 Architecture Hierarchical

**Fichier** : `src/architectures/hierarchical.py`  
**Principe** : deux sous-graphes d'équipe supervisés localement, coordonnés par un superviseur de haut niveau.

#### Décomposition en équipes

| Équipe | Agents |
|--------|--------|
| Tech team | `code_quality`, `security`, `license` |
| Community team | `community`, `documentation` |

#### Schémas d'état

```python
class TeamState(TypedDict):
    team_name: str
    agent_reports: dict[str, AgentReport]
    visited_agents: list[str]
    team_synthesis: dict   # résumé d'équipe après synthèse

class HierarchicalState(TypedDict):
    repository: str
    run_id: str
    all_agent_reports: dict[str, AgentReport]
    teams_done: list[str]
    team_summaries: dict[str, str]
    next_team: str | None
    final_report: FinalAuditReport | None
```

#### Topologie globale

```
START
  │
  ▼
top_supervisor_node
  │
  ├──► tech_team_node ──────► top_supervisor_node
  │         │                      │
  │    [sous-graphe tech]          │
  │    tech_supervisor             │
  │      ├─► code_quality          │
  │      ├─► security              │
  │      └─► license               │
  │    tech_synthesis              │
  │                                │
  ├──► community_team_node ───► top_supervisor_node
  │         │
  │    [sous-graphe community]
  │    community_supervisor
  │      ├─► community
  │      └─► documentation
  │    community_synthesis
  │
  └──► synthesize_node
              │
              ▼
             END
```

#### Sous-graphe d'équipe (`_build_team_subgraph`)

Chaque équipe est un `StateGraph(TeamState)` compilé indépendamment. Il contient :

1. **`team_supervisor_node`** : décide quel agent de l'équipe lancer (`TechTeamDecision` ou `CommunityTeamDecision`)
2. **Nœuds agents** : identiques au pattern Supervisor
3. **`team_synthesis_node`** : produit un `TeamSynthesisOutput` (score d'équipe + résumé + findings clés)
4. **Routing `should_continue_team`** : même logique que Supervisor mais scopée à l'équipe (`MAX_TEAM_ITERATIONS = 8`)

#### Nœud superviseur de haut niveau

```python
async def top_supervisor_node(state: HierarchicalState) -> dict:
    remaining_teams = {"tech_team", "community_team"} - set(state["teams_done"])
    if not remaining_teams:
        return {"next_team": "synthesize"}
    
    decision = await _top_decision_model.ainvoke(...)  # TopSupervisorDecision
    return {"next_team": decision.next_team, ...}
```

#### Modèles LLM impliqués

| Modèle | Rôle | Niveau |
|--------|------|--------|
| `_top_decision_model` | Décide quelle équipe lancer | L3 |
| `_tech_decision_model` | Décide quel agent dans tech | L2 |
| `_comm_decision_model` | Décide quel agent dans community | L2 |
| `_team_synthesis_model` × 2 | Synthèse par équipe | L2 |
| `_final_synthesis_model` | Synthèse globale | L1 |

**Overhead LLM** : 3 niveaux de supervision (top, tech, community) + 3 synthèses (tech, community, finale). C'est l'architecture avec le plus grand overhead d'orchestration ($0.0684 mesuré).

---

### 3.3 Architecture Decentralized

**Fichier** : `src/architectures/decentralized.py`  
**Principe** : aucun superviseur LLM — les agents se passent le contrôle eux-mêmes via `Command(goto=...)` selon des règles déterministes.

#### Schéma d'état

```python
class DecentralizedState(TypedDict):
    repository: str
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_reports: dict[str, AgentReport]
    visited_agents: list[str]
    iteration_count: int
    final_report: FinalAuditReport | None
```

#### Topologie

```
START
  │
  ▼
code_quality ──► security ──► license ──► community ──► documentation
                                                               │
                                                               ▼
                                                         synthesize_node
                                                               │
                                                               ▼
                                                              END
```

Il n'y a **pas** de `add_conditional_edges` : chaque nœud agent retourne un `Command` qui spécifie directement le nœud suivant.

#### Règles de routage déterministes (`_ROUTING_RULES`)

```python
_ROUTING_RULES = {
    "code_quality":  _route_code_quality,
    "security":      _route_security,
    "license":       _route_license,
    "community":     _route_community,
    "documentation": _route_documentation,
}
```

**`_route_code_quality(report, visited)`** :  
- Si findings HIGH/CRITICAL avec mots-clés dépendances → `"security"`
- Sinon → `"community"`
- Fallback → `_next_unvisited(visited)`

**`_route_security(report, visited)`** :  
- → `"license"` (toujours)
- Fallback → `_next_unvisited(visited)`

**`_route_license(report, visited)`** :  
- → `"community"` (toujours)
- Fallback → `_next_unvisited(visited)`

**`_route_community(report, visited)`** :  
- → `"documentation"` (toujours)
- Fallback → `_next_unvisited(visited)`

**`_route_documentation(report, visited)`** :  
- → `_next_unvisited(visited)` (si tous visités → `"synthesize"`)

**`_next_unvisited(visited)`** : parcourt `_AGENT_ORDER` et retourne le premier non-visité, ou `"synthesize"`.

#### Nœuds agents avec `Command`

```python
def make_agent_node(agent_name: str):
    async def node(state: DecentralizedState) -> Command:
        report = await agents[agent_name].analyze(
            state["repository"], state["run_id"], "decentralized"
        )
        
        if state["iteration_count"] >= MAX_ITERATIONS:  # 10
            next_node = "synthesize"
        else:
            routing_fn = _ROUTING_RULES[agent_name]
            next_node = routing_fn(report, state["visited_agents"])
        
        return Command(
            goto=next_node,
            update={
                "agent_reports": {agent_name: report},
                "visited_agents": [*state["visited_agents"], agent_name],
                "iteration_count": state["iteration_count"] + 1,
                "messages": [AIMessage(content=f"{agent_name} done")]
            }
        )
    return node
```

**Overhead LLM** : 0 appel superviseur. Seule la synthèse finale utilise un LLM d'orchestration ($0.0141 mesuré).

---

## 4. Intégration MCP

### Client GitHub MCP (`src/mcp/github_client.py`)

Le `github-mcp-server` v0.33.1 est lancé comme processus enfant via transport `stdio` :

```python
MultiServerMCPClient({
    "github": {
        "transport": "stdio",
        "command": str(binary),       # chemin vers le binaire local
        "args": ["stdio", f"--toolsets={toolsets}", "--read-only"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": token}
    }
})
```

**Toolsets activés** (`BENCHMARK_TOOLSETS`) :
```
repos, git, issues, pull_requests, code_security,
dependabot, secret_protection, actions, discussions, users
```

Le flag `--read-only` garantit l'absence d'effets de bord sur les dépôts audités.

### Client instrumenté (`src/mcp/instrumented_client.py`)

Chaque appel MCP est intercepté par monkey-patching de `session.call_tool` :

```python
def _wrap_session_call_tool(session, logger):
    original = session.call_tool
    
    async def instrumented_call_tool(name, arguments):
        ctx = get_run_context()
        t0 = time.time()
        try:
            result = await original(name, arguments)
            duration_ms = (time.time() - t0) * 1000
            logger.info("mcp_call",
                run_id=ctx["run_id"],
                architecture=ctx["architecture"],
                repository=ctx["repository"],
                agent_name=ctx["agent_name"],
                mcp_tool=name,
                mcp_params=arguments,
                mcp_params_hash=sha256(str(arguments))[:16],
                response_size_bytes=len(str(result)),
                duration_ms=duration_ms,
                success=True
            )
            return result
        except Exception as e:
            logger.error("mcp_call", ..., success=False, error=str(e))
            raise
    
    session.call_tool = instrumented_call_tool
```

**Propagation du contexte** : un `ContextVar` asyncio (task-local) transporte `{run_id, architecture, repository, agent_name}` sans couplage entre les couches :

```python
_run_context: ContextVar[dict] = ContextVar("run_context", default={})

def set_run_context(run_id, architecture, repository, agent_name):
    _run_context.set({...})

def get_run_context() -> dict:
    return _run_context.get()
```

Cela garantit que deux agents tournant en parallèle (possible dans les architectures futures) ne mélangent pas leurs contextes.

---

## 5. Instrumentation et métriques

### Format des logs (JSONL)

Chaque ligne de `results/logs/{arch}-{owner}-{repo}-run{N}.jsonl` est un objet JSON valide. Deux types d'événements principaux :

**Appel MCP** :
```json
{
  "event": "mcp_call",
  "run_id": "supervisor-flask-run1",
  "architecture": "supervisor",
  "repository": "pallets/flask",
  "agent_name": "security",
  "mcp_tool": "list_dependabot_alerts",
  "mcp_params": {"owner": "pallets", "repo": "flask"},
  "mcp_params_hash": "a3f2b1c4d5e6f708",
  "response_size_bytes": 1843,
  "timestamp_start": "2026-04-17T10:00:00.000Z",
  "duration_ms": 845,
  "success": true,
  "error": null
}
```

**Appel LLM** :
```json
{
  "event": "llm_call",
  "agent": "supervisor",
  "input_tokens": 2841,
  "output_tokens": 124,
  "cache_read_input_tokens": 1204,
  "cache_creation_input_tokens": 0
}
```

### Calcul des métriques (`src/instrumentation/metrics.py`)

`compute_metrics(log_file, wall_clock_duration_seconds)` parse le JSONL et agrège :

**Métriques MCP** :
- `total_mcp_calls` : compte des événements `mcp_call` avec `success=true`
- `redundant_mcp_calls` : identifié par `RedundancyDetector`
- `mcp_data_volume_bytes` : somme des `response_size_bytes`

**Coûts LLM** (tarifs `claude-sonnet-4-5`) :

| Token type | Tarif |
|------------|-------|
| Input | $3 / 1M tokens |
| Output | $15 / 1M tokens |
| Cache read | $0.30 / 1M tokens |
| Cache creation | $3.75 / 1M tokens |

**Overhead d'orchestration** : somme des coûts LLM des agents dont `agent` ∈ `{supervisor, synthesizer, tech_supervisor, community_supervisor, top_supervisor, tech_synthesizer, community_synthesizer}`.

### Détection de redondance (`src/instrumentation/redundancy_detector.py`)

Construit une matrice paire-à-paire entre agents. Pour chaque paire `(agent_A, agent_B)`, identifie les appels MCP identiques (`mcp_tool` + `mcp_params_hash` identiques).

```python
class RedundancyReport(BaseModel):
    total_mcp_calls: int
    redundant_calls: int
    redundancy_rate: float
    matrix: dict[str, int]    # "agent_A×agent_B" → count
    details: list[RedundantCall]
```

---

## 6. Pipeline d'évaluation

### Grid Scorer (`src/evaluation/grid_scorer.py`)

Évaluation **sans LLM** basée sur des heuristiques regex. Score total 0–20 = somme de 4 dimensions.

| Dimension | Méthode | Poids |
|-----------|---------|-------|
| Factual Precision | Détection d'évidences spécifiques (chemins, hashes, URLs, versions, numéros de ligne) | 0–5 |
| Coverage | Présence des 5 dimensions d'audit dans les findings | 0–5 |
| Recommendation Quality | Verbes d'action + noms concrets détectés par regex | 0–5 |
| Clarity & Prioritization | Severité valide, findings structurés, présence de HIGH/CRITICAL, variété | 0–5 |

**Regex évidences spécifiques** :
```python
r'[./\\]|[0-9a-f]{7,}|https?://|v?\d+\.\d+|line\s+\d+|\d{4}-\d{2}-\d{2}'
```

**Calcul Clarity** (composite) :
```python
score = 0.4 * severity_rate + 0.4 * structured_rate + 0.1 * has_high_critical + 0.1 * has_variety
```

### LLM Judge (`src/evaluation/llm_judge.py`)

Évaluation indépendante par `claude-haiku-4-5-20251001` sur les mêmes 4 dimensions. Retourne `JudgeVerdict` avec score (0–5) et rationale par dimension.

**Tarification distincte** : $3/$15 /1M tokens input/output, loggué sous `agent="judge"` mais exclu des métriques de pipeline.

### Factual Checker (`src/evaluation/factual_checker.py`)

Vérifie les évidences citées dans les findings contre l'API REST GitHub :

1. Échantillonne 3–5 findings par rapport (seed = `report.run_id` pour reproductibilité)
2. Classifie chaque évidence :
   - **Workflow** : vérifie l'existence de `.github/workflows/*.yml` via `GET /repos/.../contents/...`
   - **File path** : extrait et vérifie le chemin
   - **GitHub URL** : requête HEAD
   - **Autre** : marqué `unverifiable`
3. `accuracy_rate = verified / (verified + not_found)` (exclut les non-vérifiables)

---

## 7. Schémas de données

### Hiérarchie des rapports

```
FinalAuditReport
├── repository: str
├── architecture: "supervisor" | "hierarchical" | "decentralized"
├── run_id: str
├── global_score: float (0–20)
├── summary: str
├── top_recommendations: list[str]
├── agent_reports: dict[str, AgentReport]
│   └── AgentReport
│       ├── agent_name: str
│       ├── repository: str
│       ├── score: float (0–20)
│       ├── findings: list[Finding]
│       │   └── Finding
│       │       ├── severity: "critical"|"high"|"medium"|"low"|"info"
│       │       ├── category: str
│       │       ├── description: str
│       │       ├── evidence: str
│       │       └── recommendation: str
│       └── raw_data: dict[tool_call_id, any]
├── supervisor_iterations: int   (Supervisor / Hierarchical)
└── total_mcp_calls: int
```

### Configuration (`src/config.py`)

```python
class Settings(BaseSettings):
    anthropic_api_key: SecretStr
    github_token: SecretStr
    model_name: str = "claude-sonnet-4-5"
    mcp_binary_path: Path
    results_dir: Path
    max_iterations: int = 20
    max_budget_usd: float = 60.0
```

Chargée une seule fois via `get_settings()` (singleton `@lru_cache`).

---

## 8. Flux d'exécution bout-en-bout

### Script de benchmark (`experiments/run_full_benchmark.py`)

```
Pour chaque (repo, architecture, run_n) :
  1. Vérifier si results/logs/{arch}-{owner}-{repo}-run{N}.jsonl existe
     → Oui : skip (idempotence)
  2. Vérifier budget consommé > max_budget_usd ($60)
     → Dépassé : arrêt propre
  3. Lancer l'audit avec timeout=900s
  4. En cas d'erreur : écrire .error et continuer
  5. Calculer RunMetrics depuis le JSONL
  6. Accumuler dans benchmark_summary.json
```

### Exemple de séquence d'un audit Supervisor

```
run_full_benchmark.py
  └─► SupervisorOrchestrator.run("pallets/flask", "supervisor-flask-run1")
        └─► StateGraph.ainvoke(initial_state)
              │
              ├─► supervisor_node          ← Décision LLM #1 : "lance code_quality"
              ├─► code_quality_node
              │     └─► BaseAuditAgent.analyze()
              │           └─► ReAct graph
              │                 ├─► model_node (LLM + cache)
              │                 ├─► tools_node (MCP : get_repository_tree, ...)
              │                 ├─► model_node (LLM)
              │                 ├─► tools_node (MCP : list_workflows, ...)
              │                 └─► synthesize_node (structured output)
              │
              ├─► supervisor_node          ← Décision LLM #2 : "lance security"
              ├─► security_node
              │     └─► (idem ReAct)
              │
              ├─► ... (license, community, documentation)
              │
              ├─► supervisor_node          ← Détecte tous visités → "synthesize"
              └─► synthesize_node          ← LLM final : FinalAuditReport
```

**Fichiers produits** :
- `results/logs/supervisor-pallets-flask-run1.jsonl` : tous les événements `mcp_call` + `llm_call`
- `results/reports/supervisor-pallets-flask-run1.json` : `FinalAuditReport` sérialisé

---

## Synthèse comparative

| Critère | Supervisor | Hierarchical | Decentralized |
|---------|-----------|-------------|--------------|
| Nœuds LLM d'orchestration | 1 superviseur | 3 superviseurs (top+2 équipes) | 0 |
| Overhead coût mesuré | $0.0404 | $0.0684 | $0.0141 |
| Flexibilité d'ordre | Totale (LLM décide) | Partielle (par équipe) | Fixe (règles) |
| Mécanisme de routage | `add_conditional_edges` | `add_conditional_edges` (2 niveaux) | `Command(goto=...)` |
| Appels MCP moyens | 44 | 44 | 43 |
| Score moyen | 17.7 | 17.1 | 17.3 |
| Durée moyenne | 608s | 595s | 552s |
| Redondance MCP | 22.7% | 25.0% | 27.9% |
| Garantie d'exécution | Via détection boucle + fallback | Via fallback d'équipe | Via `_next_unvisited` |

La variable expérimentale est strictement isolée : les cinq agents, leurs prompts, leurs outils MCP, et leur boucle ReAct sont **identiques** dans les trois architectures. Seule la couche d'orchestration diffère.
