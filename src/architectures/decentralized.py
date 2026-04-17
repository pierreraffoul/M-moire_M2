"""
Architecture Décentralisée — handoffs entre agents via Command(goto=...).

Topologie :
    START → code_quality → [security|community] → ... → synthesize → END

Pas de nœud superviseur LLM. Chaque nœud agent exécute son BaseAuditAgent.analyze(),
examine ses findings, puis décide via des règles déterministes quel agent activer
ensuite (Command pattern). Zéro overhead LLM d'orchestration.

Règles de routage (par agent) :
    code_quality  → security si findings HIGH/CRITICAL liés aux dépendances
                  → community sinon
    security      → license (contexte légal après sécurité)
    license       → community
    community     → documentation
    documentation → synthesize (tous les agents ont tourné)
    Fallback      → next unvisited dans l'ordre _AGENT_ORDER

Hypothèse mesurée :
    - 0 overhead LLM superviseur (coût orchestration nul)
    + risque de chemin sous-optimal vs LLM-supervisor
    Les 5 agents s'exécutent toujours (fallback garanti).

Usage :
    orchestrator = DecentralizedOrchestrator(model=model, mcp_client=mcp_client)
    report = await orchestrator.run(repo="pallets/flask", run_id="dec-001")
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

from src.agents.base_agent import AgentReport
from src.agents.code_quality import CodeQualityAgent
from src.agents.community import CommunityAgent
from src.agents.documentation import DocumentationAgent
from src.agents.license import LicenseAgent
from src.agents.security import SecurityAgent
from src.architectures.supervisor import FinalAuditReport, FinalAuditSynthesisOutput
from src.instrumentation.logger import get_logger
from src.mcp.instrumented_client import set_run_context

_logger = get_logger("decentralized")

# Ordre de priorité utilisé comme fallback dans les règles de routage
_AGENT_ORDER = ["code_quality", "security", "license", "community", "documentation"]

# ── État du graphe ─────────────────────────────────────────────────────────────


class DecentralizedState(TypedDict):
    repository: str
    run_id: str
    messages: Annotated[list, add_messages]
    agent_reports: dict[str, Any]       # AgentReport par agent_name
    visited_agents: list[str]
    iteration_count: int
    final_report: FinalAuditReport | None


# ── Règles de routage déterministes ───────────────────────────────────────────


def _next_unvisited(visited: list[str]) -> str:
    """Retourne le prochain agent non visité dans _AGENT_ORDER, ou 'synthesize'."""
    for agent in _AGENT_ORDER:
        if agent not in visited:
            return agent
    return "synthesize"


def _route_code_quality(report: AgentReport, visited: list[str]) -> str:
    """
    Si des findings HIGH/CRITICAL liés aux dépendances → security en priorité.
    Sinon → community (contexte contributeur avant documentation).
    """
    dep_keywords = ("depend", "supply", "package", "vulnerab", "pip", "npm", "pypi")
    has_dep_issue = any(
        any(kw in f.category.lower() or kw in f.description.lower() for kw in dep_keywords)
        for f in report.findings
        if f.severity in ("high", "critical")
    )
    preferred = "security" if has_dep_issue else "community"
    if preferred not in visited:
        return preferred
    return _next_unvisited(visited)


def _route_security(report: AgentReport, visited: list[str]) -> str:
    """security → license (contexte légal après analyse sécurité)."""
    if "license" not in visited:
        return "license"
    return _next_unvisited(visited)


def _route_license(report: AgentReport, visited: list[str]) -> str:
    """license → community."""
    if "community" not in visited:
        return "community"
    return _next_unvisited(visited)


def _route_community(report: AgentReport, visited: list[str]) -> str:
    """community → documentation."""
    if "documentation" not in visited:
        return "documentation"
    return _next_unvisited(visited)


def _route_documentation(report: AgentReport, visited: list[str]) -> str:
    """documentation → synthesize si tout visité, sinon fallback."""
    return _next_unvisited(visited)


_ROUTING_RULES = {
    "code_quality":  _route_code_quality,
    "security":      _route_security,
    "license":       _route_license,
    "community":     _route_community,
    "documentation": _route_documentation,
}


# ── Orchestrateur ──────────────────────────────────────────────────────────────


class DecentralizedOrchestrator:
    """Architecture décentralisée : handoffs par Command, zéro LLM orchestrateur.

    Args:
        model: Instance ChatAnthropic partagée.
        mcp_client: Client MCP partagé (chaque agent ouvre sa propre session).
    """

    MAX_ITERATIONS: int = 10  # garde-fou global (5 agents + quelques imprévus)

    def __init__(
        self,
        model: ChatAnthropic,
        mcp_client: MultiServerMCPClient,
    ) -> None:
        self.model = model
        self.mcp_client = mcp_client
        self.agents: dict[str, Any] = {
            "code_quality":  CodeQualityAgent(model=model, mcp_client=mcp_client),
            "community":     CommunityAgent(model=model, mcp_client=mcp_client),
            "security":      SecurityAgent(model=model, mcp_client=mcp_client),
            "documentation": DocumentationAgent(model=model, mcp_client=mcp_client),
            "license":       LicenseAgent(model=model, mcp_client=mcp_client),
        }
        self._synthesis_model = model.with_structured_output(
            FinalAuditSynthesisOutput, include_raw=True
        )

    # ── Point d'entrée public ──────────────────────────────────────────────────

    async def run(self, repo: str, run_id: str) -> FinalAuditReport:
        _logger.info(
            "decentralized_start",
            repository=repo,
            run_id=run_id,
            architecture="decentralized",
        )

        graph = self._build_graph()
        result = await graph.ainvoke({
            "repository": repo,
            "run_id": run_id,
            "messages": [HumanMessage(content=f"Decentralized audit of {repo}. Starting with code_quality.")],
            "agent_reports": {},
            "visited_agents": [],
            "iteration_count": 0,
            "final_report": None,
        })

        report = result["final_report"]
        if report is None:
            report = FinalAuditReport(
                repository=repo, run_id=run_id, architecture="decentralized",
                global_score=0.0, summary="Synthesis failed.", top_recommendations=[],
                agent_reports={}, supervisor_iterations=result["iteration_count"],
                total_mcp_calls=0,
            )

        _logger.info(
            "decentralized_done",
            repository=repo,
            global_score=report.global_score,
            total_mcp_calls=report.total_mcp_calls,
        )
        return report

    # ── Construction du graphe ────────────────────────────────────────────────

    def _build_graph(self):
        orch = self

        # ── Factory nœuds agents ──────────────────────────────────────────────

        def make_agent_node(agent_name: str):
            agent = orch.agents[agent_name]
            routing_rule = _ROUTING_RULES[agent_name]

            async def agent_node(state: DecentralizedState) -> Command:
                """Exécute l'agent, log ses findings, décide du prochain via règle."""
                visited = state["visited_agents"]
                iteration = state["iteration_count"]

                # Garde-fou : si trop d'itérations, forcer synthesize
                if iteration >= orch.MAX_ITERATIONS:
                    _logger.warning(
                        "decentralized_max_iterations",
                        agent=agent_name,
                        iteration_count=iteration,
                    )
                    return Command(
                        goto="synthesize",
                        update={
                            "visited_agents": [*visited, agent_name],
                            "iteration_count": iteration + 1,
                        },
                    )

                set_run_context(
                    run_id=state["run_id"],
                    architecture="decentralized",
                    repository=state["repository"],
                    agent_name=agent_name,
                )
                report: AgentReport = await agent.analyze(
                    repo=state["repository"],
                    run_id=state["run_id"],
                    architecture="decentralized",
                )

                new_visited = [*visited, agent_name]
                next_node = routing_rule(report, new_visited)

                # Log de la décision de routage (overhead = 0 LLM)
                _logger.info(
                    "decentralized_route",
                    from_agent=agent_name,
                    to_agent=next_node,
                    score=report.score,
                    findings_count=len(report.findings),
                    visited=new_visited,
                    repository=state["repository"],
                )

                return Command(
                    goto=next_node,
                    update={
                        "agent_reports": {**state["agent_reports"], agent_name: report},
                        "visited_agents": new_visited,
                        "iteration_count": iteration + 1,
                        "messages": [HumanMessage(
                            content=(
                                f"[{agent_name}] score={report.score}/20, "
                                f"{len(report.findings)} findings → {next_node}"
                            )
                        )],
                    },
                )

            agent_node.__name__ = f"{agent_name}_node"
            return agent_node

        # ── Nœud synthesizer ──────────────────────────────────────────────────

        async def synthesize_node(state: DecentralizedState) -> dict:
            """Agrège tous les rapports en un FinalAuditReport."""
            agent_reports: dict[str, AgentReport] = state["agent_reports"]
            total_mcp = sum(len(r.raw_data) for r in agent_reports.values())

            reports_text = "\n\n".join(
                f"### {n} (score: {r.score}/20)\n" + "\n".join(
                    f"  [{f.severity.upper()}] {f.category}: {f.description}"
                    for f in r.findings
                )
                for n, r in agent_reports.items()
                if isinstance(r, AgentReport)
            ) or "No reports available."

            set_run_context(
                run_id=state["run_id"],
                architecture="decentralized",
                repository=state["repository"],
                agent_name="synthesizer",
            )

            result = await orch._synthesis_model.ainvoke([
                SystemMessage(content=(
                    "You are the synthesis node of a decentralized multi-agent audit system. "
                    "Aggregate all agent findings into a final structured report."
                )),
                HumanMessage(content=(
                    f"Repository: {state['repository']}\n\n"
                    f"Agent execution order: {state['visited_agents']}\n\n"
                    f"Individual agent reports:\n{reports_text}\n\n"
                    "Produce the final audit report with:\n"
                    "- A global quality score (0-20)\n"
                    "- A concise summary (3-5 sentences)\n"
                    "- 3-5 top prioritized recommendations"
                )),
            ])

            synthesis: FinalAuditSynthesisOutput = result["parsed"]
            raw_msg = result.get("raw")

            usage = getattr(raw_msg, "usage_metadata", None)
            if usage:
                details = usage.get("input_token_details", {})
                _logger.info(
                    "llm_call",
                    agent="synthesizer",
                    repository=state["repository"],
                    iteration="synthesize",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                )

            final = FinalAuditReport(
                repository=state["repository"],
                architecture="decentralized",
                run_id=state["run_id"],
                global_score=synthesis.global_score,
                summary=synthesis.summary,
                top_recommendations=synthesis.top_recommendations,
                agent_reports=agent_reports,
                supervisor_iterations=state["iteration_count"],
                total_mcp_calls=total_mcp,
            )
            return {"final_report": final}

        # ── Assemblage ────────────────────────────────────────────────────────

        builder = StateGraph(DecentralizedState)

        # Nœuds agents — chacun retourne un Command(goto=...)
        for name in _AGENT_ORDER:
            builder.add_node(name, make_agent_node(name))

        builder.add_node("synthesize", synthesize_node)

        # Entrée fixe : toujours code_quality en premier
        builder.add_edge(START, "code_quality")

        # Les agents routent eux-mêmes via Command — pas besoin d'arêtes conditionnelles.
        # LangGraph résout les Command(goto=...) dynamiquement.

        builder.add_edge("synthesize", END)

        return builder.compile()
