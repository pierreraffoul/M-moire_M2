"""
Architecture Supervisor — orchestration séquentielle par LLM.

Topologie :
    START → supervisor → agent_X → supervisor → agent_Y → ... → synthesize → END

Le nœud supervisor est un LLM qui décide quel agent spécialisé activer ensuite.
Chaque agent spécialisé invoque son BaseAuditAgent.analyze() avec sa propre
session MCP (session par agent, cohérent avec le baseline étape 5).

Contrainte scientifique : les 5 agents sont IDENTIQUES à ceux de l'étape 5.
Seule l'orchestration diffère — c'est ce qu'on mesure.

Overhead mesuré :
    - Appels LLM du supervisor (décisions de routage)
    - Appel LLM du synthesizer (rapport final agrégé)
    Ces appels sont loggués avec agent_name="supervisor" / "synthesizer".

Usage :
    orchestrator = SupervisorOrchestrator(model=model, mcp_client=mcp_client)
    report = await orchestrator.run(repo="pallets/flask", run_id="sup-001")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agents.base_agent import AgentReport, Finding
from src.agents.code_quality import CodeQualityAgent
from src.agents.community import CommunityAgent
from src.agents.documentation import DocumentationAgent
from src.agents.license import LicenseAgent
from src.agents.security import SecurityAgent
from src.instrumentation.logger import get_logger
from src.mcp.instrumented_client import set_run_context

_logger = get_logger("supervisor")

_AGENT_NAMES = ["code_quality", "community", "security", "documentation", "license"]

# ── Schémas de données ────────────────────────────────────────────────────────


class SupervisorDecision(BaseModel):
    """Décision de routage du supervisor."""

    next_agent: Literal[
        "code_quality", "community", "security",
        "documentation", "license", "synthesize"
    ]
    reasoning: str = Field(description="Justification de la décision (1-2 phrases).")


class FinalAuditSynthesisOutput(BaseModel):
    """Schéma de sortie structurée du synthesizer."""

    global_score: float = Field(ge=0, le=20, description="Score global de 0 à 20.")
    summary: str = Field(description="Synthèse narrative en 3-5 phrases.")
    top_recommendations: list[str] = Field(
        description="3 à 5 recommandations prioritaires, triées par importance.",
    )


class FinalAuditReport(BaseModel):
    """Rapport final agrégé produit par l'architecture supervisor."""

    repository: str
    architecture: str = "supervisor"
    run_id: str
    global_score: float
    summary: str
    top_recommendations: list[str]
    agent_reports: dict[str, AgentReport]
    supervisor_iterations: int
    total_mcp_calls: int


# ── État du graphe ─────────────────────────────────────────────────────────────


class SupervisorState(TypedDict):
    repository: str
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_reports: dict[str, Any]       # AgentReport sérialisé par agent_name
    visited_agents: list[str]           # agents déjà exécutés (ordre d'appel)
    next_agent: str | None              # décision courante du supervisor
    iteration_count: int                # nb de tours du graphe principal
    final_report: FinalAuditReport | None


# ── Orchestrateur ──────────────────────────────────────────────────────────────


class SupervisorOrchestrator:
    """Architecture Supervisor : un LLM décide séquentiellement quel agent appeler.

    Le supervisor est un nœud LangGraph qui utilise structured_output pour
    produire une décision de routage. Les appels LLM du supervisor sont
    loggués séparément des agents spécialisés (overhead d'orchestration).

    Args:
        model: Instance ChatAnthropic partagée.
        mcp_client: Client MCP partagé (chaque agent ouvre sa propre session).
    """

    #: Limite de sécurité sur le nombre de tours du graphe principal
    MAX_SUPERVISOR_ITERATIONS: int = 12

    def __init__(
        self,
        model: ChatAnthropic,
        mcp_client: MultiServerMCPClient,
    ) -> None:
        self.model = model
        self.mcp_client = mcp_client
        self.agents: dict[str, Any] = {
            "code_quality": CodeQualityAgent(model=model, mcp_client=mcp_client),
            "community": CommunityAgent(model=model, mcp_client=mcp_client),
            "security": SecurityAgent(model=model, mcp_client=mcp_client),
            "documentation": DocumentationAgent(model=model, mcp_client=mcp_client),
            "license": LicenseAgent(model=model, mcp_client=mcp_client),
        }
        self._supervisor_prompt = self._load_prompt("supervisor.md")
        # include_raw=True pour capturer usage_metadata
        self._decision_model = model.with_structured_output(
            SupervisorDecision, include_raw=True
        )
        self._synthesis_model = model.with_structured_output(
            FinalAuditSynthesisOutput, include_raw=True
        )

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = Path(__file__).parent.parent.parent / "prompts" / filename
        return path.read_text(encoding="utf-8")

    # ── Point d'entrée public ──────────────────────────────────────────────────

    async def run(self, repo: str, run_id: str) -> FinalAuditReport:
        """Lance l'audit supervisor sur un repo.

        Args:
            repo: Repo à auditer, format "owner/name".
            run_id: Identifiant unique du run.

        Returns:
            FinalAuditReport agrégé.
        """
        _logger.info(
            "supervisor_start",
            repository=repo,
            run_id=run_id,
            architecture="supervisor",
        )

        graph = self._build_graph()
        initial_message = HumanMessage(
            content=(
                f"Audit the GitHub repository: {repo}\n"
                f"Activate each specialized agent in turn to gather comprehensive evidence. "
                f"Run all 5 agents, then synthesize."
            )
        )
        result = await graph.ainvoke(
            {
                "repository": repo,
                "run_id": run_id,
                "messages": [initial_message],
                "agent_reports": {},
                "visited_agents": [],
                "next_agent": None,
                "iteration_count": 0,
                "final_report": None,
            }
        )

        report = result["final_report"]
        if report is None:
            _logger.warning("supervisor_no_report", repository=repo)
            report = FinalAuditReport(
                repository=repo,
                run_id=run_id,
                global_score=0.0,
                summary="Synthesis failed.",
                top_recommendations=[],
                agent_reports={},
                supervisor_iterations=result.get("iteration_count", 0),
                total_mcp_calls=sum(
                    len(r.get("raw_data", {}))
                    for r in result.get("agent_reports", {}).values()
                ),
            )

        _logger.info(
            "supervisor_done",
            repository=repo,
            run_id=run_id,
            global_score=report.global_score,
            supervisor_iterations=report.supervisor_iterations,
            total_mcp_calls=report.total_mcp_calls,
        )
        return report

    # ── Construction du graphe ────────────────────────────────────────────────

    def _build_graph(self):
        """Construit le graphe supervisor avec les 5 nœuds agents + synthesizer."""
        orch = self  # capture pour les closures

        # ── Nœud supervisor : décision de routage ─────────────────────────────

        async def supervisor_node(state: SupervisorState) -> dict:
            """Appelle le LLM supervisor pour décider le prochain agent."""
            visited = state["visited_agents"]
            remaining = [a for a in _AGENT_NAMES if a not in visited]

            # Si tous les agents ont tourné → synthèse directe sans appel LLM
            if not remaining:
                return {
                    "next_agent": "synthesize",
                    "iteration_count": state["iteration_count"] + 1,
                }

            # Résumé des rapports déjà produits
            reports_summary = _format_reports_summary(state["agent_reports"])

            context_msg = HumanMessage(
                content=(
                    f"Repository: {state['repository']}\n"
                    f"Agents already run: {visited if visited else 'none'}\n"
                    f"Remaining agents: {remaining}\n"
                    f"Reports so far:\n{reports_summary}\n\n"
                    f"Which agent should run next?"
                )
            )

            messages = [
                SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": orch._supervisor_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                ),
                *state["messages"],
                context_msg,
            ]

            result = await orch._decision_model.ainvoke(messages)
            decision: SupervisorDecision = result["parsed"]
            raw_msg = result.get("raw")

            # Log LLM overhead du supervisor
            usage = getattr(raw_msg, "usage_metadata", None)
            if usage:
                details = usage.get("input_token_details", {})
                _logger.info(
                    "llm_call",
                    agent="supervisor",
                    repository=state["repository"],
                    iteration=f"supervisor_{state['iteration_count']}",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                    next_agent=decision.next_agent,
                    reasoning=decision.reasoning,
                )

            # Forcer "synthesize" si l'agent choisi a déjà tourné
            next_agent = decision.next_agent
            if next_agent in visited and next_agent != "synthesize":
                _logger.warning(
                    "supervisor_repeated_agent",
                    agent=next_agent,
                    visited=visited,
                )
                next_agent = remaining[0] if remaining else "synthesize"

            return {
                "messages": [AIMessage(content=f"Activating {next_agent}. {decision.reasoning}")],
                "next_agent": next_agent,
                "iteration_count": state["iteration_count"] + 1,
            }

        # ── Factory : nœuds agents ─────────────────────────────────────────────

        def make_agent_node(agent_name: str):
            agent = orch.agents[agent_name]

            async def agent_node(state: SupervisorState) -> dict:
                """Invoque l'agent spécialisé et stocke son rapport."""
                # set_run_context est appelé dans agent.analyze() avec agent_name correct
                set_run_context(
                    run_id=state["run_id"],
                    architecture="supervisor",
                    repository=state["repository"],
                    agent_name=agent_name,
                )
                report: AgentReport = await agent.analyze(
                    repo=state["repository"],
                    run_id=state["run_id"],
                    architecture="supervisor",
                )
                summary_msg = HumanMessage(
                    content=(
                        f"[{agent_name}] completed: score={report.score}/20, "
                        f"{len(report.findings)} findings, {len(report.raw_data)} MCP calls."
                    )
                )
                return {
                    "agent_reports": {**state["agent_reports"], agent_name: report},
                    "visited_agents": [*state["visited_agents"], agent_name],
                    "messages": [summary_msg],
                }

            agent_node.__name__ = f"{agent_name}_node"
            return agent_node

        # ── Nœud synthesizer ──────────────────────────────────────────────────

        async def synthesize_node(state: SupervisorState) -> dict:
            """Agrège les rapports des 5 agents en un FinalAuditReport."""
            agent_reports: dict[str, AgentReport] = state["agent_reports"]
            total_mcp = sum(len(r.raw_data) for r in agent_reports.values())

            # Préparer le contexte pour le LLM synthesizer
            reports_text = _format_reports_full(agent_reports)

            set_run_context(
                run_id=state["run_id"],
                architecture="supervisor",
                repository=state["repository"],
                agent_name="synthesizer",
            )

            synthesis_result = await orch._synthesis_model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You are the synthesis node of a multi-agent audit system. "
                            "Produce a final structured audit report aggregating the findings "
                            "from all specialized agents."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Repository: {state['repository']}\n\n"
                            f"Individual agent reports:\n{reports_text}\n\n"
                            "Produce the final audit report with:\n"
                            "- A global quality score (0-20) reflecting the weighted average of all dimensions\n"
                            "- A concise summary (3-5 sentences)\n"
                            "- 3-5 top prioritized recommendations"
                        )
                    ),
                ]
            )

            synthesis: FinalAuditSynthesisOutput = synthesis_result["parsed"]
            raw_msg = synthesis_result.get("raw")

            # Log LLM overhead du synthesizer
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
                run_id=state["run_id"],
                global_score=synthesis.global_score,
                summary=synthesis.summary,
                top_recommendations=synthesis.top_recommendations,
                agent_reports=agent_reports,
                supervisor_iterations=state["iteration_count"],
                total_mcp_calls=total_mcp,
            )
            return {"final_report": final}

        # ── Routage ───────────────────────────────────────────────────────────

        def route_after_supervisor(state: SupervisorState) -> str:
            """Route vers l'agent choisi, ou vers synthesize."""
            visited = state["visited_agents"]

            # Safety net : toutes itérations épuisées
            if state["iteration_count"] >= orch.MAX_SUPERVISOR_ITERATIONS:
                _logger.info(
                    "supervisor_max_iterations",
                    iteration_count=state["iteration_count"],
                )
                return "synthesize"

            # Tous les agents ont tourné
            if len(visited) >= len(_AGENT_NAMES):
                return "synthesize"

            next_agent = state.get("next_agent")
            if next_agent == "synthesize" or next_agent is None:
                return "synthesize"
            if next_agent in orch.agents:
                return next_agent

            return "synthesize"

        # ── Assemblage ────────────────────────────────────────────────────────

        builder = StateGraph(SupervisorState)
        builder.add_node("supervisor", supervisor_node)
        for name in _AGENT_NAMES:
            builder.add_node(name, make_agent_node(name))
        builder.add_node("synthesize", synthesize_node)

        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {name: name for name in _AGENT_NAMES} | {"synthesize": "synthesize"},
        )
        for name in _AGENT_NAMES:
            builder.add_edge(name, "supervisor")
        builder.add_edge("synthesize", END)

        return builder.compile()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_reports_summary(agent_reports: dict) -> str:
    """Résumé court des rapports pour le contexte du supervisor."""
    if not agent_reports:
        return "No reports yet."
    lines = []
    for name, report in agent_reports.items():
        if isinstance(report, AgentReport):
            lines.append(
                f"- {name}: score={report.score}/20, {len(report.findings)} findings"
            )
        else:
            lines.append(f"- {name}: (report unavailable)")
    return "\n".join(lines)


def _format_reports_full(agent_reports: dict) -> str:
    """Représentation complète des rapports pour la synthèse."""
    if not agent_reports:
        return "No reports available."
    sections = []
    for name, report in agent_reports.items():
        if not isinstance(report, AgentReport):
            continue
        findings_text = "\n".join(
            f"  [{f.severity.upper()}] {f.category}: {f.description}"
            for f in report.findings
        )
        sections.append(
            f"### {name} (score: {report.score}/20)\n{findings_text or '  No findings.'}"
        )
    return "\n\n".join(sections)
