"""
Architecture Hiérarchique — 2 sous-graphes compilés, 3 niveaux de supervision.

Topologie :
    top_supervisor → tech_team_node → top_supervisor → community_team_node → synthesize → END

Sous-graphe tech team (compilé) :
    START → tech_supervisor → [code_quality | security | license] → tech_supervisor → ... → tech_synthesis → END

Sous-graphe community team (compilé) :
    START → community_supervisor → [community | documentation] → community_supervisor → ... → community_synthesis → END

Hypothèse mesurée :
    + overhead LLM (3 supervisors vs 1 dans supervisor)
    - redondance MCP (chaque team-supervisor a un scope plus étroit)

Instrumentation :
    Appels LLM loggués avec agent_name = "top_supervisor" / "tech_supervisor" /
    "community_supervisor" / "tech_synthesizer" / "community_synthesizer" /
    "synthesizer" pour distinguer les 3 niveaux d'overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
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

_logger = get_logger("hierarchical")

_TECH_AGENTS      = ["code_quality", "security", "license"]
_COMMUNITY_AGENTS = ["community", "documentation"]
_ALL_AGENTS       = _TECH_AGENTS + _COMMUNITY_AGENTS

# ── Schémas de routage ────────────────────────────────────────────────────────


class TechTeamDecision(BaseModel):
    next_agent: Literal["code_quality", "security", "license", "synthesize"]
    reasoning: str


class CommunityTeamDecision(BaseModel):
    next_agent: Literal["community", "documentation", "synthesize"]
    reasoning: str


class TopSupervisorDecision(BaseModel):
    next_team: Literal["tech_team", "community_team", "synthesize"]
    reasoning: str


class TeamSynthesisOutput(BaseModel):
    team_score: float = Field(ge=0, le=20)
    team_summary: str
    key_findings: list[str] = Field(description="2-3 points saillants de l'équipe.")


# ── États ─────────────────────────────────────────────────────────────────────


class TeamState(TypedDict):
    """État interne d'un sous-graphe équipe."""
    repository: str
    run_id: str
    team_name: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_reports: dict[str, Any]
    visited_agents: list[str]
    next_agent: str | None
    iteration_count: int
    team_synthesis: dict | None   # sérialisé de TeamSynthesisOutput


class HierarchicalState(TypedDict):
    """État du graphe parent."""
    repository: str
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    all_agent_reports: dict[str, Any]
    teams_done: list[str]
    team_summaries: dict[str, str]
    next_team: str | None
    iteration_count: int
    final_report: FinalAuditReport | None


# ── Orchestrateur ─────────────────────────────────────────────────────────────


class HierarchicalOrchestrator:
    """Architecture hiérarchique avec 2 sous-graphes compilés.

    Chaque sous-graphe a son propre supervisor et produit une synthèse d'équipe.
    Le graphe parent orchestre les 2 équipes via un top_supervisor LLM.
    """

    MAX_TEAM_ITERATIONS: int = 8
    MAX_TOP_ITERATIONS: int  = 6

    def __init__(
        self,
        model: ChatAnthropic,
        mcp_client: MultiServerMCPClient,
    ) -> None:
        self.model = model
        self.mcp_client = mcp_client
        self.agents: dict[str, Any] = {
            "code_quality": CodeQualityAgent(model=model, mcp_client=mcp_client),
            "community":    CommunityAgent(model=model, mcp_client=mcp_client),
            "security":     SecurityAgent(model=model, mcp_client=mcp_client),
            "documentation":DocumentationAgent(model=model, mcp_client=mcp_client),
            "license":      LicenseAgent(model=model, mcp_client=mcp_client),
        }
        self._top_prompt  = self._load_prompt("supervisor.md")  # réutilise le prompt top-level
        self._tech_prompt = self._load_prompt("tech_team_supervisor.md")
        self._comm_prompt = self._load_prompt("community_team_supervisor.md")

        self._top_decision_model  = model.with_structured_output(TopSupervisorDecision,    include_raw=True)
        self._tech_decision_model = model.with_structured_output(TechTeamDecision,         include_raw=True)
        self._comm_decision_model = model.with_structured_output(CommunityTeamDecision,    include_raw=True)
        self._team_synthesis_model = model.with_structured_output(TeamSynthesisOutput,      include_raw=True)
        self._final_synthesis_model = model.with_structured_output(FinalAuditSynthesisOutput, include_raw=True)

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = Path(__file__).parent.parent.parent / "prompts" / filename
        return path.read_text(encoding="utf-8")

    # ── Point d'entrée public ──────────────────────────────────────────────────

    async def run(self, repo: str, run_id: str) -> FinalAuditReport:
        _logger.info("hierarchical_start", repository=repo, run_id=run_id)

        tech_subgraph      = self._build_team_subgraph("tech",      _TECH_AGENTS)
        community_subgraph = self._build_team_subgraph("community", _COMMUNITY_AGENTS)
        parent_graph       = self._build_parent_graph(tech_subgraph, community_subgraph)

        result = await parent_graph.ainvoke({
            "repository": repo,
            "run_id": run_id,
            "messages": [HumanMessage(content=f"Hierarchical audit of {repo}. Run both teams then synthesize.")],
            "all_agent_reports": {},
            "teams_done": [],
            "team_summaries": {},
            "next_team": None,
            "iteration_count": 0,
            "final_report": None,
        })

        report = result["final_report"]
        if report is None:
            report = FinalAuditReport(
                repository=repo, run_id=run_id, architecture="hierarchical",
                global_score=0.0, summary="Synthesis failed.", top_recommendations=[],
                agent_reports={}, supervisor_iterations=result["iteration_count"],
                total_mcp_calls=0,
            )

        _logger.info(
            "hierarchical_done",
            repository=repo, global_score=report.global_score,
            supervisor_iterations=report.supervisor_iterations,
        )
        return report

    # ── Sous-graphe équipe ─────────────────────────────────────────────────────

    def _build_team_subgraph(self, team_name: str, agent_names: list[str]):
        """Compile un StateGraph interne pour une équipe d'agents."""
        orch = self

        decision_model = (
            self._tech_decision_model if team_name == "tech"
            else self._comm_decision_model
        )
        supervisor_prompt = (
            self._tech_prompt if team_name == "tech"
            else self._comm_prompt
        )
        supervisor_log_name = f"{team_name}_supervisor"

        # Nœud supervisor interne
        async def team_supervisor_node(state: TeamState) -> dict:
            visited = state["visited_agents"]
            remaining = [a for a in agent_names if a not in visited]

            if not remaining:
                return {"next_agent": "synthesize", "iteration_count": state["iteration_count"] + 1}

            reports_summary = "\n".join(
                f"- {n}: score={r.score}/20, {len(r.findings)} findings"
                for n, r in state["agent_reports"].items()
                if isinstance(r, AgentReport)
            ) or "No reports yet."

            result = await decision_model.ainvoke([
                SystemMessage(content=[{
                    "type": "text",
                    "text": supervisor_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]),
                HumanMessage(content=(
                    f"Repository: {state['repository']}\n"
                    f"Team: {team_name}\n"
                    f"Visited: {visited or 'none'}\n"
                    f"Remaining: {remaining}\n"
                    f"Reports:\n{reports_summary}\n\nWhich agent next?"
                )),
            ])

            decision = result["parsed"]
            raw_msg  = result.get("raw")
            usage    = getattr(raw_msg, "usage_metadata", None)
            if usage:
                details = usage.get("input_token_details", {})
                _logger.info(
                    "llm_call",
                    agent=supervisor_log_name,
                    repository=state["repository"],
                    iteration=f"{supervisor_log_name}_{state['iteration_count']}",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                    next_agent=decision.next_agent,
                )

            next_agent = decision.next_agent
            if next_agent in visited and next_agent != "synthesize":
                next_agent = remaining[0] if remaining else "synthesize"

            return {
                "messages": [AIMessage(content=f"[{supervisor_log_name}] → {next_agent}")],
                "next_agent": next_agent,
                "iteration_count": state["iteration_count"] + 1,
            }

        # Factory nœuds agents
        def make_agent_node(agent_name: str):
            agent = orch.agents[agent_name]

            async def agent_node(state: TeamState) -> dict:
                set_run_context(
                    run_id=state["run_id"], architecture="hierarchical",
                    repository=state["repository"], agent_name=agent_name,
                )
                report: AgentReport = await agent.analyze(
                    repo=state["repository"], run_id=state["run_id"],
                    architecture="hierarchical",
                )
                return {
                    "agent_reports":  {**state["agent_reports"], agent_name: report},
                    "visited_agents": [*state["visited_agents"], agent_name],
                    "messages": [HumanMessage(
                        content=f"[{agent_name}] score={report.score}/20, {len(report.findings)} findings"
                    )],
                }

            agent_node.__name__ = f"{agent_name}_inner"
            return agent_node

        # Nœud synthèse équipe
        async def team_synthesis_node(state: TeamState) -> dict:
            agent_reports = state["agent_reports"]
            reports_text = "\n\n".join(
                f"### {n} (score: {r.score}/20)\n" + "\n".join(
                    f"  [{f.severity.upper()}] {f.category}: {f.description}"
                    for f in r.findings
                )
                for n, r in agent_reports.items()
                if isinstance(r, AgentReport)
            )

            set_run_context(
                run_id=state["run_id"], architecture="hierarchical",
                repository=state["repository"], agent_name=f"{team_name}_synthesizer",
            )

            result = await orch._team_synthesis_model.ainvoke([
                SystemMessage(content=f"You are the {team_name} team synthesizer. Aggregate findings from your team."),
                HumanMessage(content=(
                    f"Repository: {state['repository']}\n\n"
                    f"Agent reports:\n{reports_text}\n\n"
                    "Produce a team synthesis: score, summary, key findings."
                )),
            ])

            synthesis: TeamSynthesisOutput = result["parsed"]
            raw_msg = result.get("raw")
            usage = getattr(raw_msg, "usage_metadata", None)
            if usage:
                details = usage.get("input_token_details", {})
                _logger.info(
                    "llm_call",
                    agent=f"{team_name}_synthesizer",
                    repository=state["repository"],
                    iteration="team_synthesize",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                )

            return {"team_synthesis": synthesis.model_dump()}

        # Routage interne
        def should_continue_team(state: TeamState) -> str:
            if state.get("iteration_count", 0) >= orch.MAX_TEAM_ITERATIONS:
                return "team_synthesis"
            next_a = state.get("next_agent")
            if next_a == "synthesize" or next_a is None:
                return "team_synthesis"
            if next_a in agent_names:
                return next_a
            return "team_synthesis"

        # Assemblage sous-graphe
        builder = StateGraph(TeamState)
        builder.add_node("team_supervisor", team_supervisor_node)
        for name in agent_names:
            builder.add_node(name, make_agent_node(name))
        builder.add_node("team_synthesis", team_synthesis_node)

        builder.add_edge(START, "team_supervisor")
        builder.add_conditional_edges(
            "team_supervisor", should_continue_team,
            {name: name for name in agent_names} | {"team_synthesis": "team_synthesis"},
        )
        for name in agent_names:
            builder.add_edge(name, "team_supervisor")
        builder.add_edge("team_synthesis", END)

        return builder.compile()

    # ── Graphe parent ─────────────────────────────────────────────────────────

    def _build_parent_graph(self, tech_subgraph, community_subgraph):
        """Graphe parent avec top_supervisor et 2 nœuds wrapper pour les sous-graphes."""
        orch = self

        # ── Top supervisor ────────────────────────────────────────────────────

        async def top_supervisor_node(state: HierarchicalState) -> dict:
            done = state["teams_done"]
            remaining = [t for t in ["tech_team", "community_team"] if t not in done]

            if not remaining:
                return {"next_team": "synthesize", "iteration_count": state["iteration_count"] + 1}

            summaries_text = "\n".join(
                f"- {team}: {summary}"
                for team, summary in state.get("team_summaries", {}).items()
            ) or "No team reports yet."

            result = await orch._top_decision_model.ainvoke([
                SystemMessage(content=[{
                    "type": "text",
                    "text": orch._top_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]),
                HumanMessage(content=(
                    f"Repository: {state['repository']}\n"
                    f"Teams done: {done or 'none'}\n"
                    f"Teams remaining: {remaining}\n"
                    f"Team summaries:\n{summaries_text}\n\n"
                    "Which team should run next? Options: tech_team, community_team, or synthesize."
                )),
            ])

            decision: TopSupervisorDecision = result["parsed"]
            raw_msg = result.get("raw")
            usage = getattr(raw_msg, "usage_metadata", None)
            if usage:
                details = usage.get("input_token_details", {})
                _logger.info(
                    "llm_call",
                    agent="top_supervisor",
                    repository=state["repository"],
                    iteration=f"top_{state['iteration_count']}",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                    next_team=decision.next_team,
                )

            next_team = decision.next_team
            if next_team in done and next_team != "synthesize":
                next_team = remaining[0] if remaining else "synthesize"

            return {
                "messages": [AIMessage(content=f"[top_supervisor] → {next_team}. {decision.reasoning}")],
                "next_team": next_team,
                "iteration_count": state["iteration_count"] + 1,
            }

        # ── Wrapper nœuds équipes ─────────────────────────────────────────────

        async def tech_team_node(state: HierarchicalState) -> dict:
            result = await tech_subgraph.ainvoke({
                "repository": state["repository"],
                "run_id":     state["run_id"],
                "team_name":  "tech",
                "messages":   [],
                "agent_reports":  {},
                "visited_agents": [],
                "next_agent":     None,
                "iteration_count": 0,
                "team_synthesis": None,
            })
            synthesis = result.get("team_synthesis") or {}
            summary = synthesis.get("team_summary", "Tech team completed.")
            return {
                "all_agent_reports": {**state["all_agent_reports"], **result["agent_reports"]},
                "teams_done":   [*state["teams_done"], "tech_team"],
                "team_summaries": {**state.get("team_summaries", {}), "tech_team": summary},
                "messages": [HumanMessage(content=f"[tech_team] completed. Score: {synthesis.get('team_score', '?')}/20")],
            }

        async def community_team_node(state: HierarchicalState) -> dict:
            result = await community_subgraph.ainvoke({
                "repository": state["repository"],
                "run_id":     state["run_id"],
                "team_name":  "community",
                "messages":   [],
                "agent_reports":  {},
                "visited_agents": [],
                "next_agent":     None,
                "iteration_count": 0,
                "team_synthesis": None,
            })
            synthesis = result.get("team_synthesis") or {}
            summary = synthesis.get("team_summary", "Community team completed.")
            return {
                "all_agent_reports": {**state["all_agent_reports"], **result["agent_reports"]},
                "teams_done":   [*state["teams_done"], "community_team"],
                "team_summaries": {**state.get("team_summaries", {}), "community_team": summary},
                "messages": [HumanMessage(content=f"[community_team] completed. Score: {synthesis.get('team_score', '?')}/20")],
            }

        # ── Nœud synthèse finale ──────────────────────────────────────────────

        async def synthesize_node(state: HierarchicalState) -> dict:
            agent_reports = state["all_agent_reports"]
            total_mcp = sum(
                len(r.raw_data) for r in agent_reports.values()
                if isinstance(r, AgentReport)
            )
            reports_text = "\n\n".join(
                f"### {n} (score: {r.score}/20)\n" + "\n".join(
                    f"  [{f.severity.upper()}] {f.category}: {f.description}"
                    for f in r.findings
                )
                for n, r in agent_reports.items()
                if isinstance(r, AgentReport)
            )
            team_summaries_text = "\n".join(
                f"- {team}: {summary}"
                for team, summary in state.get("team_summaries", {}).items()
            )

            set_run_context(
                run_id=state["run_id"], architecture="hierarchical",
                repository=state["repository"], agent_name="synthesizer",
            )

            result = await orch._final_synthesis_model.ainvoke([
                SystemMessage(content="You are the final synthesizer for a hierarchical multi-agent audit."),
                HumanMessage(content=(
                    f"Repository: {state['repository']}\n\n"
                    f"Team summaries:\n{team_summaries_text}\n\n"
                    f"All agent reports:\n{reports_text}\n\n"
                    "Produce the final audit report with global score, summary, and top recommendations."
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
                    iteration="final_synthesize",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                )

            final = FinalAuditReport(
                repository=state["repository"],
                architecture="hierarchical",
                run_id=state["run_id"],
                global_score=synthesis.global_score,
                summary=synthesis.summary,
                top_recommendations=synthesis.top_recommendations,
                agent_reports=agent_reports,
                supervisor_iterations=state["iteration_count"],
                total_mcp_calls=total_mcp,
            )
            return {"final_report": final}

        # ── Routage parent ────────────────────────────────────────────────────

        def route_top(state: HierarchicalState) -> str:
            if state["iteration_count"] >= orch.MAX_TOP_ITERATIONS:
                return "synthesize"
            if len(state["teams_done"]) >= 2:
                return "synthesize"
            nt = state.get("next_team")
            if nt == "synthesize" or nt is None:
                return "synthesize"
            if nt in ("tech_team", "community_team"):
                return nt
            return "synthesize"

        # ── Assemblage ────────────────────────────────────────────────────────

        builder = StateGraph(HierarchicalState)
        builder.add_node("top_supervisor",   top_supervisor_node)
        builder.add_node("tech_team",        tech_team_node)
        builder.add_node("community_team",   community_team_node)
        builder.add_node("synthesize",       synthesize_node)

        builder.add_edge(START, "top_supervisor")
        builder.add_conditional_edges(
            "top_supervisor", route_top,
            {"tech_team": "tech_team", "community_team": "community_team", "synthesize": "synthesize"},
        )
        builder.add_edge("tech_team",      "top_supervisor")
        builder.add_edge("community_team", "top_supervisor")
        builder.add_edge("synthesize",     END)

        return builder.compile()
