"""
Boucle ReAct LangGraph native — réutilisée par tous les agents spécialisés.

Architecture du graphe (3 nœuds) :

    START → model_node ──[outil appelé]──→ tools_node ──→ model_node (boucle)
                       ↘[pas d'outil, ou max_iter]↘
                                         synthesize_node → END

- model_node    : appelle le LLM avec les outils MCP bindés
- tools_node    : exécute les tool_calls via ToolNode (session MCP instrumentée)
- synthesize_node : appelle le LLM en mode structured_output pour produire
                    l'AgentReport final

Règle respectée : TOUS les appels LLM sont dans des nœuds de ce StateGraph.

Exemple d'utilisation :
    from src.agents.code_quality import CodeQualityAgent
    from src.mcp.github_client import build_github_mcp_client
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-4-5")
    mcp_client = build_github_mcp_client(token=..., binary_path=...)
    agent = CodeQualityAgent(model=model, mcp_client=mcp_client)
    report = await agent.analyze(repo="pallets/flask", run_id="run-001", architecture="supervisor")
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.instrumentation.logger import get_logger
from src.mcp.instrumented_client import InstrumentedMCPClient, set_run_context

# ── Modèles de données ────────────────────────────────────────────────────────

_logger = get_logger("agent")


class Finding(BaseModel):
    """Un constat d'audit, avec preuve vérifiable."""

    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str
    description: str
    evidence: str = Field(
        description="Donnée brute citée depuis un appel MCP (vérifiable)."
    )
    recommendation: str


class AgentReportOutput(BaseModel):
    """Schéma de sortie pour l'étape de synthèse structurée (LLM → JSON).

    Séparé d'AgentReport pour ne pas demander au LLM de produire agent_name,
    repository et raw_data (qui sont remplis programmatiquement).
    """

    findings: list[Finding]
    score: float = Field(
        ge=0, le=20,
        description="Score de qualité global de 0 à 20."
    )


class AgentReport(BaseModel):
    """Rapport d'audit complet retourné par analyze()."""

    agent_name: str
    repository: str
    findings: list[Finding]
    score: float
    raw_data: dict = Field(
        description="Réponses MCP brutes indexées par tool_call_id (audit scientifique)."
    )


# ── État du graphe ─────────────────────────────────────────────────────────────


def _add_messages(left: list, right: list) -> list:
    """Reducer : ajoute les nouveaux messages à la liste existante."""
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], _add_messages]
    repository: str
    iteration_count: int
    agent_report: AgentReport | None
    context_limit_reached: bool


# ── Agent de base ─────────────────────────────────────────────────────────────


class BaseAuditAgent:
    """Boucle ReAct LangGraph native, réutilisée par tous les agents spécialisés.

    Sous-classes : définissent uniquement name, system_prompt_file, allowed_mcp_tools.
    La boucle ReAct (model → tools → model → ... → synthesize) est implémentée ici.

    Contrainte respectée : aucun appel LLM en dehors des nœuds du graphe.
    """

    #: Nombre max d'itérations ReAct — évite les boucles infinies
    MAX_ITERATIONS: int = 8

    #: Seuil de tokens estimés à partir duquel on force la synthèse
    #: (marge de sécurité sur les 200 000 tokens max d'Anthropic)
    CONTEXT_TOKEN_LIMIT: int = 150_000

    def __init__(
        self,
        name: str,
        system_prompt: str,
        allowed_mcp_tools: list[str],
        model: ChatAnthropic,
        mcp_client: MultiServerMCPClient,
    ) -> None:
        """Initialise l'agent.

        Args:
            name: Nom de l'agent (ex: "code_quality").
            system_prompt: Contenu du prompt système (chargé depuis prompts/<name>.md).
            allowed_mcp_tools: Whitelist des noms d'outils MCP accessibles.
            model: Instance ChatAnthropic partagée entre tous les agents.
            mcp_client: Client MCP brut (MultiServerMCPClient) — la session
                        est ouverte dans analyze(), pas ici.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.allowed_mcp_tools = allowed_mcp_tools
        self.model = model
        self.mcp_client = mcp_client
        self._instrumented = InstrumentedMCPClient()

    @classmethod
    def load_prompt(cls, prompt_file: str) -> str:
        """Charge un prompt depuis prompts/<name>.md.

        Args:
            prompt_file: Nom du fichier (ex: "code_quality.md").

        Returns:
            Contenu du fichier prompt.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
        """
        path = Path(__file__).parent.parent.parent / "prompts" / prompt_file
        return path.read_text(encoding="utf-8")

    def _make_cached_system_message(self) -> SystemMessage:
        """Retourne le SystemMessage avec cache_control Anthropic.

        Le system prompt est identique à chaque tour de la boucle ReAct.
        Le marquer en cache_control="ephemeral" force Anthropic à le stocker
        côté serveur : cache_creation_input_tokens au 1er tour, puis
        cache_read_input_tokens aux tours suivants (×10 moins cher).

        Returns:
            SystemMessage avec contenu structuré (liste de blocs).
        """
        return SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        """Estime le nombre de tokens dans une liste de messages.

        Méthode : somme des caractères de tous les contenus, divisée par 4
        (approximation Claude : ~4 chars = 1 token, précision ±10%).

        Inclut le contenu textuel, les tool_calls et les tool_results.

        Args:
            messages: Liste de messages LangChain (System/Human/AI/Tool).

        Returns:
            Estimation du nombre de tokens.
        """
        total_chars = 0
        for msg in messages:
            # Contenu principal (str ou liste de blocs)
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    total_chars += len(str(block))
            # Tool calls (AIMessage avec appels d'outils)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                total_chars += len(str(tool_calls))
        return total_chars // 4

    async def analyze(
        self,
        repo: str,
        run_id: str,
        architecture: str,
    ) -> AgentReport:
        """Exécute la boucle ReAct sur un repo et retourne un rapport structuré.

        Toute la durée de cette méthode se fait dans une session MCP persistante.
        Le contexte ContextVar est positionné avant l'ouverture de la session pour
        que les logs MCP portent les bons métadonnées de run.

        Args:
            repo: Repo audité, format "owner/name" (ex: "pallets/flask").
            run_id: Identifiant unique du run benchmark.
            architecture: "supervisor" | "hierarchical" | "decentralized".

        Returns:
            Rapport d'audit structuré.
        """
        _logger.info(
            "agent_start",
            agent=self.name,
            repository=repo,
            run_id=run_id,
            architecture=architecture,
        )

        # Positionner le contexte AVANT l'ouverture de session
        # (le ContextVar est lu par l'intercepteur session.call_tool)
        set_run_context(
            run_id=run_id,
            architecture=architecture,
            repository=repo,
            agent_name=self.name,
        )

        async with self.mcp_client.session("github") as session:
            tools = await self._instrumented.get_instrumented_tools(
                session, self.allowed_mcp_tools
            )
            model_with_tools = self.model.bind_tools(tools)

            # Cache les définitions d'outils MCP (identiques à chaque tour).
            # L'API Anthropic cache tout jusqu'au dernier outil marqué ephemeral.
            # On modifie directement la liste dans les kwargs du RunnableBinding.
            try:
                api_tools = list(model_with_tools.kwargs.get("tools", []))
                if api_tools:
                    last = dict(api_tools[-1])
                    last["cache_control"] = {"type": "ephemeral"}
                    api_tools[-1] = last
                    model_with_tools = model_with_tools.bind(tools=api_tools)
            except Exception as exc:
                _logger.warning("tool_cache_setup_failed", error=str(exc))

            graph = self._build_react_graph(model_with_tools, tools)

            owner, repo_name = repo.split("/", 1)
            initial_message = HumanMessage(
                content=(
                    f"Please audit the GitHub repository: {repo}\n"
                    f"Owner: {owner}, Repository: {repo_name}\n"
                    f"Use the available tools to gather evidence before concluding."
                )
            )

            result = await graph.ainvoke(
                {
                    "messages": [initial_message],
                    "repository": repo,
                    "iteration_count": 0,
                    "agent_report": None,
                    "context_limit_reached": False,
                }
            )

        report = result["agent_report"]
        if report is None:
            # Fallback si synthesize_node n'a pas pu produire un rapport
            _logger.warning("agent_no_report", agent=self.name, repository=repo)
            report = AgentReport(
                agent_name=self.name,
                repository=repo,
                findings=[],
                score=0.0,
                raw_data={},
            )

        _logger.info(
            "agent_done",
            agent=self.name,
            repository=repo,
            score=report.score,
            findings_count=len(report.findings),
            findings=[f.model_dump() for f in report.findings],
        )
        return report

    # ── Construction du graphe ─────────────────────────────────────────────────

    def _build_react_graph(
        self,
        model_with_tools: Any,
        tools: list[Any],
    ):
        """Construit et compile le graphe ReAct à 3 nœuds.

        Les closures capturent self, model_with_tools et tools.
        Le graphe est reconstruit à chaque appel d'analyze() car les outils
        sont spécifiques à la session MCP courante.

        Args:
            model_with_tools: ChatAnthropic avec tools bindés.
            tools: Liste d'outils LangChain (pour ToolNode).

        Returns:
            Graphe compilé prêt pour ainvoke().
        """
        # include_raw=True → renvoie {"raw": AIMessage, "parsed": AgentReportOutput}
        # nécessaire pour accéder à usage_metadata après la synthèse
        synthesis_model = self.model.with_structured_output(AgentReportOutput, include_raw=True)

        # Handler d'erreur pour ToolNode : LangGraph 1.1.x ne gère par défaut
        # que ToolInvocationError, pas ToolException (lancée par les tools MCP).
        # Ce handler convertit TOUTE exception en ToolMessage d'erreur,
        # permettant au LLM de continuer et de reporter l'échec gracieusement.
        def _mcp_tool_error_handler(e: Exception) -> str:
            return f"Tool call failed — {type(e).__name__}: {e}"

        tool_node = ToolNode(tools, handle_tool_errors=_mcp_tool_error_handler)
        agent = self  # pour les closures ci-dessous

        # ── Nœud 1 : appel LLM ────────────────────────────────────────────────

        async def model_node(state: AgentState) -> dict:
            """Appelle le LLM avec le system prompt + historique des messages.

            Vérifie la limite de contexte AVANT l'appel LLM. Si le prompt
            dépasse CONTEXT_TOKEN_LIMIT tokens (estimé), force la synthèse
            sans appeler le LLM et log l'événement pour analyse scientifique.
            """
            messages = [agent._make_cached_system_message()] + state["messages"]
            estimated_tokens = agent._estimate_tokens(messages)

            if estimated_tokens > agent.CONTEXT_TOKEN_LIMIT:
                _logger.info(
                    "context_limit_reached",
                    agent=agent.name,
                    repository=state["repository"],
                    estimated_tokens=estimated_tokens,
                    limit=agent.CONTEXT_TOKEN_LIMIT,
                    iteration_count=state["iteration_count"],
                )
                return {
                    "iteration_count": state["iteration_count"] + 1,
                    "context_limit_reached": True,
                }

            response = await model_with_tools.ainvoke(messages)

            # Log token usage pour mesure scientifique du coût
            usage = getattr(response, "usage_metadata", None)
            if usage:
                details = usage.get("input_token_details", {})
                _logger.info(
                    "llm_call",
                    agent=agent.name,
                    repository=state["repository"],
                    iteration=state["iteration_count"],
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=details.get("cache_read", 0),
                    cache_creation_tokens=details.get("cache_creation", 0),
                )

            return {
                "messages": [response],
                "iteration_count": state["iteration_count"] + 1,
            }

        # ── Nœud 2 : synthèse structurée ──────────────────────────────────────

        async def synthesize_node(state: AgentState) -> dict:
            """Appelle le LLM en mode structured_output pour produire l'AgentReport.

            C'est le seul appel LLM qui ne produit pas de tool_calls — il force
            la sortie dans le schéma AgentReportOutput (Pydantic).
            raw_data est extrait des ToolMessages de la conversation.
            """
            # Extraire les réponses MCP brutes de la conversation
            raw_data = {
                msg.tool_call_id: msg.content
                for msg in state["messages"]
                if isinstance(msg, ToolMessage)
            }

            try:
                synthesis_result = await synthesis_model.ainvoke(
                    [
                        agent._make_cached_system_message(),
                        *state["messages"],
                        HumanMessage(
                            content=(
                                "Based on all your research above, produce the final structured "
                                "audit report for this repository. Include specific findings with "
                                "evidence from the tool results, and assign an overall quality "
                                "score from 0 to 20."
                            )
                        ),
                    ]
                )
                # include_raw=True → {"raw": AIMessage, "parsed": AgentReportOutput}
                llm_output: AgentReportOutput = synthesis_result["parsed"]
                raw_ai_msg = synthesis_result.get("raw")

                # Log token usage de la synthèse
                usage = getattr(raw_ai_msg, "usage_metadata", None)
                if usage:
                    details = usage.get("input_token_details", {})
                    _logger.info(
                        "llm_call",
                        agent=agent.name,
                        repository=state["repository"],
                        iteration="synthesize",
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_read_tokens=details.get("cache_read", 0),
                        cache_creation_tokens=details.get("cache_creation", 0),
                    )

                report = AgentReport(
                    agent_name=agent.name,
                    repository=state["repository"],
                    findings=llm_output.findings,
                    score=llm_output.score,
                    raw_data=raw_data,
                )
            except Exception as exc:
                _logger.error(
                    "synthesize_failed",
                    agent=agent.name,
                    error=str(exc),
                )
                report = AgentReport(
                    agent_name=agent.name,
                    repository=state["repository"],
                    findings=[],
                    score=0.0,
                    raw_data=raw_data,
                )

            return {"agent_report": report}

        # ── Routage après model_node ───────────────────────────────────────────

        def should_continue(state: AgentState) -> Literal["tools", "synthesize"]:
            """Route vers tools si le LLM a fait des tool_calls, sinon synthèse.

            Ordre de priorité des gardes-fous :
            1. Limite de contexte atteinte (context_limit_reached)
            2. Nombre max d'itérations atteint (MAX_ITERATIONS)
            3. Pas de tool_calls → fin naturelle de la boucle ReAct
            """
            # Garde-fou 1 : limite de contexte (loggée dans model_node)
            if state.get("context_limit_reached"):
                return "synthesize"

            last_message = state["messages"][-1]

            # Garde-fou 2 : nombre max d'itérations
            if state["iteration_count"] >= agent.MAX_ITERATIONS:
                _logger.info(
                    "forced_synthesis_max_iterations",
                    agent=agent.name,
                    repository=state["repository"],
                    iteration_count=state["iteration_count"],
                    max_iterations=agent.MAX_ITERATIONS,
                )
                return "synthesize"

            # Route normale : tool_calls présents → continuer la boucle
            if last_message.tool_calls:
                return "tools"

            return "synthesize"

        # ── Assemblage du graphe ───────────────────────────────────────────────

        builder = StateGraph(AgentState)
        builder.add_node("model", model_node)
        builder.add_node("tools", tool_node)
        builder.add_node("synthesize", synthesize_node)

        builder.add_edge(START, "model")
        builder.add_conditional_edges(
            "model",
            should_continue,
            {"tools": "tools", "synthesize": "synthesize"},
        )
        builder.add_edge("tools", "model")
        builder.add_edge("synthesize", END)

        return builder.compile()
