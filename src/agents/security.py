"""
Agent d'audit de posture de sécurité.

Évalue : politique de sécurité, analyse statique CI, outils de vulnérabilités,
patterns dangereux dans le code, hygiène supply-chain.

Note : les tools list_*_alerts échouent systématiquement sur les repos publics
d'autrui (permissions admin requises). Le prompt instruis l'agent à pivoter
vers des signaux indirects (SECURITY.md, workflows CodeQL, search_code).
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.base_agent import BaseAuditAgent

#: Outils MCP autorisés pour cet agent.
#: Les tools d'alertes (list_dependabot_alerts, etc.) sont inclus intentionnellement
#: pour mesurer leur échec (403 → ToolMessage d'erreur → LLM pivote vers signaux indirects).
ALLOWED_TOOLS = [
    # Signaux directs (échoueront sur repos publics d'autrui — 403 attendu)
    "list_dependabot_alerts",
    "list_secret_scanning_alerts",
    "list_code_scanning_alerts",
    # Signaux indirects (accessibles publiquement)
    "get_file_contents",
    "search_code",
    "get_repository_tree",
    "get_repository",
    "list_workflows",
]


class SecurityAgent(BaseAuditAgent):
    """Audite la posture de sécurité d'un dépôt GitHub."""

    def __init__(self, model: ChatAnthropic, mcp_client: MultiServerMCPClient) -> None:
        super().__init__(
            name="security",
            system_prompt=self.load_prompt("security.md"),
            allowed_mcp_tools=ALLOWED_TOOLS,
            model=model,
            mcp_client=mcp_client,
        )
