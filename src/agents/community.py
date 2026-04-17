"""
Agent d'audit de santé communautaire.

Évalue : gestion des issues, activité des contributeurs, gouvernance,
engagement des discussions, cadence de releases.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.base_agent import BaseAuditAgent

#: Outils MCP autorisés pour cet agent
ALLOWED_TOOLS = [
    "list_issues",
    "get_issue",
    "list_commits",
    "get_file_contents",
    "get_repository",
    "get_repository_tree",
    "search_code",
]


class CommunityAgent(BaseAuditAgent):
    """Audite la santé communautaire d'un dépôt GitHub."""

    def __init__(self, model: ChatAnthropic, mcp_client: MultiServerMCPClient) -> None:
        super().__init__(
            name="community",
            system_prompt=self.load_prompt("community.md"),
            allowed_mcp_tools=ALLOWED_TOOLS,
            model=model,
            mcp_client=mcp_client,
        )
