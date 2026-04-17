"""
Agent d'audit de qualité de documentation.

Évalue : qualité du README, documentation API/code, changelog,
guide de contribution, exemples d'utilisation.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.base_agent import BaseAuditAgent

#: Outils MCP autorisés pour cet agent
ALLOWED_TOOLS = [
    "get_file_contents",
    "get_repository_tree",
    "search_code",
    "get_repository",
    "list_commits",
]


class DocumentationAgent(BaseAuditAgent):
    """Audite la qualité de documentation d'un dépôt GitHub."""

    def __init__(self, model: ChatAnthropic, mcp_client: MultiServerMCPClient) -> None:
        super().__init__(
            name="documentation",
            system_prompt=self.load_prompt("documentation.md"),
            allowed_mcp_tools=ALLOWED_TOOLS,
            model=model,
            mcp_client=mcp_client,
        )
