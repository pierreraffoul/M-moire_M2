"""
Agent d'audit de qualité de code.

Évalue : CI/CD, patterns code, hygiène PR, structure repo, dépendances.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.base_agent import BaseAuditAgent

#: Outils MCP autorisés pour cet agent
ALLOWED_TOOLS = [
    "get_file_contents",
    "search_code",
    "list_commits",
    "list_pull_requests",
    "list_workflows",
    "get_repository",
    "get_repository_tree",
    "list_branches",
]


class CodeQualityAgent(BaseAuditAgent):
    """Audite la qualité de code d'un dépôt GitHub."""

    def __init__(self, model: ChatAnthropic, mcp_client: MultiServerMCPClient) -> None:
        super().__init__(
            name="code_quality",
            system_prompt=self.load_prompt("code_quality.md"),
            allowed_mcp_tools=ALLOWED_TOOLS,
            model=model,
            mcp_client=mcp_client,
        )
