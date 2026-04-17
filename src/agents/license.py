"""
Agent d'audit de licence et conformité légale.

Évalue : présence de licence, déclaration dans les métadonnées,
cohérence des headers, compatibilité des dépendances, fichiers NOTICE/COPYING.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.base_agent import BaseAuditAgent

#: Outils MCP autorisés pour cet agent
ALLOWED_TOOLS = [
    "get_file_contents",
    "get_repository_tree",
    "get_repository",
    "search_code",
]


class LicenseAgent(BaseAuditAgent):
    """Audite la licence et la conformité légale d'un dépôt GitHub."""

    def __init__(self, model: ChatAnthropic, mcp_client: MultiServerMCPClient) -> None:
        super().__init__(
            name="license",
            system_prompt=self.load_prompt("license.md"),
            allowed_mcp_tools=ALLOWED_TOOLS,
            model=model,
            mcp_client=mcp_client,
        )
