"""
Wrapper de configuration pour le GitHub MCP server officiel.

Ce module fournit une factory qui construit un MultiServerMCPClient
correctement configuré pour le serveur github-mcp-server v0.33+.

Choix de conception :
    - Session persistante (client.session) obligatoire : le mode stateless
      (client.get_tools()) crashe avec --toolsets=all (BrokenResourceError
      sur le teardown du subprocess — découvert à l'étape 2).
    - --read-only activé : sécurité ; le benchmark ne doit jamais écrire.
    - Le toolset est spécifié par l'appelant (agent ou exploration).

Usage :
    from src.mcp.github_client import build_github_mcp_client

    client = build_github_mcp_client(token="ghp_...", binary_path="./github-mcp-server")

    async with client.session("github") as session:
        # session prête pour load_mcp_tools(session)
        ...
"""

from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


def build_github_mcp_client(
    token: str,
    binary_path: str | Path,
    toolsets: str = "all",
) -> MultiServerMCPClient:
    """Construit un MultiServerMCPClient pour le GitHub MCP server.

    Args:
        token: GitHub Personal Access Token (GITHUB_TOKEN depuis .env).
        binary_path: Chemin vers le binaire github-mcp-server.
        toolsets: Toolsets à activer. Défaut "all" pour l'exploration et les tests.
                  En production, utiliser uniquement les toolsets nécessaires
                  (ex: "repos,issues,code_security,dependabot,secret_protection,actions").

    Returns:
        Client MCP configuré, prêt pour client.session("github").

    Raises:
        FileNotFoundError: Si le binaire n'existe pas au chemin donné.
    """
    binary = Path(binary_path)
    if not binary.exists():
        raise FileNotFoundError(
            f"Binaire GitHub MCP introuvable : {binary}\n"
            "Télécharger depuis : https://github.com/github/github-mcp-server/releases"
        )

    return MultiServerMCPClient(
        {
            "github": {
                "transport": "stdio",
                "command": str(binary),
                # "stdio" : sous-commande qui démarre le serveur MCP
                # "--toolsets" : liste des groupes d'outils à activer
                # "--read-only" : refuse toute opération d'écriture (sécurité)
                "args": ["stdio", f"--toolsets={toolsets}", "--read-only"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": token,
                },
            }
        }
    )


# Toolsets nécessaires pour le benchmark complet (tous agents confondus)
# Utiliser ce preset en production pour éviter de charger les outils inutiles.
BENCHMARK_TOOLSETS = ",".join([
    "repos",            # search_repositories, get_repository_tree, get_file_contents
    "git",              # list_commits, get_commit
    "issues",           # list_issues, issue_read
    "pull_requests",    # list_pull_requests, pull_request_read
    "code_security",    # list_code_scanning_alerts, get_code_scanning_alert
    "dependabot",       # list_dependabot_alerts, get_dependabot_alert
    "secret_protection", # list_secret_scanning_alerts, get_secret_scanning_alert
    "actions",          # actions_list, actions_get, get_job_logs
    "discussions",      # list_discussions, list_discussion_categories
    "users",            # search_users, get_me
])
