"""
Étape 2 — Exploration MCP pure.

Objectifs :
1. Lister TOUS les outils disponibles sur le GitHub MCP server officiel (v0.33)
2. Vérifier quels outils du spec existent réellement
3. Appeler search_repositories + list_commits sur un repo public
4. Afficher les résultats bruts (aucun agent, aucun LLM)

Usage :
    uv run python experiments/explore_mcp.py

Prérequis :
    - Fichier .env avec GITHUB_TOKEN=ghp_...
    - Binaire ./github-mcp-server présent (voir README)

Note architecture :
    On utilise client.session() (session persistante) pour toute l'exploration,
    afin d'éviter les problèmes de reconnexion en mode stateless avec --toolsets=all.
    Pour le benchmark réel, on choisira la granularité adaptée à chaque architecture.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv(Path(__file__).parent.parent / ".env")


def get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERREUR : GITHUB_TOKEN manquant dans .env")
        sys.exit(1)
    return token


def get_mcp_binary_path() -> str:
    binary = Path(__file__).parent.parent / "github-mcp-server"
    if not binary.exists():
        print(f"ERREUR : binaire MCP introuvable à {binary}")
        sys.exit(1)
    return str(binary)


def build_mcp_client(token: str, binary_path: str) -> MultiServerMCPClient:
    """Client MCP pour exploration — tous les toolsets, lecture seule."""
    return MultiServerMCPClient(
        {
            "github": {
                "transport": "stdio",
                "command": binary_path,
                "args": ["stdio", "--toolsets=all", "--read-only"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": token},
            }
        }
    )


def display_result(result) -> None:
    raw = result if isinstance(result, str) else str(result)
    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=2, ensure_ascii=False)[:3000])
    except (json.JSONDecodeError, TypeError):
        print(raw[:3000])


def check_whitelist(tool_names: set[str]) -> None:
    """Compare la whitelist du spec avec les outils réels du serveur v0.33."""
    print("\n--- Mapping spec → outils réels (v0.33) ---")
    spec = [
        ("get_repository",            "community / license"),
        ("search_code",               "code_quality"),
        ("list_workflows",            "code_quality"),
        ("get_file_contents",         "code_quality / doc / license"),
        ("list_pull_requests",        "code_quality"),
        ("list_commits",              "community"),
        ("list_contributors",         "community"),
        ("list_issues",               "community"),
        ("list_alerts",               "security"),
        ("list_dependabot_alerts",    "security"),
        ("list_secret_scanning_alerts", "security"),
    ]
    for target, agent in spec:
        found = target in tool_names
        keywords = [p for p in target.replace("list_", "").replace("get_", "").split("_") if len(p) > 3]
        close = sorted(n for n in tool_names if any(kw in n for kw in keywords))[:3]
        status = "OK" if found else f"→ alternatives: {close}" if close else "ABSENT"
        print(f"  {'✓' if found else '✗'} {target:40s} [{agent}]  {status}")


async def main() -> None:
    token = get_github_token()
    binary_path = get_mcp_binary_path()
    print(f"Token GitHub : {token[:8]}... (OK)")
    print(f"Binaire MCP  : {binary_path}\n")

    client = build_mcp_client(token, binary_path)

    # ── Session persistante : toutes les opérations dans le même process MCP ──
    async with client.session("github") as session:
        # 1. Charger tous les outils de la session
        print("=" * 60)
        print("OUTILS MCP DISPONIBLES (--toolsets=all)")
        print("=" * 60)

        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        print(f"Total : {len(tools)} outil(s)\n")

        for i, tool in enumerate(tools, 1):
            desc = (getattr(tool, "description", "") or "").strip().split("\n")[0][:78]
            print(f"{i:3}. {tool.name}")
            if desc:
                print(f"      {desc}")

        check_whitelist(set(tools_by_name))

        # 2. search_repositories (remplace get_repository absent en v0.33)
        print("\n" + "=" * 60)
        print("APPEL MCP : search_repositories(repo:pallets/flask)")
        print("(get_repository absent en v0.33 → search_repositories)")
        print("=" * 60)
        result = await tools_by_name["search_repositories"].ainvoke(
            {"query": "repo:pallets/flask"}
        )
        display_result(result)

        # 3. list_commits (outil community agent)
        print("\n" + "=" * 60)
        print("APPEL MCP : list_commits(pallets/flask, perPage=3)")
        print("=" * 60)
        result = await tools_by_name["list_commits"].ainvoke(
            {"owner": "pallets", "repo": "flask", "sha": "main", "perPage": 3}
        )
        display_result(result)

    print("\n" + "=" * 60)
    print("Exploration terminée. Whitelist à corriger avant étape 3.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
