"""
Vérification de l'instrumentation sur un vrai appel MCP (étape 3).

Fait 2 appels MCP identiques (list_commits) avec InstrumentedMCPClient,
vérifie que le fichier JSONL produit contient bien 2 entrées avec tous
les champs requis et le même mcp_params_hash.

Usage :
    uv run python experiments/test_instrumentation.py
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from src.instrumentation.logger import setup_logging
from src.mcp.github_client import build_github_mcp_client
from src.mcp.instrumented_client import InstrumentedMCPClient, set_run_context

LOG_FILE = "results/logs/test-instrumentation-001.jsonl"

REQUIRED_FIELDS = [
    "run_id", "architecture", "repository", "agent_name",
    "mcp_tool", "mcp_params_hash", "response_size_bytes",
    "timestamp_start", "duration_ms", "success",
]


async def main() -> None:
    token = os.environ["GITHUB_TOKEN"]
    binary = Path(__file__).parent.parent / "github-mcp-server"

    # Logs persistés dans le fichier JSONL dès maintenant
    setup_logging(log_file=LOG_FILE)

    print("=" * 60)
    print("TEST INSTRUMENTATION MCP — appel réel")
    print(f"Log file : {LOG_FILE}")
    print("=" * 60)

    set_run_context(
        run_id="test-instrumentation-001",
        architecture="supervisor",
        repository="pallets/flask",
        agent_name="community",
    )

    client_raw = build_github_mcp_client(token=token, binary_path=binary)
    instrumented = InstrumentedMCPClient()

    async with client_raw.session("github") as session:
        tools = await instrumented.get_instrumented_tools(
            session,
            allowed_tools=["list_commits"],
        )
        tool = tools[0]

        params = {"owner": "pallets", "repo": "flask", "sha": "main", "perPage": 2}

        print("\nAppel 1 : list_commits ...")
        await tool.ainvoke(params)
        print("  → OK")

        print("Appel 2 : list_commits (identique — test redondance) ...")
        await tool.ainvoke(params)
        print("  → OK")

    # ── Vérification du fichier JSONL produit ──────────────────────────────
    print("\n" + "=" * 60)
    print("VÉRIFICATION DU FICHIER JSONL")
    print("=" * 60)

    log_path = Path(__file__).parent.parent / LOG_FILE
    assert log_path.exists(), f"ÉCHEC : fichier introuvable : {log_path}"

    entries = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "mcp_call":
                entries.append(entry)

    print(f"Entrées mcp_call trouvées : {len(entries)}")
    assert len(entries) == 2, f"ÉCHEC : attendu 2 entrées, trouvé {len(entries)}"
    print("  ✓ 2 entrées mcp_call présentes")

    # Vérifier les champs requis sur chaque entrée
    for i, entry in enumerate(entries, 1):
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        assert not missing, f"ÉCHEC entrée {i} — champs manquants : {missing}"
    print(f"  ✓ Les 10 champs requis présents dans les 2 entrées")

    # Vérifier que les 2 hashes sont identiques (redondance détectable)
    h1, h2 = entries[0]["mcp_params_hash"], entries[1]["mcp_params_hash"]
    assert h1 == h2, f"ÉCHEC : hashes différents ({h1} ≠ {h2})"
    print(f"  ✓ mcp_params_hash identiques : {h1} (redondance détectable)")

    # Afficher le contenu complet des 2 entrées pour inspection
    print("\nEntrée 1 :")
    print(json.dumps(entries[0], indent=2, ensure_ascii=False))
    print("\nEntrée 2 (champs différenciants seulement) :")
    diff = {k: v for k, v in entries[1].items()
            if v != entries[0].get(k) or k in ("timestamp_start", "duration_ms", "timestamp")}
    print(json.dumps(diff, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Étape 3 validée. Fichier JSONL persisté et structuré correctement.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
