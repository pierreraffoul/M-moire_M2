"""
Client MCP instrumenté — composant de mesure scientifique central.

Intercepte chaque appel MCP au niveau de session.call_tool (MCP SDK),
avant que les outils LangChain ne soient invoqués. Cette approche évite
tout problème de monkey-patch sur les StructuredTool Pydantic v2.

Propagation du contexte :
    Via contextvars.ContextVar (task-local asyncio). La tâche appelante
    appelle set_run_context() avant d'entrer dans la session MCP.
    ContextVar garantit l'isolation entre agents exécutés en parallèle.

Pourquoi pas tool_interceptors de langchain-mcp-adapters ?
    Ils nécessitent request.runtime qui n'existe qu'avec create_agent.
    En StateGraph pur, runtime est None → ContextVar est la seule solution.

Format de log produit pour chaque appel MCP :
    {
      "event": "mcp_call",
      "run_id": "abc",
      "architecture": "supervisor",
      "repository": "pallets/flask",
      "agent_name": "code_quality",
      "mcp_tool": "search_code",
      "mcp_params_hash": "a3f2b1c4...",
      "mcp_params": {...},
      "response_size_bytes": 1843,
      "timestamp_start": "2026-04-16T10:00:00.000Z",
      "duration_ms": 845,
      "success": true,
      "error": null
    }
"""

import hashlib
import json
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from langchain_mcp_adapters.tools import load_mcp_tools

from src.instrumentation.logger import get_logger

# ── Contexte de propagation ────────────────────────────────────────────────────

_run_context: ContextVar[dict[str, str]] = ContextVar(
    "mcp_run_context",
    default={
        "run_id": "unknown",
        "architecture": "unknown",
        "repository": "unknown",
        "agent_name": "unknown",
    },
)


def set_run_context(
    *,
    run_id: str,
    architecture: str,
    repository: str,
    agent_name: str,
) -> None:
    """Définit le contexte MCP pour la tâche asyncio courante.

    À appeler depuis le nœud du graphe, avant l'invocation des tools.
    Chaque tâche asyncio a son propre ContextVar — pas de contamination
    entre agents concurrents.

    Args:
        run_id: Identifiant unique du run (ex: "supervisor_react_001").
        architecture: "supervisor" | "hierarchical" | "decentralized".
        repository: Repo audité, format "owner/name".
        agent_name: Nom de l'agent courant (ex: "code_quality").
    """
    _run_context.set(
        {
            "run_id": run_id,
            "architecture": architecture,
            "repository": repository,
            "agent_name": agent_name,
        }
    )


def get_run_context() -> dict[str, str]:
    """Retourne le contexte MCP de la tâche asyncio courante."""
    return _run_context.get()


# ── Hashing canonique des paramètres ──────────────────────────────────────────


def _hash_params(params: dict | Any) -> str:
    """Calcule un hash sha256 canonique des paramètres d'un appel MCP.

    Utilisé pour détecter les appels redondants : même hash = même requête.

    Args:
        params: Paramètres de l'appel.

    Returns:
        16 premiers caractères hex du sha256.
    """
    try:
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = str(params)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ── Instrumentation au niveau session ─────────────────────────────────────────


def _wrap_session_call_tool(session: Any, logger=None) -> None:
    """Intercepte session.call_tool pour logger chaque appel MCP.

    Modifie la session en place : remplace call_tool par une version
    instrumentée. Fonctionne avec mcp.ClientSession (pas Pydantic),
    donc le monkey-patch simple est valide ici.

    Args:
        session: Session MCP active (mcp.ClientSession).
        logger: Logger structlog. Si None, utilise le logger "mcp".
    """
    if logger is None:
        logger = get_logger("mcp")

    original_call_tool = session.call_tool

    async def instrumented_call_tool(name: str, arguments: dict | None = None, **kwargs) -> Any:
        ctx = get_run_context()
        params = arguments or {}
        params_hash = _hash_params(params)
        timestamp_start = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        error_msg = None
        result = None

        try:
            result = await original_call_tool(name, arguments, **kwargs)
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)

            response_size = 0
            if result is not None:
                try:
                    response_size = len(json.dumps(result, default=str).encode("utf-8"))
                except Exception:
                    response_size = len(str(result).encode("utf-8"))

            logger.info(
                "mcp_call",
                run_id=ctx["run_id"],
                architecture=ctx["architecture"],
                repository=ctx["repository"],
                agent_name=ctx["agent_name"],
                mcp_tool=name,
                mcp_params_hash=params_hash,
                mcp_params=params,
                response_size_bytes=response_size,
                timestamp_start=timestamp_start,
                duration_ms=duration_ms,
                success=error_msg is None,
                error=error_msg,
            )

    session.call_tool = instrumented_call_tool


# ── Client instrumenté ────────────────────────────────────────────────────────


class InstrumentedMCPClient:
    """Fournit des outils MCP instrumentés depuis une session persistante.

    Usage dans BaseAuditAgent :

        async with raw_mcp_client.session("github") as session:
            tools = await self.instrumented_client.get_instrumented_tools(
                session, allowed_tools=["search_code", "get_file_contents"]
            )
            # Tous les appels via ces tools sont loggés automatiquement.
    """

    def __init__(self, logger=None) -> None:
        self._logger = logger or get_logger("mcp")

    async def get_instrumented_tools(
        self,
        session: Any,
        allowed_tools: list[str] | None = None,
    ) -> list[Any]:
        """Installe l'instrumentation sur la session et retourne les outils filtrés.

        L'instrumentation est installée UNE FOIS sur la session (via call_tool).
        Tous les appels ultérieurs aux outils de cette session seront loggés,
        quel que soit l'outil appelé.

        Args:
            session: Session MCP active (obtenue via client.session("github")).
            allowed_tools: Whitelist de noms d'outils. None = tous les outils.
                           Noms absents → warning, pas d'exception.

        Returns:
            Liste d'outils LangChain (StructuredTool), filtrés et instrumentés
            via la session sous-jacente.
        """
        # 1. Instrumenter la session AVANT de charger les outils
        _wrap_session_call_tool(session, self._logger)

        # 2. Charger les outils depuis la session instrumentée
        all_tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in all_tools}

        # 3. Filtrer selon la whitelist
        if allowed_tools is None:
            return list(tools_by_name.values())

        selected = []
        for name in allowed_tools:
            if name in tools_by_name:
                selected.append(tools_by_name[name])
            else:
                self._logger.warning(
                    "mcp_tool_not_found",
                    requested_tool=name,
                    available_tools=sorted(tools_by_name.keys()),
                )
        return selected
