"""
Tests pour InstrumentedMCPClient.

L'instrumentation se fait au niveau session.call_tool (pas tool.ainvoke),
ce qui évite les conflits avec Pydantic v2 (StructuredTool non monkey-patchable).

Tests :
1. _hash_params — déterminisme, ordre canonique, longueur
2. ContextVar — set/get, propagation dans async
3. _wrap_session_call_tool — log produit, champs requis, erreur loggée
4. InstrumentedMCPClient.get_instrumented_tools — filtrage, warning outil manquant
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.instrumentation.logger import setup_logging
from src.mcp.instrumented_client import (
    InstrumentedMCPClient,
    _hash_params,
    _wrap_session_call_tool,
    get_run_context,
    set_run_context,
)


@pytest.fixture(autouse=True)
def configure_logging():
    setup_logging()


def make_mock_session(call_tool_return=None) -> MagicMock:
    """Crée une session MCP mock avec call_tool async."""
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=call_tool_return or {"content": "ok"})
    return session


def make_mock_tool(name: str) -> MagicMock:
    """Crée un outil LangChain mock (pas Pydantic, donc patchable librement)."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value="result")
    return tool


# ── Tests : hashing ───────────────────────────────────────────────────────────


def test_hash_params_deterministic():
    params = {"owner": "pallets", "repo": "flask", "sha": "main"}
    assert _hash_params(params) == _hash_params(params)


def test_hash_params_canonical_order():
    p1 = {"owner": "pallets", "repo": "flask"}
    p2 = {"repo": "flask", "owner": "pallets"}
    assert _hash_params(p1) == _hash_params(p2)


def test_hash_params_different_values():
    assert _hash_params({"repo": "flask"}) != _hash_params({"repo": "django"})


def test_hash_params_length():
    h = _hash_params({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ── Tests : ContextVar ────────────────────────────────────────────────────────


def test_set_get_run_context():
    set_run_context(
        run_id="test-001",
        architecture="supervisor",
        repository="pallets/flask",
        agent_name="code_quality",
    )
    ctx = get_run_context()
    assert ctx["run_id"] == "test-001"
    assert ctx["architecture"] == "supervisor"
    assert ctx["repository"] == "pallets/flask"
    assert ctx["agent_name"] == "code_quality"


@pytest.mark.asyncio
async def test_context_propagates_in_async_task():
    """Le contexte est propagé dans la tâche asyncio courante."""
    set_run_context(
        run_id="async-ctx",
        architecture="hierarchical",
        repository="django/django",
        agent_name="security",
    )

    logged = {}
    mock_logger = MagicMock()
    mock_logger.info = lambda event, **kw: logged.update({"event": event, **kw})
    mock_logger.warning = MagicMock()

    session = make_mock_session()
    _wrap_session_call_tool(session, logger=mock_logger)

    await session.call_tool("search_code", {"query": "eval(", "owner": "django"})

    assert logged["run_id"] == "async-ctx"
    assert logged["architecture"] == "hierarchical"
    assert logged["agent_name"] == "security"


# ── Tests : _wrap_session_call_tool ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_wrap_logs_all_required_fields():
    """Un appel réussi produit un log avec tous les champs requis par le spec."""
    set_run_context(
        run_id="run-fields",
        architecture="supervisor",
        repository="pallets/flask",
        agent_name="community",
    )

    logged = {}
    mock_logger = MagicMock()
    mock_logger.info = lambda event, **kw: logged.update({"event": event, **kw})

    session = make_mock_session(call_tool_return='[{"sha": "abc123"}]')
    _wrap_session_call_tool(session, logger=mock_logger)

    await session.call_tool("list_commits", {"owner": "pallets", "repo": "flask", "sha": "main"})

    required = [
        "run_id", "architecture", "repository", "agent_name",
        "mcp_tool", "mcp_params_hash", "mcp_params",
        "response_size_bytes", "timestamp_start", "duration_ms",
        "success", "error",
    ]
    for field in required:
        assert field in logged, f"Champ manquant dans le log : {field}"

    assert logged["event"] == "mcp_call"
    assert logged["success"] is True
    assert logged["error"] is None
    assert logged["mcp_tool"] == "list_commits"
    assert logged["duration_ms"] >= 0
    assert logged["response_size_bytes"] > 0


@pytest.mark.asyncio
async def test_wrap_params_hash_is_correct():
    """Le mcp_params_hash correspond bien au hash des paramètres envoyés."""
    set_run_context(run_id="x", architecture="x", repository="x", agent_name="x")
    logged = {}
    mock_logger = MagicMock()
    mock_logger.info = lambda event, **kw: logged.update({"event": event, **kw})

    session = make_mock_session()
    _wrap_session_call_tool(session, logger=mock_logger)

    params = {"owner": "pallets", "repo": "flask", "sha": "main"}
    await session.call_tool("list_commits", params)

    assert logged["mcp_params_hash"] == _hash_params(params)


@pytest.mark.asyncio
async def test_wrap_logs_error_on_exception():
    """Une exception dans call_tool est loggée avec success=False."""
    set_run_context(run_id="x", architecture="x", repository="x", agent_name="x")
    logged = {}
    mock_logger = MagicMock()
    mock_logger.info = lambda event, **kw: logged.update({"event": event, **kw})

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=PermissionError("403 Forbidden"))
    _wrap_session_call_tool(session, logger=mock_logger)

    with pytest.raises(PermissionError):
        await session.call_tool("list_dependabot_alerts", {"owner": "facebook", "repo": "react"})

    assert logged["success"] is False
    assert "PermissionError" in logged["error"]
    assert "403" in logged["error"]


@pytest.mark.asyncio
async def test_wrap_same_params_same_hash():
    """Deux appels identiques produisent le même mcp_params_hash (détection redondance)."""
    set_run_context(run_id="x", architecture="x", repository="x", agent_name="x")
    hashes = []
    mock_logger = MagicMock()
    mock_logger.info = lambda event, **kw: hashes.append(kw.get("mcp_params_hash"))

    session = make_mock_session()
    _wrap_session_call_tool(session, logger=mock_logger)

    params = {"owner": "pallets", "repo": "flask", "sha": "main", "perPage": 3}
    await session.call_tool("list_commits", params)
    await session.call_tool("list_commits", params)

    assert len(hashes) == 2
    assert hashes[0] == hashes[1], "Deux appels identiques doivent avoir le même hash"


# ── Tests : InstrumentedMCPClient.get_instrumented_tools ─────────────────────


@pytest.mark.asyncio
async def test_get_instrumented_tools_filters_by_whitelist():
    """Seuls les outils de la whitelist sont retournés."""
    all_tools = [
        make_mock_tool("search_code"),
        make_mock_tool("list_commits"),
        make_mock_tool("get_file_contents"),
        make_mock_tool("list_issues"),
    ]

    with patch(
        "src.mcp.instrumented_client.load_mcp_tools",
        new=AsyncMock(return_value=all_tools),
    ):
        client = InstrumentedMCPClient()
        session = make_mock_session()
        result = await client.get_instrumented_tools(
            session,
            allowed_tools=["search_code", "get_file_contents"],
        )

    assert len(result) == 2
    assert {t.name for t in result} == {"search_code", "get_file_contents"}


@pytest.mark.asyncio
async def test_get_instrumented_tools_warns_on_missing():
    """Un outil absent produit un warning, pas une exception."""
    all_tools = [make_mock_tool("search_code")]

    mock_logger = MagicMock()
    mock_logger.warning = MagicMock()

    with patch(
        "src.mcp.instrumented_client.load_mcp_tools",
        new=AsyncMock(return_value=all_tools),
    ):
        client = InstrumentedMCPClient(logger=mock_logger)
        session = make_mock_session()
        result = await client.get_instrumented_tools(
            session,
            allowed_tools=["search_code", "list_alerts"],
        )

    assert len(result) == 1
    assert result[0].name == "search_code"
    mock_logger.warning.assert_called_once()
    call_kwargs = mock_logger.warning.call_args
    assert "list_alerts" in str(call_kwargs)


@pytest.mark.asyncio
async def test_get_instrumented_tools_no_filter():
    """Sans whitelist, tous les outils sont retournés."""
    all_tools = [make_mock_tool(f"tool_{i}") for i in range(5)]

    with patch(
        "src.mcp.instrumented_client.load_mcp_tools",
        new=AsyncMock(return_value=all_tools),
    ):
        client = InstrumentedMCPClient()
        session = make_mock_session()
        result = await client.get_instrumented_tools(session, allowed_tools=None)

    assert len(result) == 5


@pytest.mark.asyncio
async def test_session_call_tool_is_wrapped_after_get_instrumented_tools():
    """Après get_instrumented_tools, session.call_tool est bien remplacé."""
    all_tools = [make_mock_tool("search_code")]

    with patch(
        "src.mcp.instrumented_client.load_mcp_tools",
        new=AsyncMock(return_value=all_tools),
    ):
        client = InstrumentedMCPClient()
        session = make_mock_session()
        original_call_tool = session.call_tool
        await client.get_instrumented_tools(session, allowed_tools=["search_code"])

    # call_tool doit avoir été remplacé par notre wrapper
    assert session.call_tool is not original_call_tool
