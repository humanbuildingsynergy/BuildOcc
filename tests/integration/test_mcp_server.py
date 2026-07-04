"""
MCP server tests.

Tests every MCP tool by routing httpx calls through the FastAPI app via
httpx.ASGITransport — no real HTTP server or network access required.
Also verifies the tool registry (list_tools) and the full call_tool handler
including error paths.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from occupant_agent.mcp_server.server import _dispatch, call_tool, list_tools

# Save a reference to the real AsyncClient before any test patches it,
# so _asgi_client_factory can create real clients even while the mock is active.
_AsyncClient = httpx.AsyncClient

# ── Shared test data ──────────────────────────────────────────────────────────

_STEP_RESPONSE = {
    "action_type": "do_nothing",
    "target_id": None,
    "value": None,
    "reasoning": "No action needed.",
    "_memory_note": "Uneventful step.",
    "_importance": 2,
}

_SIGNAL_RESPONSE = {
    "response": "accepted",
    "reasoning": "Complying.",
    "_importance": 5,
}

_ENV = {
    "timestep": "2025-08-12T10:00:00Z",
    "zone_temp_c": 24.5,
    "outdoor_temp_c": 32.0,
    "tou_rate": 0.12,
    "thermostat_setpoint_c": 22.0,
    "devices": [{"device_id": "hvac", "state": True, "power_w": 3500}],
    "rooms": [{"room_id": "living_room", "occupied": True}],
}

BASE = "http://test"


# ── Async fixtures ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def rest_client():
    """httpx AsyncClient pointing at the FastAPI app via ASGI transport."""
    from occupant_agent.api.app import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def agent_id(rest_client):
    """Create a fresh agent for each test; delete it on teardown."""
    result = await _dispatch(rest_client, BASE, "initialize_agent", {"stratum": "O1", "seed": 42})
    aid = result["agent_id"]
    yield aid
    await _dispatch(rest_client, BASE, "reset_agent", {"agent_id": aid})


# ── Tool registry ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_tools_returns_five():
    tools = await list_tools()
    assert len(tools) == 5


@pytest.mark.asyncio
async def test_list_tools_names():
    tools = await list_tools()
    names = {t.name for t in tools}
    assert names == {"initialize_agent", "step", "send_signal", "get_state", "reset_agent"}


@pytest.mark.asyncio
async def test_list_tools_have_input_schema():
    tools = await list_tools()
    for tool in tools:
        assert tool.inputSchema is not None, f"{tool.name} missing inputSchema"
        assert "required" in tool.inputSchema or "properties" in tool.inputSchema


# ── _dispatch: all 5 tools ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_initialize_agent(rest_client):
    result = await _dispatch(rest_client, BASE, "initialize_agent", {"stratum": "O1", "seed": 0})
    assert "agent_id" in result
    assert result["stratum"] == "O1"
    # cleanup
    await _dispatch(rest_client, BASE, "reset_agent", {"agent_id": result["agent_id"]})


@pytest.mark.parametrize("stratum", ["O1", "O2", "O3", "O4"])
@pytest.mark.asyncio
async def test_dispatch_initialize_all_strata(rest_client, stratum):
    result = await _dispatch(rest_client, BASE, "initialize_agent", {"stratum": stratum, "seed": 7})
    assert result["stratum"] == stratum
    await _dispatch(rest_client, BASE, "reset_agent", {"agent_id": result["agent_id"]})


@pytest.mark.asyncio
async def test_dispatch_step(rest_client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_STEP_RESPONSE):
        result = await _dispatch(rest_client, BASE, "step", {
            "agent_id": agent_id,
            "environment_state": _ENV,
        })
    assert result["action_type"] in {"do_nothing", "adjust_thermostat", "toggle_device", "move_room"}


@pytest.mark.asyncio
async def test_dispatch_step_with_atus_code(rest_client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_STEP_RESPONSE):
        result = await _dispatch(rest_client, BASE, "step", {
            "agent_id": agent_id,
            "environment_state": _ENV,
            "atus_code": "030101",
        })
    assert "action_type" in result


@pytest.mark.asyncio
async def test_dispatch_step_with_extra_context(rest_client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_STEP_RESPONSE):
        result = await _dispatch(rest_client, BASE, "step", {
            "agent_id": agent_id,
            "environment_state": _ENV,
            "extra_context": "Today is a holiday.",
        })
    assert "action_type" in result


@pytest.mark.parametrize("sig_type", ["A", "B", "C"])
@pytest.mark.asyncio
async def test_dispatch_send_signal(rest_client, agent_id, sig_type):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_SIGNAL_RESPONSE):
        result = await _dispatch(rest_client, BASE, "send_signal", {
            "agent_id": agent_id,
            "signal_type": sig_type,
            "content": "Please reduce HVAC use during peak hours.",
            "environment_state": _ENV,
        })
    assert result["response"] in {"accepted", "rejected", "deferred"}
    assert "reasoning" in result


@pytest.mark.asyncio
async def test_dispatch_send_signal_with_atus_code(rest_client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_SIGNAL_RESPONSE):
        result = await _dispatch(rest_client, BASE, "send_signal", {
            "agent_id": agent_id,
            "signal_type": "B",
            "content": "This change saves money and reduces grid stress.",
            "environment_state": _ENV,
            "atus_code": "030101",
        })
    assert result["response"] in {"accepted", "rejected", "deferred"}


@pytest.mark.asyncio
async def test_dispatch_get_state(rest_client, agent_id):
    result = await _dispatch(rest_client, BASE, "get_state", {"agent_id": agent_id})
    assert result["stratum"] == "O1"
    assert "memory_count" in result
    assert "action_count" in result
    assert result["action_count"] == 0


@pytest.mark.asyncio
async def test_dispatch_get_state_reflects_step(rest_client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_STEP_RESPONSE):
        await _dispatch(rest_client, BASE, "step", {
            "agent_id": agent_id, "environment_state": _ENV,
        })
    result = await _dispatch(rest_client, BASE, "get_state", {"agent_id": agent_id})
    assert result["action_count"] >= 1
    assert result["memory_count"] >= 1


@pytest.mark.asyncio
async def test_dispatch_reset_agent(rest_client):
    init = await _dispatch(rest_client, BASE, "initialize_agent", {"stratum": "O2", "seed": 1})
    aid = init["agent_id"]
    result = await _dispatch(rest_client, BASE, "reset_agent", {"agent_id": aid})
    assert result["deleted"] is True

    # Agent should no longer exist
    try:
        await _dispatch(rest_client, BASE, "get_state", {"agent_id": aid})
        assert False, "Should have raised on deleted agent"
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 404


@pytest.mark.asyncio
async def test_dispatch_unknown_tool(rest_client):
    with pytest.raises(ValueError, match="Unknown tool"):
        await _dispatch(rest_client, BASE, "nonexistent_tool", {})


# ── _dispatch: 404 and error propagation ─────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_step_unknown_agent_raises(rest_client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _dispatch(rest_client, BASE, "step", {
            "agent_id": "does-not-exist",
            "environment_state": _ENV,
        })
    assert exc_info.value.response.status_code == 404


@pytest.mark.asyncio
async def test_dispatch_get_state_unknown_agent_raises(rest_client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _dispatch(rest_client, BASE, "get_state", {"agent_id": "ghost-id"})
    assert exc_info.value.response.status_code == 404


# ── call_tool: full handler including error wrapping ─────────────────────────
#
# call_tool() does `async with httpx.AsyncClient(...) as client:` internally.
# We can't pass an already-open client as return_value — httpx raises if you
# re-enter an active context manager. Instead we patch AsyncClient with a
# side_effect factory that returns a fresh ASGI-backed client each time.

def _asgi_client_factory(**kwargs):
    from occupant_agent.api.app import app
    return _AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE,
    )


@pytest.mark.asyncio
async def test_call_tool_initialize_returns_text_content():
    with patch("occupant_agent.mcp_server.server._api_url", return_value=BASE), \
         patch("occupant_agent.mcp_server.server.httpx.AsyncClient", side_effect=_asgi_client_factory):
        contents = await call_tool("initialize_agent", {"stratum": "O1", "seed": 99})

    assert len(contents) == 1
    assert contents[0].type == "text"
    data = json.loads(contents[0].text)
    assert "agent_id" in data
    # cleanup
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=__import__("occupant_agent.api.app", fromlist=["app"]).app),
        base_url=BASE,
    ) as c:
        await _dispatch(c, BASE, "reset_agent", {"agent_id": data["agent_id"]})


@pytest.mark.asyncio
async def test_call_tool_404_returns_error_json():
    """call_tool wraps HTTPStatusError and returns error JSON (never raises)."""
    with patch("occupant_agent.mcp_server.server._api_url", return_value=BASE), \
         patch("occupant_agent.mcp_server.server.httpx.AsyncClient", side_effect=_asgi_client_factory):
        contents = await call_tool("get_state", {"agent_id": "nonexistent"})

    assert len(contents) == 1
    data = json.loads(contents[0].text)
    assert "error" in data


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_returns_error_json():
    """Unknown tool name is caught by call_tool and returned as error JSON."""
    with patch("occupant_agent.mcp_server.server._api_url", return_value=BASE), \
         patch("occupant_agent.mcp_server.server.httpx.AsyncClient", side_effect=_asgi_client_factory):
        contents = await call_tool("mystery_tool", {})

    assert len(contents) == 1
    data = json.loads(contents[0].text)
    assert "error" in data
