"""
BuildOcc MCP Server — Layer 3 of the three-layer platform interface.

A thin wrapper over the REST API (Layer 2) that exposes BuildOcc as MCP
tools for LLM-orchestrated building management systems, such as:
  - Home Assistant with MCP client support
  - Claude-based building management systems
  - Any LLM host that supports the Model Context Protocol

This layer contains NO business logic — all operations delegate to the REST
API via httpx. The REST API (occupant_agent/api/app.py) must be running before
starting the MCP server.

Environment variables:
  BUILDOCC_API_URL   REST API base URL (default: http://localhost:8000)

Run:
    # First start the REST API:
    buildocc-api
    # or: uvicorn occupant_agent.api.app:app --port 8000

    # Then start the MCP server (stdio transport):
    BUILDOCC_API_URL=http://localhost:8000 buildocc-mcp
    # or: python -m occupant_agent.mcp_server.server

MCP Tools exposed:
    initialize_agent   — create a new agent, returns agent_id
    step               — advance one 15-min timestep, returns action
    send_signal        — deliver A/B/C signal, returns response
    get_state          — return agent state summary
    reset_agent        — delete agent and all associated records
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# ── Configuration ─────────────────────────────────────────────────────────────

def _api_url() -> str:
    return os.getenv("BUILDOCC_API_URL", "http://localhost:8000").rstrip("/")


# ── Server ────────────────────────────────────────────────────────────────────

server = Server("buildocc")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="initialize_agent",
            description=(
                "Create a new OccupantAgent from an ATUS demographic stratum. "
                "Returns an agent_id that must be passed to all subsequent calls."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "stratum": {
                        "type": "string",
                        "enum": ["O1", "O2", "O3", "O4"],
                        "description": (
                            "O1=Employed single 25-44, O2=Retired couple 65+, "
                            "O3=Employed parent 35-54, O4=Unemployed 25-44"
                        ),
                    },
                    "seed": {
                        "type": ["integer", "null"],
                        "description": "RNG seed for reproducible persona sampling (null = random).",
                        "default": None,
                    },
                    "llm_provider": {
                        "type": "string",
                        "enum": ["anthropic", "openai", "google", "ollama"],
                        "description": "anthropic=claude-haiku, openai=gpt-4o-mini, google=gemini-2.0-flash, ollama=llama3.2 (local)",
                        "default": "anthropic",
                    },
                },
                "required": ["stratum"],
            },
        ),
        types.Tool(
            name="step",
            description=(
                "Advance the agent one 15-minute timestep and return its action. "
                "Call this once per simulation timestep with the current environment state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "environment_state": {
                        "type": "object",
                        "description": (
                            "EnvironmentState v1.0: {timestep (ISO 8601), "
                            "zone_temp_c, outdoor_temp_c, tou_rate, "
                            "devices: [{device_id, state, power_w}], "
                            "rooms: [{room_id, occupied}]}"
                        ),
                    },
                    "atus_code": {
                        "type": ["string", "null"],
                        "description": "6-digit ATUS activity code for this timestep (optional).",
                        "default": None,
                    },
                    "extra_context": {
                        "type": "string",
                        "description": (
                            "Optional situational note injected into the agent's reasoning prompt "
                            "(e.g., 'Today is a holiday', 'Air quality AQI 180'). "
                            "Not stored in the returned AgentAction."
                        ),
                    },
                },
                "required": ["agent_id", "environment_state"],
            },
        ),
        types.Tool(
            name="send_signal",
            description=(
                "Deliver a building control signal to the agent and return its response "
                "(accepted / rejected / deferred). "
                "Signal types: A=direct command, B=educational/boost, C=social norm/nudge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "signal_type": {
                        "type": "string",
                        "enum": ["A", "B", "C"],
                        "description": "A=direct command, B=competence-building, C=social norm",
                    },
                    "content": {
                        "type": "string",
                        "description": "The signal message text shown to the occupant.",
                    },
                    "environment_state": {
                        "type": "object",
                        "description": "Current EnvironmentState (same schema as step).",
                    },
                    "atus_code": {
                        "type": ["string", "null"],
                        "description": "6-digit ATUS activity code for this timestep (optional). Adds activity context to the signal response.",
                        "default": None,
                    },
                    "extra_context": {
                        "type": "string",
                        "description": (
                            "Optional situational note injected into the agent's reasoning prompt "
                            "(e.g., 'Occupant is hosting guests'). "
                            "Not stored in the returned SignalResponse."
                        ),
                    },
                },
                "required": ["agent_id", "signal_type", "content", "environment_state"],
            },
        ),
        types.Tool(
            name="get_state",
            description=(
                "Return the agent's current state summary: stratum, memory count, "
                "action count, last action, and last reflection time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            },
        ),
        types.Tool(
            name="reset_agent",
            description="Delete an agent and all associated records from the database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Route tool calls to the REST API and return results as TextContent.

    All business logic lives in the REST API layer. This layer only handles
    routing and HTTP transport.
    """
    base = _api_url()
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            result = await _dispatch(client, base, name, arguments)
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"HTTP {exc.response.status_code}", "detail": error_body}),
            )]
        except Exception as exc:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": str(exc)}),
            )]

    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2),
    )]


async def _dispatch(
    client: httpx.AsyncClient,
    base: str,
    name: str,
    args: dict[str, Any],
) -> Any:
    """Route a tool call to the appropriate REST endpoint."""

    if name == "initialize_agent":
        resp = await client.post(f"{base}/agents/initialize", json={
            "stratum": args["stratum"],
            "seed": args.get("seed"),
            "llm_provider": args.get("llm_provider", "anthropic"),
        })
        resp.raise_for_status()
        return resp.json()

    elif name == "step":
        body: dict[str, Any] = {
            "environment": args["environment_state"],
            "atus_code": args.get("atus_code"),
        }
        if args.get("extra_context") is not None:
            body["extra_context"] = args["extra_context"]
        resp = await client.post(f"{base}/agents/{args['agent_id']}/step", json=body)
        resp.raise_for_status()
        return resp.json()

    elif name == "send_signal":
        body = {
            "signal_type": args["signal_type"],
            "content": args["content"],
            "environment": args["environment_state"],
        }
        if args.get("atus_code") is not None:
            body["atus_code"] = args["atus_code"]
        if args.get("extra_context") is not None:
            body["extra_context"] = args["extra_context"]
        resp = await client.post(
            f"{base}/agents/{args['agent_id']}/signal",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    elif name == "get_state":
        resp = await client.get(f"{base}/agents/{args['agent_id']}/state")
        resp.raise_for_status()
        return resp.json()

    elif name == "reset_agent":
        resp = await client.delete(f"{base}/agents/{args['agent_id']}")
        resp.raise_for_status()
        return resp.json()

    else:
        raise ValueError(f"Unknown tool: {name!r}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main_sync() -> None:
    """Synchronous entry point for the `buildocc-mcp` CLI command."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
