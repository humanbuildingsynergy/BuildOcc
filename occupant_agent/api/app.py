"""
BuildOcc REST API — Layer 2 of the three-layer platform interface.

Exposes the Python agent library (Layer 1) over standard HTTP so any building
energy platform can integrate without bespoke coupling code.

Target integrations:
  - EnergyPlus: Python callback calls POST /agents/{id}/step at each timestep
  - VOLTTRON / OpenStudio: HTTP requests from their Python SDKs
  - Research scripts: simple curl or httpx calls

The Layer 3 MCP server (occupant_agent/mcp_server/server.py) is a thin wrapper
over this API.

Run:
    uvicorn occupant_agent.api.app:app --reload --port 8000
    # or via CLI entry point:
    buildocc-api

Endpoints:
    POST   /agents/initialize              → {agent_id}
    POST   /agents/{agent_id}/step         → AgentAction
    POST   /agents/{agent_id}/signal       → SignalResponse
    GET    /agents/{agent_id}/state        → state summary
    GET    /agents/                        → list of all agents
    DELETE /agents/{agent_id}              → {deleted: true}
    GET    /health                         → {status, version}
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.environment.state import AgentAction, EnvironmentState, SignalResponse
from occupant_agent.persistence.store import AgentNotFoundError, AgentStore

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BuildOcc REST API",
    version="1.0.0",
    description=(
        "ATUS-grounded LLM occupant agents for building energy simulation. "
        "Layer 2 of the three-layer platform interface."
    ),
)

_API_VERSION = "1.0.0"


def _store() -> AgentStore:
    """Create a new AgentStore per request (SQLAlchemy handles connection pooling)."""
    db_path = os.getenv("OCCUPANT_AGENT_DB", "./occupant_agent.db")
    return AgentStore(db_path=db_path)


# ── Request/Response models ───────────────────────────────────────────────────

class InitializeRequest(BaseModel):
    stratum: Literal["O1", "O2", "O3", "O4"]
    seed: int | None = None
    llm_provider: str = "anthropic"
    llm_model: str | None = None
    state_fips: int = 48  # Texas (Pecan Street cohort)


class InitializeResponse(BaseModel):
    agent_id: str
    stratum: str
    seed: int | None


class StepRequest(BaseModel):
    environment: EnvironmentState
    atus_code: str | None = None
    extra_context: str | None = None
    wfh_today: bool | None = None


class SignalRequest(BaseModel):
    signal_type: Literal["A", "B", "C"]
    content: str
    environment: EnvironmentState
    atus_code: str | None = None
    extra_context: str | None = None


class AgentStateResponse(BaseModel):
    agent_id: str
    stratum: str
    age: int
    work_from_home: bool
    home_gym: bool
    llm_provider: str
    memory_count: int
    action_count: int
    last_reflected_at: str | None
    last_action: dict[str, Any] | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": _API_VERSION}


@app.post("/agents/initialize", response_model=InitializeResponse)
async def initialize(request: InitializeRequest) -> InitializeResponse:
    """
    Create a new OccupantAgent and return its agent_id.

    The persona is sampled from the given ATUS stratum using the provided seed
    (or randomly if seed is None). The agent is persisted to SQLite immediately.
    """
    store = _store()
    agent = OccupantAgent.from_stratum(
        stratum=request.stratum,
        seed=request.seed,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        state_fips=request.state_fips,
    )
    agent_id = store.create_agent(agent, seed=request.seed)
    return InitializeResponse(
        agent_id=agent_id,
        stratum=request.stratum,
        seed=request.seed,
    )


@app.post("/agents/{agent_id}/step", response_model=AgentAction)
async def step(agent_id: str, request: StepRequest) -> AgentAction:
    """
    Advance the agent one 15-minute timestep.

    Loads the agent from SQLite, calls step(), persists the new memory and
    action, then returns the AgentAction.
    """
    store = _store()
    agent = _load_or_404(store, agent_id)

    try:
        action = agent.step(
            request.environment,
            atus_code=request.atus_code,
            extra_context=request.extra_context,
            wfh_today=request.wfh_today,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    store.sync_memories(agent_id, agent.memory, last_thermostat_step=agent._last_thermostat_step)
    store.save_step(agent_id, action, request.environment, request.atus_code, extra_context=request.extra_context)

    return action


@app.post("/agents/{agent_id}/signal", response_model=SignalResponse)
async def signal(agent_id: str, request: SignalRequest) -> SignalResponse:
    """
    Deliver a building control signal to the agent and return its response.

    Signal types:
      A — Direct command (e.g., "Turn off the dishwasher until 9pm")
      B — Competence-building / boost (explains the reason)
      C — Social norm / nudge (comparison to neighbors)
    """
    store = _store()
    agent = _load_or_404(store, agent_id)

    try:
        response = agent.receive_signal(
            signal_type=request.signal_type,
            content=request.content,
            env=request.environment,
            atus_code=request.atus_code,
            extra_context=request.extra_context,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    store.sync_memories(agent_id, agent.memory, last_thermostat_step=agent._last_thermostat_step)
    store.save_signal(
        agent_id, request.signal_type, request.content, response, request.environment,
        atus_code=request.atus_code, extra_context=request.extra_context,
    )

    return response


@app.get("/agents/{agent_id}/state", response_model=AgentStateResponse)
async def get_state(agent_id: str) -> AgentStateResponse:
    """Return the agent's current state summary (no LLM call)."""
    store = _store()
    agent = _load_or_404(store, agent_id)
    state = agent.get_state()
    return AgentStateResponse(
        agent_id=agent_id,
        stratum=state["stratum"],
        age=state["age"],
        work_from_home=state["work_from_home"],
        home_gym=state["home_gym"],
        llm_provider=state["llm_provider"],
        memory_count=state["memory_count"],
        action_count=state["action_count"],
        last_reflected_at=state["last_reflected_at"],
        last_action=state["last_action"],
    )


@app.get("/agents/")
async def list_agents() -> list[dict[str, Any]]:
    """Return a summary list of all agents in the database."""
    store = _store()
    return store.list_agents()


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict[str, Any]:
    """Delete an agent and all associated records."""
    store = _store()
    deleted = store.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return {"deleted": True, "agent_id": agent_id}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_404(store: AgentStore, agent_id: str) -> OccupantAgent:
    try:
        return store.load_agent(agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def run() -> None:
    """Entry point for the `buildocc-api` CLI command."""
    import uvicorn
    uvicorn.run("occupant_agent.api.app:app", host="0.0.0.0", port=8000, reload=False)
