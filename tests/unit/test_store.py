from __future__ import annotations

from datetime import datetime

import pytest

from occupant_agent.persistence.store import AgentNotFoundError, AgentStore
from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.environment.state import AgentAction, DeviceState, EnvironmentState, RoomState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return OccupantAgent.from_stratum("O1", seed=42)


@pytest.fixture
def store():
    return AgentStore(db_path=":memory:")


# ── Shared helper ─────────────────────────────────────────────────────────────

def _make_env() -> EnvironmentState:
    return EnvironmentState(
        timestep=datetime(2024, 7, 15, 14, 0),
        zone_temp_c=22.0,
        outdoor_temp_c=35.0,
        tou_rate=0.12,
        devices=[DeviceState(device_id="hvac", state=True, power_w=2000.0)],
        rooms=[RoomState(room_id="living_room", occupied=True)],
    )


# ── create_agent ──────────────────────────────────────────────────────────────

def test_create_agent_returns_id(store, agent):
    """create_agent() returns a non-empty UUID string."""
    agent_id = store.create_agent(agent, seed=42)

    assert isinstance(agent_id, str)
    assert len(agent_id) > 0
    # UUID4 format: 8-4-4-4-12 hex groups separated by hyphens
    parts = agent_id.split("-")
    assert len(parts) == 5


# ── load_agent ────────────────────────────────────────────────────────────────

def test_load_agent_restores_stratum(store, agent):
    """load_agent() reconstructs the agent with the correct persona.stratum."""
    agent_id = store.create_agent(agent, seed=42)

    loaded = store.load_agent(agent_id)

    assert loaded.persona.stratum == "O1"


def test_load_agent_not_found_raises(store):
    """load_agent() raises AgentNotFoundError for an unknown id."""
    with pytest.raises(AgentNotFoundError):
        store.load_agent("nonexistent-id")


# ── sync_memories ─────────────────────────────────────────────────────────────

def test_sync_memories_persists_entries(store, agent):
    """After adding 3 entries and syncing, a freshly loaded agent has 3 memories."""
    agent_id = store.create_agent(agent, seed=42)

    sim_time = datetime(2024, 7, 15, 14, 0)
    agent.memory.add("First observation.", importance=5.0, memory_type="observation", sim_time=sim_time)
    agent.memory.add("Second observation.", importance=3.0, memory_type="observation", sim_time=sim_time)
    agent.memory.add("Third observation.", importance=4.0, memory_type="observation", sim_time=sim_time)

    store.sync_memories(agent_id, agent.memory)

    loaded = store.load_agent(agent_id)
    assert len(loaded.memory.entries) == 3


def test_sync_memories_persists_accumulator(store, agent):
    """The importance_accumulator is persisted and restored correctly."""
    agent_id = store.create_agent(agent, seed=42)

    sim_time = datetime(2024, 7, 15, 14, 0)
    # importance is clamped to [0, 10] by MemoryStream.add(), so use values ≤ 10.
    # 10 + 10 + 10 + 7 + 5 = 42
    agent.memory.add("Entry A.", importance=10.0, memory_type="observation", sim_time=sim_time)
    agent.memory.add("Entry B.", importance=10.0, memory_type="observation", sim_time=sim_time)
    agent.memory.add("Entry C.", importance=10.0, memory_type="observation", sim_time=sim_time)
    agent.memory.add("Entry D.", importance=7.0, memory_type="observation", sim_time=sim_time)
    agent.memory.add("Entry E.", importance=5.0, memory_type="observation", sim_time=sim_time)

    assert agent.memory.importance_accumulator == pytest.approx(42.0)

    store.sync_memories(agent_id, agent.memory)

    loaded = store.load_agent(agent_id)
    assert loaded.memory.importance_accumulator == pytest.approx(42.0)


# ── list_agents ───────────────────────────────────────────────────────────────

def test_list_agents_empty(store):
    """A fresh store returns an empty list."""
    result = store.list_agents()
    assert result == []


def test_list_agents_after_create(store):
    """After creating 2 agents, list_agents returns 2 entries with correct agent_ids."""
    agent_a = OccupantAgent.from_stratum("O1", seed=1)
    agent_b = OccupantAgent.from_stratum("O2", seed=2)

    id_a = store.create_agent(agent_a, seed=1)
    id_b = store.create_agent(agent_b, seed=2)

    result = store.list_agents()
    assert len(result) == 2

    returned_ids = {r["agent_id"] for r in result}
    assert id_a in returned_ids
    assert id_b in returned_ids


# ── delete_agent ──────────────────────────────────────────────────────────────

def test_delete_agent(store, agent):
    """After deletion, list_agents is empty and load_agent raises AgentNotFoundError."""
    agent_id = store.create_agent(agent, seed=42)

    deleted = store.delete_agent(agent_id)
    assert deleted is True

    assert store.list_agents() == []

    with pytest.raises(AgentNotFoundError):
        store.load_agent(agent_id)


# ── save_step ─────────────────────────────────────────────────────────────────

def test_save_step_persists(store, agent):
    """After save_step(), action_count in list_agents reflects the saved step."""
    agent_id = store.create_agent(agent, seed=42)

    env = _make_env()
    action = AgentAction(action_type="do_nothing")
    store.save_step(agent_id, action, env)

    result = store.list_agents()
    assert len(result) == 1
    assert result[0]["action_count"] == 1

    # A second save_step increments to 2
    store.save_step(agent_id, action, env)
    result2 = store.list_agents()
    assert result2[0]["action_count"] == 2
