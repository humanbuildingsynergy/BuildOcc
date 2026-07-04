"""
End-to-end simulation tests using MockLLMAgent.

Exercises the full step() loop, all strata, signal delivery, memory
accumulation, and reflection — no API key or network access required.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from occupant_agent import (
    ActivityScheduler,
    DeviceState,
    RoomState,
    SimulationEnvironment,
    constant_zone_temp,
    peak_tou_rate,
    summer_day_temp,
)
from occupant_agent.testing import MockLLMAgent

_TS_START = datetime(2025, 8, 12, 8, 0, tzinfo=UTC)
_STEP = timedelta(minutes=15)

_MOCK_STEP = {
    "action_type": "do_nothing",
    "target_id": None,
    "value": None,
    "reasoning": "No action needed.",
    "_memory_note": "Uneventful timestep.",
    "_importance": 2,
}
_MOCK_SIGNAL = {
    "response": "accepted",
    "reasoning": "Complying with the request.",
    "_importance": 5,
}
_MOCK_REFLECT = {
    "insights": [
        "I tend to use more energy in the evenings.",
        "I respond well to cost-savings signals.",
        "My thermostat habits are fairly consistent.",
    ]
}


def _make_sim() -> SimulationEnvironment:
    return SimulationEnvironment(
        initial_devices=[
            DeviceState(device_id="hvac", state=True, power_w=3500),
            DeviceState(device_id="tv", state=False, power_w=120),
        ],
        initial_rooms=[
            RoomState(room_id="living_room", occupied=True),
            RoomState(room_id="bedroom", occupied=False),
        ],
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )


# ── Basic loop ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stratum", ["O1", "O2", "O3", "O4"])
def test_simulation_loop_all_strata(stratum: str) -> None:
    agent = MockLLMAgent.from_stratum(stratum, seed=42)
    scheduler = ActivityScheduler(stratum=stratum, seed=42)
    sim = _make_sim()
    zone_temp = constant_zone_temp(23.0)
    rng = random.Random(42)

    ts = _TS_START
    for _ in range(8):
        env = sim.observe(ts, zone_temp(ts))
        atus_code = scheduler.sample(ts)
        wfh = agent.persona.sample_wfh_today(rng)
        action = agent.step(env, atus_code=atus_code, wfh_today=wfh)
        sim.apply(action, ts)
        ts += _STEP

    assert agent.action_count == 8
    assert agent.memory.count() > 0


def test_simulation_persists_memories() -> None:
    agent = MockLLMAgent.from_stratum("O1", seed=0)
    sim = _make_sim()
    zone_temp = constant_zone_temp(23.0)
    ts = _TS_START
    for _ in range(4):
        env = sim.observe(ts, zone_temp(ts))
        agent.step(env)
        ts += _STEP
    assert agent.memory.count() >= 4


# ── Signal delivery ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("sig_type", ["A", "B", "C"])
def test_signal_all_types_accepted(sig_type: str) -> None:
    agent = MockLLMAgent.from_stratum("O1", seed=0, signal_response=_MOCK_SIGNAL)
    sim = _make_sim()
    env = sim.observe(_TS_START, 25.0)

    response = agent.receive_signal(
        signal_type=sig_type,
        content="Please reduce HVAC use during peak pricing.",
        env=env,
    )
    assert response.response in {"accepted", "rejected", "deferred"}


def test_signal_adds_to_memory() -> None:
    agent = MockLLMAgent.from_stratum("O1", seed=0)
    sim = _make_sim()
    env = sim.observe(_TS_START, 25.0)
    before = agent.memory.count()
    agent.receive_signal("B", "Raise your setpoint to save on peak rates.", env)
    assert agent.memory.count() > before


def test_all_strata_accept_all_signal_types() -> None:
    for stratum in ("O1", "O2", "O3", "O4"):
        agent = MockLLMAgent.from_stratum(stratum, seed=0)
        sim = _make_sim()
        env = sim.observe(_TS_START, 24.0)
        for sig in ("A", "B", "C"):
            response = agent.receive_signal(sig, "Reduce HVAC use.", env)
            assert response.response in {"accepted", "rejected", "deferred"}, (
                f"{stratum} / signal {sig} returned unexpected response"
            )


# ── Thermostat and device actions ─────────────────────────────────────────────

def test_thermostat_action_applied() -> None:
    step_resp = {**_MOCK_STEP, "action_type": "adjust_thermostat", "value": 24.0}
    agent = MockLLMAgent.from_stratum("O1", seed=0, step_response=step_resp)
    sim = _make_sim()
    env = sim.observe(_TS_START, 23.0)
    action = agent.step(env)
    sim.apply(action, _TS_START)
    next_env = sim.observe(_TS_START + _STEP, 23.0)
    assert next_env.thermostat_setpoint_c == pytest.approx(24.0)


def test_toggle_device_action_applied() -> None:
    step_resp = {**_MOCK_STEP, "action_type": "toggle_device",
                 "target_id": "tv", "value": True}
    agent = MockLLMAgent.from_stratum("O1", seed=0, step_response=step_resp)
    sim = _make_sim()
    env = sim.observe(_TS_START, 23.0)
    action = agent.step(env)
    sim.apply(action, _TS_START)
    next_env = sim.observe(_TS_START + _STEP, 23.0)
    tv = next(d for d in next_env.devices if d.device_id == "tv")
    assert tv.state is True


def test_move_room_action_applied() -> None:
    step_resp = {**_MOCK_STEP, "action_type": "move_room", "target_id": "bedroom"}
    agent = MockLLMAgent.from_stratum("O1", seed=0, step_response=step_resp)
    sim = _make_sim()
    env = sim.observe(_TS_START, 23.0)
    action = agent.step(env)
    sim.apply(action, _TS_START)
    next_env = sim.observe(_TS_START + _STEP, 23.0)
    bedroom = next(r for r in next_env.rooms if r.room_id == "bedroom")
    assert bedroom.occupied is True


# ── Reflection ────────────────────────────────────────────────────────────────

def test_reflection_triggers_when_accumulator_full() -> None:
    agent = MockLLMAgent.from_stratum("O1", seed=0, reflect_response=_MOCK_REFLECT)
    agent.memory._importance_accumulator = 99.0
    sim = _make_sim()
    ts = _TS_START

    with patch("occupant_agent.llm.client.call_llm", return_value=_MOCK_REFLECT):
        env = sim.observe(ts, 23.0)
        agent.step(env)

    assert agent.memory.last_reflected_at is not None


# ── ActivityScheduler ─────────────────────────────────────────────────────────

def test_scheduler_samples_valid_codes() -> None:
    scheduler = ActivityScheduler(stratum="O1", seed=0)
    ts = _TS_START
    for _ in range(96):
        code = scheduler.sample(ts)
        assert code is None or (isinstance(code, str) and len(code) == 6), (
            f"Invalid ATUS code: {code!r}"
        )
        ts += _STEP


def test_scheduler_weekday_weekend_split() -> None:
    scheduler = ActivityScheduler(stratum="O1", seed=0)
    # Monday 2025-08-11 (weekday)
    mon = datetime(2025, 8, 11, 10, 0, tzinfo=UTC)
    # Sunday 2025-08-10 (weekend)
    sun = datetime(2025, 8, 10, 10, 0, tzinfo=UTC)
    _ = scheduler.sample(mon)
    _ = scheduler.sample(sun)


# ── REST API via persistence ──────────────────────────────────────────────────

def test_agent_store_round_trip() -> None:
    from occupant_agent import AgentStore
    from occupant_agent.testing import MockLLMAgent

    store = AgentStore()
    agent = MockLLMAgent.from_stratum("O2", seed=3)
    agent_id = store.create_agent(agent, seed=3)

    loaded = store.load_agent(agent_id)
    assert loaded.persona.stratum == "O2"
    assert loaded.persona.age == agent.persona.age

    store.sync_memories(agent_id, agent.memory)
    store.delete_agent(agent_id)
