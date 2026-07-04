"""
Unit tests for occupant_agent.environment.simulation.

Tests cover:
  - SimulationEnvironment.observe() / apply() stateful loop
  - summer_day_temp() sinusoidal temperature model
  - peak_tou_rate() peak/off-peak tariff

Zone temperature is passed explicitly to observe() — the simulation
environment does not model thermal physics (that belongs to the building
energy platform: EnergyPlus, Home Assistant, etc.).

Base timestep: datetime(2024, 7, 15, 18, 0) — peak hour, summer day.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from occupant_agent.environment.simulation import (
    SimulationEnvironment,
    peak_tou_rate,
    summer_day_temp,
)
from occupant_agent.environment.state import AgentAction, DeviceState, RoomState

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_TS = datetime(2024, 7, 15, 18, 0)  # peak hour, summer
ZONE_TEMP = 24.0                         # test fixture zone temperature (°C)


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def sim():
    return SimulationEnvironment(
        initial_devices=[
            DeviceState(device_id="hvac",   state=True,  power_w=3500),
            DeviceState(device_id="tv",     state=False, power_w=150),
            DeviceState(device_id="washer", state=False, power_w=500),
        ],
        initial_rooms=[
            RoomState(room_id="living_room", occupied=True),
            RoomState(room_id="bedroom",     occupied=False),
        ],
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )


# ── SimulationEnvironment.observe() ──────────────────────────────────────────

def test_observe_returns_environment_state(sim):
    """observe() returns an EnvironmentState with correct field values."""
    env = sim.observe(BASE_TS, ZONE_TEMP)

    assert env.zone_temp_c == pytest.approx(ZONE_TEMP)
    assert env.thermostat_setpoint_c == pytest.approx(22.0)
    assert len(env.devices) == 3
    assert len(env.rooms) == 2


def test_observe_includes_setpoint(sim):
    """thermostat_setpoint_c must not be None."""
    env = sim.observe(BASE_TS, ZONE_TEMP)

    assert env.thermostat_setpoint_c is not None


def test_observe_passes_through_zone_temp(sim):
    """zone_temp_c in EnvironmentState matches the value passed to observe()."""
    env = sim.observe(BASE_TS, 78.5)

    assert env.zone_temp_c == pytest.approx(78.5)


# ── SimulationEnvironment.apply() — device toggle ────────────────────────────

def test_apply_toggle_device_on(sim):
    """Toggling tv to True → tv.state is True in the next observe()."""
    action = AgentAction(action_type="toggle_device", target_id="tv", value=True)
    sim.apply(action, BASE_TS)

    env = sim.observe(BASE_TS, ZONE_TEMP)
    tv = next(d for d in env.devices if d.device_id == "tv")
    assert tv.state is True


def test_apply_toggle_device_off(sim):
    """Toggling hvac to False → hvac.state is False in the next observe()."""
    action = AgentAction(action_type="toggle_device", target_id="hvac", value=False)
    sim.apply(action, BASE_TS)

    env = sim.observe(BASE_TS, ZONE_TEMP)
    hvac = next(d for d in env.devices if d.device_id == "hvac")
    assert hvac.state is False


# ── SimulationEnvironment.apply() — thermostat ───────────────────────────────

def test_apply_thermostat_adjusts_setpoint(sim):
    """adjust_thermostat to 24.0°C (within ±3°C delta guard from 22.0) → setpoint == 24.0."""
    action = AgentAction(action_type="adjust_thermostat", value=24.0)
    sim.apply(action, BASE_TS)

    env = sim.observe(BASE_TS, ZONE_TEMP)
    assert env.thermostat_setpoint_c == pytest.approx(24.0)


def test_apply_thermostat_clamped_low(sim):
    """Request well below 15.0°C floor: delta guard (±3°C from 22.0) clamps to 19.0°C."""
    action = AgentAction(action_type="adjust_thermostat", value=5.0)
    sim.apply(action, BASE_TS)

    env = sim.observe(BASE_TS, ZONE_TEMP)
    assert env.thermostat_setpoint_c == pytest.approx(19.0)


def test_apply_thermostat_clamped_high(sim):
    """Request well above 31.0°C ceiling: delta guard (±3°C from 22.0) clamps to 25.0°C."""
    action = AgentAction(action_type="adjust_thermostat", value=100.0)
    sim.apply(action, BASE_TS)

    env = sim.observe(BASE_TS, ZONE_TEMP)
    assert env.thermostat_setpoint_c == pytest.approx(25.0)


# ── SimulationEnvironment.apply() — room occupancy ───────────────────────────

def test_apply_move_room(sim):
    """move_room to 'bedroom' → bedroom is occupied, living_room is not."""
    action = AgentAction(action_type="move_room", target_id="bedroom")
    sim.apply(action, BASE_TS)

    env = sim.observe(BASE_TS, ZONE_TEMP)
    bedroom = next(r for r in env.rooms if r.room_id == "bedroom")
    living_room = next(r for r in env.rooms if r.room_id == "living_room")

    assert bedroom.occupied is True
    assert living_room.occupied is False


def test_apply_move_room_repeated_cycles_occupancy(sim):
    """Consecutive move_room calls correctly transfer occupancy each time."""
    # The sim fixture has living_room and bedroom (from the sim fixture setup)
    for target in ("bedroom", "living_room", "bedroom"):
        sim.apply(AgentAction(action_type="move_room", target_id=target), BASE_TS)
        env = sim.observe(BASE_TS, ZONE_TEMP)
        occupied = [r.room_id for r in env.rooms if r.occupied]
        assert occupied == [target], f"Expected {target} occupied; got {occupied}"


def test_apply_move_room_invalid_target_is_noop(sim):
    """move_room to a non-existent room_id leaves occupancy unchanged (no-op).

    apply() guards on `tid in self._rooms` so an unknown target never clears
    all occupancy — the occupant stays where they were.
    """
    before = sim.observe(BASE_TS, ZONE_TEMP)
    before_occ = {r.room_id: r.occupied for r in before.rooms}

    sim.apply(AgentAction(action_type="move_room", target_id="nonexistent_room"), BASE_TS)

    after = sim.observe(BASE_TS, ZONE_TEMP)
    after_occ = {r.room_id: r.occupied for r in after.rooms}
    assert after_occ == before_occ


# ── SimulationEnvironment.apply() — do_nothing ───────────────────────────────

def test_apply_do_nothing(sim):
    """do_nothing leaves device states, room occupancy, and setpoint unchanged."""
    before = sim.observe(BASE_TS, ZONE_TEMP)
    before_states = {d.device_id: d.state for d in before.devices}
    before_rooms = {r.room_id: r.occupied for r in before.rooms}
    before_setpoint = before.thermostat_setpoint_c

    action = AgentAction(action_type="do_nothing")
    sim.apply(action, BASE_TS)

    after = sim.observe(BASE_TS, ZONE_TEMP)
    after_states = {d.device_id: d.state for d in after.devices}
    after_rooms = {r.room_id: r.occupied for r in after.rooms}

    assert after_states == before_states
    assert after_rooms == before_rooms
    assert after.thermostat_setpoint_c == pytest.approx(before_setpoint)


# ── summer_day_temp() ─────────────────────────────────────────────────────────

def test_summer_day_temp_peak_at_3pm():
    """
    summer_day_temp returns °C. Peak (~35°C) is near 15:00; minimum (22°C) is at 06:00.
    We assert temp > 30°C at 3pm and ≈22°C at 6am.
    """
    ts_3pm = datetime(2024, 7, 15, 15, 0)
    ts_6am = datetime(2024, 7, 15, 6, 0)

    assert summer_day_temp(ts_3pm) > 30.0
    assert summer_day_temp(ts_6am) == pytest.approx(22.0, abs=0.5)


# ── peak_tou_rate() ───────────────────────────────────────────────────────────

def test_peak_tou_rate_peak_hours():
    """Rate at 18:00 (peak window 16–21) is $0.22; at 10:00 off-peak is $0.08."""
    ts_peak = datetime(2024, 7, 15, 18, 0)
    ts_offpeak = datetime(2024, 7, 15, 10, 0)

    assert peak_tou_rate(ts_peak) == pytest.approx(0.22)
    assert peak_tou_rate(ts_offpeak) == pytest.approx(0.08)
