from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from occupant_agent.environment.state import (
    AgentAction,
    DeviceState,
    EnvironmentState,
    RoomState,
    SignalResponse,
)

# ── Shared test data ──────────────────────────────────────────────────────────

_TIMESTEP = datetime(2024, 7, 15, 14, 0)

_DEVICES = [
    DeviceState(device_id="hvac", state=True, power_w=2000.0),
    DeviceState(device_id="thermostat", state=22.5, power_w=0.0),
]
_ROOMS = [
    RoomState(room_id="living_room", occupied=True),
    RoomState(room_id="bedroom", occupied=False),
]


def _make_env(**overrides) -> EnvironmentState:
    """Return a valid EnvironmentState, optionally overriding fields."""
    defaults = dict(
        timestep=_TIMESTEP,
        zone_temp_c=22.0,
        outdoor_temp_c=35.0,
        tou_rate=0.12,
        thermostat_setpoint_c=22.0,
        devices=list(_DEVICES),
        rooms=list(_ROOMS),
    )
    defaults.update(overrides)
    return EnvironmentState(**defaults)


# ── EnvironmentState ──────────────────────────────────────────────────────────

def test_environment_state_valid():
    """Construct with all fields; verify schema_version is '1.0'."""
    env = _make_env()

    assert env.schema_version == "1.0"
    assert env.timestep == _TIMESTEP
    assert env.zone_temp_c == pytest.approx(22.0)
    assert env.outdoor_temp_c == pytest.approx(35.0)
    assert env.tou_rate == pytest.approx(0.12)
    assert env.thermostat_setpoint_c == pytest.approx(22.0)
    assert len(env.devices) == 2
    assert len(env.rooms) == 2


def test_environment_state_thermostat_setpoint_optional():
    """thermostat_setpoint_c defaults to None when not supplied."""
    env = _make_env(thermostat_setpoint_c=None)
    assert env.thermostat_setpoint_c is None

    # Also omitting the field entirely should default to None
    env2 = EnvironmentState(
        timestep=_TIMESTEP,
        zone_temp_c=22.0,
        outdoor_temp_c=35.0,
        tou_rate=0.12,
        devices=[],
        rooms=[],
    )
    assert env2.thermostat_setpoint_c is None


def test_environment_state_tou_rate_nonnegative():
    """tou_rate=-0.1 must raise a ValidationError (ge=0 constraint)."""
    with pytest.raises(ValidationError):
        _make_env(tou_rate=-0.1)


# ── DeviceState ───────────────────────────────────────────────────────────────

def test_device_state_bool():
    """DeviceState with state=True (ON/OFF device) is valid."""
    d = DeviceState(device_id="dishwasher", state=True, power_w=1200.0)
    assert d.device_id == "dishwasher"
    assert d.state is True
    assert d.power_w == pytest.approx(1200.0)


def test_device_state_float():
    """DeviceState with state=22.0 (thermostat setpoint °C) is valid."""
    d = DeviceState(device_id="thermostat", state=22.0, power_w=0.0)
    assert d.state == pytest.approx(22.0)


# ── AgentAction ───────────────────────────────────────────────────────────────

def test_agent_action_valid_types():
    """Each of the four action_type values is accepted by AgentAction."""
    valid_types = ["do_nothing", "toggle_device", "adjust_thermostat", "move_room"]
    for action_type in valid_types:
        action = AgentAction(action_type=action_type)
        assert action.action_type == action_type

    # Verify optional fields default to None
    noop = AgentAction(action_type="do_nothing")
    assert noop.target_id is None
    assert noop.value is None
    assert noop.reasoning is None

    # Verify value can be bool (toggle) or float (thermostat)
    toggle = AgentAction(action_type="toggle_device", target_id="dishwasher", value=False)
    assert toggle.value is False

    setpoint = AgentAction(action_type="adjust_thermostat", target_id="thermostat", value=20.0)
    assert setpoint.value == pytest.approx(20.0)


# ── SignalResponse ────────────────────────────────────────────────────────────

def test_signal_response_valid():
    """response must be one of accepted/rejected/deferred; reasoning is optional."""
    for resp in ("accepted", "rejected", "deferred"):
        sr = SignalResponse(response=resp)
        assert sr.response == resp
        assert sr.reasoning is None

    # reasoning field is accepted when provided
    sr_with_reason = SignalResponse(
        response="accepted",
        reasoning="The signal offers a clear cost saving.",
    )
    assert sr_with_reason.reasoning == "The signal offers a clear cost saving."


# ── Round-trip JSON serialization ─────────────────────────────────────────────

def test_environment_state_round_trip_json():
    """model_dump_json() then model_validate_json() preserves all fields."""
    env = _make_env()

    json_str = env.model_dump_json()
    restored = EnvironmentState.model_validate_json(json_str)

    assert restored.schema_version == env.schema_version
    assert restored.timestep == env.timestep
    assert restored.zone_temp_c == pytest.approx(env.zone_temp_c)
    assert restored.outdoor_temp_c == pytest.approx(env.outdoor_temp_c)
    assert restored.tou_rate == pytest.approx(env.tou_rate)
    assert restored.thermostat_setpoint_c == pytest.approx(env.thermostat_setpoint_c)

    assert len(restored.devices) == len(env.devices)
    for orig, back in zip(env.devices, restored.devices):
        assert back.device_id == orig.device_id
        assert back.power_w == pytest.approx(orig.power_w)

    assert len(restored.rooms) == len(env.rooms)
    for orig, back in zip(env.rooms, restored.rooms):
        assert back.room_id == orig.room_id
        assert back.occupied == orig.occupied
