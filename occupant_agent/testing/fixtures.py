"""
Fixture factories for OccupantAgent tests.

These remove boilerplate from test files so that each test can focus on
the behavior under test rather than object construction.

Usage
─────
    from occupant_agent.testing import make_env, make_persona, make_memory_stream

    def test_my_persona():
        p = make_persona(stratum="P5", comfort_band_c=2.0)
        assert p.comfort_band_c == 2.0

    def test_my_scheduler(my_scheduler):
        env = make_env(zone_temp_c=26.0, tou_rate=0.22)
        code = my_scheduler.sample(env.timestep)
        assert len(code) == 6
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from occupant_agent.agent.memory import MemoryStream
from occupant_agent.agent.persona import Persona, create_persona
from occupant_agent.environment.state import DeviceState, EnvironmentState, RoomState

# ── EnvironmentState factory ──────────────────────────────────────────────────

def make_env(
    *,
    timestep: datetime | None = None,
    zone_temp_c: float = 22.0,
    outdoor_temp_c: float = 20.0,
    tou_rate: float = 0.09,
    thermostat_setpoint_c: float | None = 22.0,
    devices: list[DeviceState] | None = None,
    rooms: list[RoomState] | None = None,
    extensions: dict[str, Any] | None = None,
) -> EnvironmentState:
    """
    Build a minimal valid EnvironmentState for testing.

    All arguments are keyword-only. Unspecified arguments get sensible
    defaults (summer weekday evening, HVAC on, living room occupied).

    Args:
        timestep:           Defaults to 2024-07-15 18:00 (peak-rate evening).
        zone_temp_c:        Indoor temperature (°C).
        outdoor_temp_c:     Outdoor temperature (°C).
        tou_rate:           TOU electricity rate ($/kWh).
        thermostat_setpoint_c: Current thermostat setpoint (°C); None = unknown.
        devices:            Device list; defaults to [hvac ON 3500W].
        rooms:              Room list; defaults to [living_room occupied].
        extensions:         Platform-specific extensions dict.
    """
    return EnvironmentState(
        timestep=timestep or datetime(2024, 7, 15, 18, 0),
        zone_temp_c=zone_temp_c,
        outdoor_temp_c=outdoor_temp_c,
        tou_rate=tou_rate,
        thermostat_setpoint_c=thermostat_setpoint_c,
        devices=devices or [DeviceState(device_id="hvac", state=True, power_w=3500)],
        rooms=rooms or [RoomState(room_id="living_room", occupied=True)],
        extensions=extensions or {},
    )


def make_peak_env(**kwargs) -> EnvironmentState:
    """make_env preset: high TOU rate, hot outdoor temp — typical DR scenario."""
    return make_env(tou_rate=0.22, outdoor_temp_c=34.0, zone_temp_c=25.0, **kwargs)


def make_offpeak_env(**kwargs) -> EnvironmentState:
    """make_env preset: low TOU rate, mild outdoor temp — baseline conditions."""
    return make_env(tou_rate=0.06, outdoor_temp_c=17.0, zone_temp_c=21.0, **kwargs)


# ── Persona factory ───────────────────────────────────────────────────────────

def make_persona(
    stratum: str = "O1",
    seed: int = 0,
    **field_overrides,
) -> Persona:
    """
    Build a Persona for testing, optionally overriding specific fields.

    Uses create_persona() for a fully grounded baseline, then applies
    any field_overrides via dataclasses.replace().

    Args:
        stratum:         "O1" | "O2" | "O3" | "O4" (default "O1").
        seed:            RNG seed (default 0 for reproducibility).
        **field_overrides: Any Persona field to override, e.g.
                           comfort_band_c=2.0, work_from_home=True.

    Example:
        p = make_persona("O2", comfort_band_c=1.5, home_gym=True)
        assert p.comfort_band_c == 1.5
    """
    persona = create_persona(stratum, seed=seed)
    if field_overrides:
        persona = replace(persona, **field_overrides)
    return persona


# ── MemoryStream factory ──────────────────────────────────────────────────────

def make_memory_stream(
    n_observations: int = 3,
    n_reflections: int = 1,
    base_time: datetime | None = None,
    importance: float = 5.0,
) -> MemoryStream:
    """
    Build a MemoryStream pre-populated with synthetic entries.

    Useful for testing retrieve(), should_reflect(), and reflection
    without running a full simulation loop.

    Args:
        n_observations: Number of observation entries to add.
        n_reflections:  Number of reflection entries to add.
        base_time:      Timestamp for the first entry; subsequent entries
                        are spaced 15 minutes apart. Defaults to the
                        make_env() default timestep minus the total entries.
        importance:     Importance score for all observation entries.
    """
    from datetime import timedelta

    stream = MemoryStream()
    t = base_time or datetime(2024, 7, 15, 16, 0)

    for i in range(n_observations):
        stream.add(
            content=f"Observation {i + 1}: noticed the temperature was comfortable.",
            importance=importance,
            memory_type="observation",
            sim_time=t + timedelta(minutes=15 * i),
        )

    for j in range(n_reflections):
        stream.add(
            content=f"Insight {j + 1}: I tend to use more energy during peak hours.",
            importance=9.0,
            memory_type="reflection",
            sim_time=t + timedelta(minutes=15 * (n_observations + j)),
        )

    return stream


# ── Device / Room helpers ─────────────────────────────────────────────────────

def make_device(
    device_id: str = "hvac",
    state: bool | float = True,
    power_w: float = 3500.0,
) -> DeviceState:
    """Shorthand for DeviceState(...)."""
    return DeviceState(device_id=device_id, state=state, power_w=power_w)


def make_room(room_id: str = "living_room", occupied: bool = True) -> RoomState:
    """Shorthand for RoomState(...)."""
    return RoomState(room_id=room_id, occupied=occupied)
