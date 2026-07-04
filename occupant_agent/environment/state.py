"""
Shared environment state schema — v1.0.

This is the frozen contract between external platforms and the agent.
Both the REST API (Layer 2) and the MCP server (Layer 3) use this schema.
Do not change field names or types after Phase 2 begins (breaks reproducibility).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class DeviceState(BaseModel):
    device_id: str
    state: bool | float = Field(
        description="bool for ON/OFF devices; float for thermostat setpoint (°C)"
    )
    power_w: float = Field(ge=0, description="Rated power draw in watts when ON")


class RoomState(BaseModel):
    room_id: str
    occupied: bool


class EnvironmentState(BaseModel):
    """
    State snapshot sent to the agent at each 15-minute timestep.
    Produced by the external platform (EnergyPlus, Home Assistant, etc.)
    and consumed by OccupantAgent.step().
    """

    schema_version: str = Field(default=SCHEMA_VERSION, frozen=True)
    timestep: datetime = Field(description="ISO 8601 datetime of this timestep")
    zone_temp_c: float = Field(description="Current indoor zone temperature (°C)")
    outdoor_temp_c: float = Field(description="Current outdoor dry-bulb temperature (°C)")
    tou_rate: float = Field(
        ge=0, description="Current time-of-use electricity rate ($/kWh)"
    )
    thermostat_setpoint_c: float | None = Field(
        default=None,
        description="Current thermostat setpoint (°C); None if unknown or not applicable",
    )
    devices: list[DeviceState]
    rooms: list[RoomState]
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Platform-specific extension data (Home Assistant entity states, "
            "EnergyPlus zone variables, custom sensors, etc.). "
            "The core agent ignores this field; platform integrations may "
            "populate it to pass data through without modifying the base schema."
        ),
    )


class AgentAction(BaseModel):
    """Action returned by OccupantAgent.step()."""

    action_type: str = Field(
        description="One of: move_room, toggle_device, adjust_thermostat, do_nothing"
    )
    target_id: str | None = Field(
        default=None,
        description="device_id or room_id the action applies to; None for do_nothing",
    )
    value: bool | float | None = Field(
        default=None,
        description="New state for toggle (bool) or thermostat setpoint (float °C)",
    )
    reasoning: str | None = Field(
        default=None, description="Agent's natural-language rationale (for logging)"
    )


class SignalResponse(BaseModel):
    """Response returned by OccupantAgent.receive_signal()."""

    response: str = Field(description="One of: accepted, rejected, deferred")
    reasoning: str | None = None
