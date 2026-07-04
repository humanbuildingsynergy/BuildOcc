"""
Stateful simulation environment: applies agent actions between timesteps.

Manages device states, room occupancy, and thermostat setpoint across steps.
Zone temperature is an external input — it comes from the building energy
platform (EnergyPlus, Home Assistant, ecobee API, etc.) or from a test fixture.
This agent does not model thermal physics or accumulate energy; those
responsibilities belong to the platform that integrates the agent.

Usage:
    sim = SimulationEnvironment(
        initial_devices=[...],
        initial_rooms=[...],
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )

    for timestep in timesteps:
        zone_temp_c = platform.get_zone_temp()     # from EnergyPlus / HA / etc.
        env = sim.observe(timestep, zone_temp_c)
        action = agent.step(env, atus_code=scheduler.sample(timestep))
        sim.apply(action, timestep)
"""

from __future__ import annotations

import csv as _csv_module
import math
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from occupant_agent.environment.state import (
    AgentAction,
    DeviceState,
    EnvironmentState,
    RoomState,
)

# ── Zone-temperature helpers (for users without a building energy model) ────────

def zone_temp_from_csv(
    csv_path: str | Path,
    datetime_col: str = "datetime",
    temp_col: str = "zone_temp_c",
) -> Callable[[datetime], float]:
    """
    Load a pre-computed zone temperature schedule from a CSV file.

    The CSV must have at minimum two columns: a datetime column (ISO 8601 strings
    or any format pandas can parse) and a numeric zone temperature column (°C).
    Missing timestamps are filled by nearest-neighbor lookup.

    Args:
        csv_path:     Path to the CSV file.
        datetime_col: Name of the datetime column (default ``"datetime"``).
        temp_col:     Name of the temperature column in °C (default ``"zone_temp_c"``).

    Returns:
        A callable ``(timestep: datetime) -> float`` that can be passed to
        ``SimulationEnvironment`` or called directly inside a simulation loop.

    Example CSV (zone_temps.csv)::

        datetime,zone_temp_c
        2024-07-15 00:00,22.0
        2024-07-15 00:15,21.7
        2024-07-15 00:30,21.5
        ...

    Example usage::

        from occupant_agent.environment.simulation import (
            SimulationEnvironment, zone_temp_from_csv, peak_tou_rate
        )

        zone_temp = zone_temp_from_csv("zone_temps.csv")
        sim = SimulationEnvironment(
            initial_devices=[...],
            initial_rooms=[...],
            thermostat_setpoint=22.0,
            outdoor_temp_fn=lambda ts: 29.0,   # fixed outdoor temp
            tou_rate_fn=peak_tou_rate,
        )

        for timestep in timesteps:
            env = sim.observe(timestep, zone_temp(timestep))
            action = agent.step(env)
            sim.apply(action, timestep)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for zone_temp_from_csv()") from e

    df = pd.read_csv(csv_path, parse_dates=[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    series: pd.Series = df[temp_col].astype(float)
    if series.empty:
        raise ValueError(
            f"CSV at {csv_path!r} contains no data rows; "
            "cannot build a zone temperature function."
        )

    def _lookup(timestep: datetime) -> float:
        ts = pd.Timestamp(timestep)
        # Strip timezone when the CSV index is naive (common with EnergyPlus exports).
        if ts.tzinfo is not None and series.index.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        if ts in series.index:
            return float(series[ts])
        # Nearest-neighbor fallback
        idx = series.index.get_indexer([ts], method="nearest")[0]
        return float(series.iloc[idx])

    return _lookup


def outdoor_temp_from_csv(
    csv_path: str | Path,
    datetime_col: str = "datetime",
    temp_col: str = "outdoor_temp_c",
) -> Callable[[datetime], float]:
    """
    Load outdoor dry-bulb temperatures from a CSV file.

    Identical in interface to :func:`zone_temp_from_csv` but reads the
    ``outdoor_temp_c`` column by default.  Use this when you have outdoor
    temperatures from a weather station, ecobee export, or pre-processed
    EnergyPlus output rather than a full EPW file.

    Args:
        csv_path:     Path to the CSV file.
        datetime_col: Name of the datetime column (default ``"datetime"``).
        temp_col:     Name of the outdoor temperature column in °C
                      (default ``"outdoor_temp_c"``).

    Returns:
        Callable ``(timestep: datetime) -> float``.

    Example CSV (outdoor_temps.csv)::

        datetime,outdoor_temp_c
        2025-08-10 00:00,26.3
        2025-08-10 00:15,26.0
        2025-08-10 00:30,25.7
        ...

    Example usage::

        outdoor_temp = outdoor_temp_from_csv("outdoor_temps.csv")
        sim = SimulationEnvironment(
            ...,
            outdoor_temp_fn=outdoor_temp,
            tou_rate_fn=peak_tou_rate,
        )
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for outdoor_temp_from_csv()") from e

    df = pd.read_csv(csv_path, parse_dates=[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    series = df[temp_col].astype(float)

    def _lookup(timestep: datetime) -> float:
        ts = pd.Timestamp(timestep)
        if ts.tzinfo is not None and series.index.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        if ts in series.index:
            return float(series[ts])
        idx = series.index.get_indexer([ts], method="nearest")[0]
        return float(series.iloc[idx])

    return _lookup


def tou_rate_from_csv(
    csv_path: str | Path,
    hour_start_col: str = "hour_start",
    hour_end_col: str = "hour_end",
    rate_col: str = "rate_kwh",
) -> Callable[[datetime], float]:
    """
    Load a time-of-use electricity rate schedule from a CSV file.

    The CSV defines contiguous hourly windows covering a full day (0–24).
    The returned callable is stateless and applies the same schedule every day
    regardless of weekday/weekend distinction.  For season- or day-type-aware
    tariffs, call this function once per tariff period and select the
    appropriate callable in your simulation loop.

    Args:
        csv_path:       Path to the CSV file.
        hour_start_col: Column name for window start hour, inclusive (default ``"hour_start"``).
        hour_end_col:   Column name for window end hour, exclusive (default ``"hour_end"``).
        rate_col:       Column name for the rate in $/kWh (default ``"rate_kwh"``).

    Returns:
        Callable ``(timestep: datetime) -> float`` returning the $/kWh rate.

    Example CSV (tou_rate.csv)::

        hour_start,hour_end,rate_kwh,label
        0,16,0.08,off-peak
        16,21,0.22,peak
        21,24,0.08,off-peak

    Example usage::

        tou_rate = tou_rate_from_csv("tou_rate.csv")
        sim = SimulationEnvironment(
            ...,
            outdoor_temp_fn=summer_day_temp,
            tou_rate_fn=tou_rate,
        )
    """
    windows: list[tuple[int, int, float]] = []
    with open(csv_path, newline="") as fh:
        reader = _csv_module.DictReader(fh)
        for row in reader:
            windows.append((
                int(row[hour_start_col]),
                int(row[hour_end_col]),
                float(row[rate_col]),
            ))

    if not windows:
        raise ValueError(f"No rate windows found in {csv_path}")

    def _rate(timestep: datetime) -> float:
        h = timestep.hour
        for start, end, rate in windows:
            if start <= h < end:
                return rate
        # Fallback to last window rate (handles hour 24 edge)
        return windows[-1][2]

    return _rate


def outdoor_temp_from_epw(
    epw_path: str | Path,
    year: int = 2025,
) -> Callable[[datetime], float]:
    """
    Load outdoor dry-bulb temperatures from an EnergyPlus Weather (EPW) file.

    Parses the 8760 centre-of-hour dry-bulb values, linearly interpolates to
    15-minute resolution, and returns a callable suitable for
    ``outdoor_temp_fn`` in :class:`SimulationEnvironment`.

    Args:
        epw_path: Path to a ``.epw`` file (any TMY3/TMY/CWEC source).
        year:     Calendar year to assign timestamps to (EPW data is
                  year-agnostic; default 2025 matches the repository sample).

    Returns:
        Callable ``(timestep: datetime) -> float`` returning outdoor temp in °C.

    Example::

        from occupant_agent.environment.simulation import (
            SimulationEnvironment, outdoor_temp_from_epw, zone_temp_from_csv,
            peak_tou_rate,
        )

        outdoor_temp = outdoor_temp_from_epw("Tucson_TMY3.epw")
        sim = SimulationEnvironment(
            initial_devices=[...],
            initial_rooms=[...],
            thermostat_setpoint=22.0,
            outdoor_temp_fn=outdoor_temp,
            tou_rate_fn=peak_tou_rate,
        )
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for outdoor_temp_from_epw()") from e

    path = Path(epw_path)
    hourly_c: list[float] = []
    with open(path, newline="") as fh:
        for i, row in enumerate(_csv_module.reader(fh)):
            if i < 8:   # EPW has 8 header rows
                continue
            hourly_c.append(float(row[6]))  # EPW column 7 is dry-bulb temp in °C

    if len(hourly_c) != 8760:
        raise ValueError(
            f"Expected 8760 hourly data rows in EPW file, got {len(hourly_c)}. "
            "Ensure the file is a standard annual EPW."
        )

    # EPW hour labels are 1-indexed and represent the centre of each clock hour
    # (hour 1 = 00:30, hour 24 = 23:30). We interpolate to 15-minute values
    # whose timestamps represent the start of each 15-minute slot (00:00, 00:15, …).
    base = datetime(year, 1, 1, 0, 0)
    timestamps: list[datetime] = []
    temps: list[float] = []
    for step in range(365 * 24 * 4):   # 35 040 quarter-hour steps
        t_h = step * 0.25              # hours since Jan 1 00:00
        idx_f = t_h - 0.5             # shifted to centre-of-hour grid
        i0 = int(math.floor(idx_f)) % 8760
        i1 = (i0 + 1) % 8760
        frac = idx_f - math.floor(idx_f)
        temps.append(hourly_c[i0] * (1.0 - frac) + hourly_c[i1] * frac)
        timestamps.append(base + timedelta(minutes=15 * step))

    series = pd.Series(temps, index=pd.DatetimeIndex(timestamps))

    def _lookup(timestep: datetime) -> float:
        ts = pd.Timestamp(timestep)
        if ts in series.index:
            return float(series[ts])
        loc = series.index.get_indexer([ts], method="nearest")[0]
        return float(series.iloc[loc])

    return _lookup


def constant_zone_temp(temp_c: float) -> Callable[[datetime], float]:
    """
    Return a zone-temperature callable that always returns a fixed value.

    Useful for simple experiments where thermal dynamics are not being studied.

    Args:
        temp_c: Fixed indoor zone temperature in °C.

    Returns:
        A callable ``(timestep: datetime) -> float``.

    Example::

        from occupant_agent.environment.simulation import (
            SimulationEnvironment, constant_zone_temp, peak_tou_rate
        )

        sim = SimulationEnvironment(
            ...,
            outdoor_temp_fn=lambda ts: 29.0,
            tou_rate_fn=peak_tou_rate,
        )
        zone_temp = constant_zone_temp(23.0)

        for timestep in timesteps:
            env = sim.observe(timestep, zone_temp(timestep))
    """
    return lambda _ts: float(temp_c)


# ── Helper schedule functions (test fixtures / simple integrations) ────────────

def summer_day_temp(timestep: datetime, peak_c: float = 35.0, min_c: float = 22.0) -> float:
    """
    Sinusoidal outdoor temperature for a summer day (°C).
    Peak at 6pm (hour 18), minimum at 6am (hour 6).
    Replace with NOAA hourly data for Phase 2 validation.
    """
    hour = timestep.hour + timestep.minute / 60.0
    angle = (hour - 6.0) / 24.0 * 2.0 * math.pi
    amplitude = (peak_c - min_c) / 2.0
    midpoint = min_c + amplitude
    return midpoint + amplitude * math.sin(angle - math.pi / 2.0)


def peak_tou_rate(
    timestep: datetime,
    peak_rate: float = 0.22,
    offpeak_rate: float = 0.08,
    peak_start: int = 16,
    peak_end: int = 21,
) -> float:
    """
    Simple peak / off-peak TOU tariff.
    Default: peak 4pm–9pm at $0.22/kWh, off-peak $0.08/kWh.
    Replace with utility tariff schedule for Phase 2 validation.
    """
    return peak_rate if peak_start <= timestep.hour < peak_end else offpeak_rate


# ── Preset household configurations ──────────────────────────────────────────

# Canonical power ratings (W) for LLM-controllable devices.
# HVAC always starts on; all others start off.
# Strata vary in which devices they own — see persona_devices().
DEVICE_POWER_W: dict[str, int] = {
    "hvac":             3500,
    "lighting_living":    60,
    "lighting_bedroom":   40,
    "lighting_kitchen":   50,
    "tv":                120,
    "washer":            500,
    "dishwasher":       1200,
}


def persona_devices(persona) -> list[DeviceState]:
    """
    Build the controllable device list for a specific persona.

    HVAC is always included and starts on. Lighting and appliances are included
    only when present in persona.appliances, so the device set adapts to each
    stratum's ownership profile (e.g., O2 has no dishwasher).

    Args:
        persona: Any object with an ``appliances: set[str]`` attribute
                 (Persona, BasePersona, or a third-party persona extension).

    Returns:
        list[DeviceState] ordered by DEVICE_POWER_W.
    """
    return [
        DeviceState(
            device_id=dev_id,
            state=(dev_id == "hvac"),
            power_w=pw,
        )
        for dev_id, pw in DEVICE_POWER_W.items()
        if dev_id == "hvac" or dev_id in persona.appliances
    ]


def typical_household_devices() -> list[DeviceState]:
    """
    A representative US household device inventory for rapid prototyping.
    Replace with persona_devices(persona) when a specific stratum is available.
    """
    return [
        DeviceState(device_id="hvac",             state=True,  power_w=3500),
        DeviceState(device_id="lighting_living",  state=False, power_w=60),
        DeviceState(device_id="lighting_bedroom", state=False, power_w=40),
        DeviceState(device_id="lighting_kitchen", state=False, power_w=50),
        DeviceState(device_id="tv",               state=False, power_w=120),
        DeviceState(device_id="washer",           state=False, power_w=500),
        DeviceState(device_id="dishwasher",       state=False, power_w=1200),
    ]


def typical_household_rooms() -> list[RoomState]:
    """
    A representative room layout for rapid prototyping.
    Replace with persona.room_ids when a specific stratum is available.
    """
    return [
        RoomState(room_id="living_room",  occupied=True),
        RoomState(room_id="kitchen",      occupied=False),
        RoomState(room_id="bedroom",      occupied=False),
        RoomState(room_id="laundry_room", occupied=False),
    ]


# ── SimulationEnvironment ─────────────────────────────────────────────────────

class SimulationEnvironment:
    """
    Stateful environment that applies agent actions between timesteps.

    Tracks device states, room occupancy, and thermostat setpoint. Zone
    temperature is provided externally — this class does not model heat
    transfer or energy consumption.

    Design:
        env = sim.observe(timestep, zone_temp_c)    # agent reads this
        action = agent.step(env)
        sim.apply(action, timestep)                  # state updates
    """

    def __init__(
        self,
        initial_devices: list[DeviceState],
        initial_rooms: list[RoomState],
        thermostat_setpoint: float,
        outdoor_temp_fn: Callable[[datetime], float],
        tou_rate_fn: Callable[[datetime], float],
    ) -> None:
        self._devices: dict[str, DeviceState] = {d.device_id: d for d in initial_devices}
        self._rooms: dict[str, RoomState] = {r.room_id: r for r in initial_rooms}
        self._setpoint: float = float(thermostat_setpoint)
        self._outdoor_temp_fn = outdoor_temp_fn
        self._tou_rate_fn = tou_rate_fn

    # ── Core loop interface ───────────────────────────────────────────────────

    def observe(self, timestep: datetime, zone_temp_c: float) -> EnvironmentState:
        """
        Produce an EnvironmentState snapshot for the current timestep.

        Args:
            timestep:    Current simulation time.
            zone_temp_c: Current indoor zone temperature (°C) from the building
                         energy platform or test fixture.
        """
        return EnvironmentState(
            timestep=timestep,
            zone_temp_c=round(float(zone_temp_c), 1),
            outdoor_temp_c=round(self._outdoor_temp_fn(timestep), 1),
            tou_rate=self._tou_rate_fn(timestep),
            thermostat_setpoint_c=round(self._setpoint, 1),
            devices=list(self._devices.values()),
            rooms=list(self._rooms.values()),
        )

    def apply(self, action: AgentAction, timestep: datetime) -> None:
        """
        Apply an agent action, updating device, room, and setpoint state.

        Zone temperature and energy accounting are the responsibility of the
        calling platform (EnergyPlus, Home Assistant, etc.).
        """
        atype = action.action_type
        tid = action.target_id
        val = action.value

        if atype == "toggle_device" and tid and tid in self._devices:
            dev = self._devices[tid]
            new_state: bool | float
            if val is not None:
                new_state = val
            else:
                new_state = not bool(dev.state)
            self._devices[tid] = DeviceState(
                device_id=dev.device_id,
                state=new_state,
                power_w=dev.power_w,
            )

        elif atype == "adjust_thermostat" and val is not None:
            new_sp = max(15.0, min(31.0, float(val)))
            # Per-step delta guard: no single action can shift setpoint by more than 3°C.
            new_sp = max(self._setpoint - 3.0, min(self._setpoint + 3.0, new_sp))
            self._setpoint = new_sp
            for dev_id, dev in self._devices.items():
                if dev_id in ("thermostat", "smart_thermostat"):
                    self._devices[dev_id] = DeviceState(
                        device_id=dev.device_id,
                        state=new_sp,
                        power_w=dev.power_w,
                    )

        elif atype == "move_room" and tid and tid in self._rooms:
            for room_id in self._rooms:
                self._rooms[room_id] = RoomState(
                    room_id=room_id,
                    occupied=(room_id == tid),
                )

        # do_nothing: no state change

    # ── State accessors ───────────────────────────────────────────────────────

    @property
    def thermostat_setpoint(self) -> float:
        return self._setpoint

    @property
    def device_states(self) -> list[DeviceState]:
        return list(self._devices.values())

    @property
    def room_states(self) -> list[RoomState]:
        return list(self._rooms.values())

    # ── Snapshot / restore ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a JSON-serialisable snapshot of the current environment state.

        Captures device on/off states, room occupancy, and thermostat setpoint.
        Does not capture ``outdoor_temp_fn`` or ``tou_rate_fn`` — those must be
        re-supplied to :meth:`from_snapshot`.

        Use this to pause and resume a multi-day simulation, export current
        device states to a building management system, or feed state into
        EnergyPlus co-simulation callbacks.

        Example::

            import json

            snapshot = sim.snapshot()
            json.dump(snapshot, open("sim_state.json", "w"))

            # Later / in another process:
            sim2 = SimulationEnvironment.from_snapshot(
                json.load(open("sim_state.json")),
                outdoor_temp_fn=summer_day_temp,
                tou_rate_fn=peak_tou_rate,
            )
        """
        return {
            "thermostat_setpoint_c": self._setpoint,
            "devices": [
                {"device_id": d.device_id, "state": d.state, "power_w": d.power_w}
                for d in self._devices.values()
            ],
            "rooms": [
                {"room_id": r.room_id, "occupied": r.occupied}
                for r in self._rooms.values()
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
        outdoor_temp_fn: Callable[[datetime], float],
        tou_rate_fn: Callable[[datetime], float],
    ) -> SimulationEnvironment:
        """
        Restore a :class:`SimulationEnvironment` from a :meth:`snapshot` dict.

        Args:
            snapshot:        Dict produced by :meth:`snapshot`.
            outdoor_temp_fn: Outdoor temperature callable — must be re-supplied
                             since callables are not serialisable.
            tou_rate_fn:     TOU rate callable — must be re-supplied.

        Returns:
            A new :class:`SimulationEnvironment` with the restored state.
        """
        devices = [
            DeviceState(
                device_id=d["device_id"],
                state=d["state"],
                power_w=d["power_w"],
            )
            for d in snapshot["devices"]
        ]
        rooms = [
            RoomState(room_id=r["room_id"], occupied=r["occupied"])
            for r in snapshot["rooms"]
        ]
        return cls(
            initial_devices=devices,
            initial_rooms=rooms,
            thermostat_setpoint=snapshot["thermostat_setpoint_c"],
            outdoor_temp_fn=outdoor_temp_fn,
            tou_rate_fn=tou_rate_fn,
        )
