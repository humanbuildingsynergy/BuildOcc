"""
EnergyPlus integration via REST API (Layer 2).

Drop this into your EnergyPlus Python API workflow. At each timestep the
callback reads zone conditions, sends them to the OccupantAgent REST API,
and writes the returned action back to EnergyPlus.

Requires:
  - BuildOcc REST API running: buildocc-api
    (or: uvicorn occupant_agent.api.app:app --port 8000)
  - pyenergyplus installed alongside EnergyPlus
"""

from datetime import datetime, timedelta

import httpx
from pyenergyplus.api import EnergyPlusAPI  # type: ignore

API_BASE = "http://localhost:8000"
AGENT_ID: str | None = None

# Integer handles fetched once in initialize() and reused every callback call.
# get_variable_handle / get_actuator_handle return -1 if the variable/actuator
# is not found — check for -1 before calling get_variable_value / set_actuator_value.
_ZONE_TEMP_HANDLE: int = -1
_OUTDOOR_TEMP_HANDLE: int = -1
_THERMOSTAT_HANDLE: int = -1


def initialize(api: EnergyPlusAPI, state: object) -> None:
    """
    Called once at the start of each EnergyPlus run period.
    Fetches EnergyPlus variable/actuator handles and creates the BuildOcc agent.

    Register this as a callback:
        api.runtime.callback_begin_new_environment(state, initialize)
    """
    global AGENT_ID, _ZONE_TEMP_HANDLE, _OUTDOOR_TEMP_HANDLE, _THERMOSTAT_HANDLE

    # Fetch integer handles for EnergyPlus outputs and actuators.
    # Adjust zone/object names to match your IDD/IDF file.
    _ZONE_TEMP_HANDLE = api.exchange.get_variable_handle(
        state, "Zone Mean Air Temperature", "LIVING ZONE"
    )
    _OUTDOOR_TEMP_HANDLE = api.exchange.get_variable_handle(
        state, "Site Outdoor Air Drybulb Temperature", "Environment"
    )
    _THERMOSTAT_HANDLE = api.exchange.get_actuator_handle(
        state, "Zone Temperature Control", "Cooling Setpoint", "LIVING ZONE"
    )

    try:
        response = httpx.post(
            f"{API_BASE}/agents/initialize",
            json={"stratum": "O1", "seed": 42},
            timeout=10.0,
        )
        response.raise_for_status()
        AGENT_ID = response.json()["agent_id"]
    except Exception as exc:
        raise RuntimeError(
            f"BuildOcc API initialization failed ({API_BASE}): {exc}"
        ) from exc


def occupant_callback(api: EnergyPlusAPI, state: object) -> None:
    """
    Called at each zone timestep. Reads zone conditions, asks the BuildOcc
    agent what to do, and writes the thermostat setpoint back to EnergyPlus.

    Register this as a callback:
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, occupant_callback)
    """
    if AGENT_ID is None:
        return
    if _ZONE_TEMP_HANDLE < 0 or _OUTDOOR_TEMP_HANDLE < 0:
        return  # handles not resolved — variable names may not match the IDF

    zone_temp_c = api.exchange.get_variable_value(state, _ZONE_TEMP_HANDLE)
    outdoor_temp_c = api.exchange.get_variable_value(state, _OUTDOOR_TEMP_HANDLE)

    # current_sim_time() returns elapsed hours as float; convert to ISO datetime for the API
    sim_hours = api.exchange.current_sim_time(state)
    sim_start = datetime(2025, 1, 1)  # replace with your EnergyPlus run period start date
    current_time = (sim_start + timedelta(hours=sim_hours)).isoformat()

    env_state = {
        "timestep": current_time,
        "zone_temp_c": zone_temp_c,
        "outdoor_temp_c": outdoor_temp_c,
        "tou_rate": 0.12,  # replace with live TOU rate lookup
        "devices": [],     # populate from EnergyPlus actuator handles
        "rooms": [],
    }

    try:
        response = httpx.post(
            f"{API_BASE}/agents/{AGENT_ID}/step",
            json={"environment": env_state},
            timeout=30.0,
        )
        response.raise_for_status()
        action = response.json()
    except Exception as exc:
        print(f"[BuildOcc] /step failed: {exc} — skipping this timestep")
        return

    # Write thermostat setpoint back to EnergyPlus (API returns °C; write directly)
    if (
        action["action_type"] == "adjust_thermostat"
        and action["value"] is not None
        and _THERMOSTAT_HANDLE >= 0
    ):
        api.exchange.set_actuator_value(state, _THERMOSTAT_HANDLE, float(action["value"]))


def main() -> None:
    """
    Example entry point showing how to register the callbacks and run EnergyPlus.
    Replace the idf_path and epw_path with your own files.
    """
    import sys

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    api.runtime.callback_begin_new_environment(state, initialize)
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, occupant_callback)

    exit_code = api.runtime.run_energyplus(state, [
        "-w", "path/to/weather.epw",
        "-d", "output/",
        "path/to/model.idf",
    ])
    api.state_manager.delete_state(state)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
