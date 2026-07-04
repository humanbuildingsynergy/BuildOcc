"""
buildocc: ATUS-grounded LLM occupant agents for building energy simulation.

Quick start::

    from occupant_agent import OccupantAgent, ActivityScheduler, SimulationEnvironment
    from occupant_agent import DeviceState, RoomState, summer_day_temp, peak_tou_rate
    from datetime import datetime

    agent = OccupantAgent.from_stratum("O1", seed=42)
    scheduler = ActivityScheduler(stratum="O1", seed=42)

    sim = SimulationEnvironment(
        initial_devices=[DeviceState(device_id="hvac", state=True, power_w=3500)],
        initial_rooms=[RoomState(room_id="living_room", occupied=True)],
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )

    timestep = datetime(2025, 8, 11, 16, 0)
    zone_temp_c = 22.0   # from your building platform (EnergyPlus, Home Assistant, etc.)
    env = sim.observe(timestep, zone_temp_c)
    action = agent.step(env, atus_code=scheduler.sample(timestep))
    sim.apply(action, timestep)

    print(action.action_type, action.reasoning)

Platform extension (define a new persona or scheduler)::

    from occupant_agent.core import BasePersona, register_stratum
    import random

    @register_stratum("P5")
    class LowIncomeElderly(BasePersona):
        def __init__(self, seed=None, **kw):
            self._age = random.Random(seed).randint(65, 80)
        @property
        def stratum(self): return "P5"
        @property
        def age(self): return self._age
        @property
        def sex(self): return "female"
        @property
        def income_bracket(self): return 3
        @property
        def work_from_home(self): return False
        @property
        def home_gym(self): return False
        @property
        def wfh_probability(self): return 0.0
        @property
        def comfort_band_c(self): return 2.2
        @property
        def appliances(self): return {"hvac", "thermostat", "tv", "refrigerator"}
        @property
        def schedule_priors(self): return {}
        @property
        def core_memory_text(self): return f"I am a {self._age}-year-old woman living alone on a fixed income."
        def sample_wfh_today(self, rng): return False

    agent = OccupantAgent.from_stratum("P5", seed=42)

REST API::

    buildocc-api  (or uvicorn occupant_agent.api.app:app --port 8000)

MCP server::

    BUILDOCC_API_URL=http://localhost:8000 buildocc-mcp
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("buildocc")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

# ── Core agent ────────────────────────────────────────────────────────────────
from occupant_agent.agent.memory import MemoryEntry, MemoryStream
from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.agent.persona import ROOM_DEFAULTS, Persona, create_persona

# ── Analysis ──────────────────────────────────────────────────────────────────
from occupant_agent.analysis import (
    SimulationLog,
    compute_cvrmse,
    compute_kl,
    compute_kl_by_hour,
    compute_ks,
    compute_mbe,
)

# ── Platform extension APIs ────────────────────────────────────────────────────
from occupant_agent.core import (
    BaseMemoryStream,
    BasePersona,
    BaseScheduler,
    get_scheduler,
    get_stratum,
    list_schedulers,
    list_strata,
    register_scheduler,
    register_stratum,
)
from occupant_agent.environment.simulation import (
    DEVICE_POWER_W,
    SimulationEnvironment,
    constant_zone_temp,
    outdoor_temp_from_csv,
    outdoor_temp_from_epw,
    peak_tou_rate,
    persona_devices,
    summer_day_temp,
    tou_rate_from_csv,
    typical_household_devices,
    typical_household_rooms,
    zone_temp_from_csv,
)

# ── Environment ───────────────────────────────────────────────────────────────
from occupant_agent.environment.state import (
    AgentAction,
    DeviceState,
    EnvironmentState,
    RoomState,
    SignalResponse,
)
from occupant_agent.grounding.activity_code_map import lookup, resolve_occupancy

# ── Grounding ─────────────────────────────────────────────────────────────────
from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
from occupant_agent.grounding.scheduler import ActivityScheduler

# ── Persistence ───────────────────────────────────────────────────────────────
from occupant_agent.persistence.store import AgentStore

__all__ = [
    "__version__",
    # Agent
    "OccupantAgent",
    "Persona",
    "create_persona",
    "ROOM_DEFAULTS",
    "MemoryEntry",
    "MemoryStream",
    # Environment
    "EnvironmentState",
    "AgentAction",
    "SignalResponse",
    "DeviceState",
    "RoomState",
    "DEVICE_POWER_W",
    "SimulationEnvironment",
    "persona_devices",
    "summer_day_temp",
    "peak_tou_rate",
    "zone_temp_from_csv",
    "typical_household_devices",
    "typical_household_rooms",
    "outdoor_temp_from_csv",
    "outdoor_temp_from_epw",
    "tou_rate_from_csv",
    "constant_zone_temp",
    # Grounding
    "ActivityScheduler",
    "FixedScheduleScheduler",
    "lookup",
    "resolve_occupancy",
    # Persistence
    "AgentStore",
    # Platform extension APIs
    "BasePersona",
    "BaseMemoryStream",
    "BaseScheduler",
    "register_stratum",
    "register_scheduler",
    "get_stratum",
    "get_scheduler",
    "list_strata",
    "list_schedulers",
    # Analysis
    "SimulationLog",
    "compute_kl",
    "compute_kl_by_hour",
    "compute_ks",
    "compute_cvrmse",
    "compute_mbe",
]
