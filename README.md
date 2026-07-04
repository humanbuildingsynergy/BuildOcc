# BuildOcc

ATUS-grounded LLM occupant agents for building energy simulation.

Agents are initialized from American Time Use Survey (ATUS) population microdata, reason over environment state using an LLM at each 15-minute timestep, and accumulate a memory stream that enables persistent behavior change. The library is exposed through a three-layer platform interface — Python library, REST API, and MCP server — so any building energy tool can integrate an occupant behavioral layer without bespoke coupling code.

## Architecture

```
┌─────────────────────────────────────────┐
│  Layer 3 — MCP Server                   │  ← Claude / LLM orchestrators,
│  (tool-calling interface for LLM apps)  │     Home Assistant (native MCP client)
└────────────────┬────────────────────────┘
                 │ wraps
┌────────────────▼────────────────────────┐
│  Layer 2 — REST API (FastAPI)           │  ← EnergyPlus Python callbacks,
│  (standard HTTP, platform-agnostic)     │     VOLTTRON, OpenStudio, any script
└────────────────┬────────────────────────┘
                 │ calls
┌────────────────▼────────────────────────┐
│  Layer 1 — Python Agent Library         │  ← Direct import, unit tests,
│  (core logic, no network dependency)    │     research scripts
└─────────────────────────────────────────┘
```

## Requirements

- Python ≥ 3.11
- An LLM API key — set one of:
  - `ANTHROPIC_API_KEY` (default provider, `claude-haiku-4-5`)
  - `OPENAI_API_KEY` (provider `openai`, `gpt-4o-mini`)
  - `GOOGLE_API_KEY` (provider `google`, `gemini-2.0-flash`) — install: `pip install "buildocc[google]"`
  - Ollama running locally (provider `ollama`, `llama3.2`) — no extra package needed

## Installation

```bash
git clone https://github.com/humanbuildingsynergy/BuildOcc
cd BuildOcc
pip install -e ".[dev]"

# Set your API key — the library loads .env automatically via python-dotenv
cp .env.example .env
# Then edit .env and replace the placeholder with your real key
```

## Quick Start

### Layer 1 — Python library (direct import)

```python
from datetime import datetime
from occupant_agent import OccupantAgent, DeviceState, EnvironmentState, RoomState, ActivityScheduler

# Create an ATUS-grounded agent for an employed single adult (25–44)
agent = OccupantAgent.from_stratum("O1", seed=42, llm_provider="anthropic")

# Sample the agent's current activity from empirical ATUS distributions
scheduler = ActivityScheduler(stratum="O1", seed=42)
timestep = datetime(2024, 7, 15, 19, 0)
atus_code = scheduler.sample(timestep)  # e.g. "120303" — computer leisure

# Build an environment state
env = EnvironmentState(
    timestep=timestep,
    zone_temp_c=25.0,
    outdoor_temp_c=34.0,
    tou_rate=0.22,  # peak rate $/kWh
    devices=[
        DeviceState(device_id="hvac",   state=True,  power_w=3500),
        DeviceState(device_id="washer", state=False, power_w=500),
    ],
    rooms=[RoomState(room_id="living_room", occupied=True)],
)

# Step: agent reasons over persona + memory + environment → action
action = agent.step(env, atus_code=atus_code)
print(action.action_type, action.target_id, action.reasoning)
# e.g. "toggle_device" "hvac" "Peak rate is $0.22/kWh and I'm comfortable..."

# Send a demand-response signal (Type B — educational)
response = agent.receive_signal(
    signal_type="B",
    content="Your HVAC costs 3× more before 9pm. Raising the setpoint by 2°C saves ~$0.35 today.",
    env=env,
)
print(response.response, response.reasoning)
# e.g. "accepted" "Makes economic sense and I'm not uncomfortable at 26°C..."
```

**Signal types:**
| Type | Label | Description |
|------|-------|-------------|
| A | Direct command | "Turn off the dishwasher until 9pm" |
| B | Competence-building (boost) | Price/cost explanation for why a change helps |
| C | Social norm (nudge) | Comparison to similar households |

**Demographic strata:**
| ID | ATUS Stratum |
|----|-------------|
| O1 | Employed adult, single, 25–44 |
| O2 | Retired couple, 65+ |
| O3 | Employed parent with children, 35–54 |
| O4 | Unemployed adult, 25–44 |

### Full simulation loop

The library reads `ANTHROPIC_API_KEY` (or whichever provider key) directly from `.env` — no shell export needed.

```bash
# Runs 8 timesteps of a O1 evening with ATUS-sampled activities + a Type B signal
python3 examples/simulation_loop.py --stratum O1 --seed 42 --steps 8

# Other providers
python3 examples/simulation_loop.py --stratum O2 --provider openai --steps 16 --start-hour 6
python3 examples/simulation_loop.py --provider google --stratum O3   # Google Gemini
python3 examples/simulation_loop.py --provider ollama --stratum O1   # local Ollama

# Offline / no API key (deterministic mock responses — useful for CI and testing)
python3 examples/simulation_loop.py --mock --stratum O1 --steps 4

# Other flags
python3 examples/simulation_loop.py --zone-temp 78.5   # override zone temp (°C)
python3 examples/simulation_loop.py --hardcode          # fixed ATUS codes (ablation baseline)
```

### Demand response signal demo

Shows all three signal types (A/B/C), cross-stratum comparison, and the `extra_context` kwarg:

```bash
python3 examples/signal_demo.py                         # all three parts, Anthropic
python3 examples/signal_demo.py --part 2                # stratum comparison only
python3 examples/signal_demo.py --provider openai --warmup 4
python3 examples/signal_demo.py --mock                  # offline mock mode
```

### Layer 2 — REST API

```bash
# Start the server
ANTHROPIC_API_KEY=... buildocc-api
# or: uvicorn occupant_agent.api.app:app --reload --port 8000
```

```bash
# Initialize an agent
curl -s -X POST http://localhost:8000/agents/initialize \
  -H "Content-Type: application/json" \
  -d '{"stratum": "O1", "seed": 42}' | python3 -m json.tool
# → {"agent_id": "...", "stratum": "O1", "seed": 42}

# Step (replace AGENT_ID)
curl -s -X POST http://localhost:8000/agents/AGENT_ID/step \
  -H "Content-Type: application/json" \
  -d '{
    "environment": {
      "timestep": "2024-07-15T19:00:00",
      "zone_temp_c": 25.0, "outdoor_temp_c": 34.0, "tou_rate": 0.22,
      "devices": [{"device_id": "hvac", "state": true, "power_w": 3500}],
      "rooms": [{"room_id": "living_room", "occupied": true}]
    },
    "atus_code": "120303"
  }' | python3 -m json.tool

# Send a demand-response signal
curl -s -X POST http://localhost:8000/agents/AGENT_ID/signal \
  -H "Content-Type: application/json" \
  -d '{
    "signal_type": "B",
    "content": "Your HVAC costs 3x more before 9pm. Raising the setpoint 2°C saves ~$0.35 today.",
    "environment": {
      "timestep": "2024-07-15T17:00:00",
      "zone_temp_c": 25.0,
      "outdoor_temp_c": 34.0,
      "tou_rate": 0.22,
      "devices": [{"device_id": "hvac", "state": true, "power_w": 3500}],
      "rooms": [{"room_id": "living_room", "occupied": true}]
    }
  }' | python3 -m json.tool

# Check agent state (memory count, last action, reflection history)
curl -s http://localhost:8000/agents/AGENT_ID/state | python3 -m json.tool
```

**REST endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agents/initialize` | Create agent; returns `agent_id` |
| POST | `/agents/{id}/step` | Advance one 15-min timestep; returns `AgentAction` |
| POST | `/agents/{id}/signal` | Deliver A/B/C signal; returns `SignalResponse` |
| GET  | `/agents/{id}/state` | Memory count, last action, reflection history |
| GET  | `/agents/` | List all agents |
| DELETE | `/agents/{id}` | Delete agent and all records |
| GET  | `/health` | `{status: "ok", version: "1.0.0"}` |

### Layer 3 — MCP server

```bash
# Requires the REST API to be running first
BUILDOCC_API_URL=http://localhost:8000 buildocc-mcp
# or: python3 -m occupant_agent.mcp_server.server
```

MCP tools: `initialize_agent`, `step`, `send_signal`, `get_state`, `reset_agent`.

**Configure in Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "buildocc": {
      "command": "buildocc-mcp",
      "env": {
        "BUILDOCC_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

Start the REST API first (`buildocc-api`), then restart Claude Desktop. The `initialize_agent`, `step`, and `send_signal` tools will appear in Claude's tool list.

## Testing

```bash
# Unit tests (no API key needed — uses MockLLMAgent)
pytest tests/unit/

# Integration tests (no API key needed — uses MockLLMAgent end-to-end)
pytest tests/integration/

# Offline smoke-test of the simulation loop (--mock bypasses the LLM entirely)
python3 examples/simulation_loop.py --mock --stratum O1 --steps 4
python3 examples/signal_demo.py --mock --part 1
```

## How it works

### ATUS grounding
Agents are initialized from ATUS 2022+2023 microdata (16,684 respondents; 299,513 activity episodes) matched by demographic stratum. Activity scheduling uses *time-at-activity* distributions — the population-weighted fraction of respondents in each activity at each clock hour, computed via episode-overlap rather than start-time sampling. This avoids the start-time bias that over-represents long-duration activities (e.g., sleeping). Eight categories are modeled: sleeping, work, food prep, laundry, TV, eating, exercise, and other. Weekday and weekend distributions are computed separately (`TUDIARYDAY`) and the scheduler selects the appropriate distribution from `timestep.weekday()`.

### LLM reasoning at each timestep
`step()` synthesizes six inputs: the agent's persona (core memory), top-5 retrieved memories by recency + importance score, current environment state (temperatures, TOU rate, device states), current ATUS activity, time of day, and a per-day WFH flag (sampled from `persona.sample_wfh_today()` and resampled on date rollover). The LLM returns a structured action and a self-rated importance score (1–10) for the observation. Valid `target_id` values (device IDs and room IDs) are injected into the prompt schema so the LLM cannot hallucinate invalid targets.

### Persona diversity
Each `create_persona()` call samples a demographically grounded agent: income bracket drives `comfort_band_c` (°C deviation from setpoint before acting — wider for lower income, per RECS 2020), appliance ownership (scaled ±30% around RECS base priors by income position within stratum), and signal preference framing in the LLM prompt (dollar-savings vs. social-comparison language). Four LLM providers are supported for cost/privacy trade-offs: Anthropic, OpenAI, Google Gemini, and local Ollama.

### Memory stream and reflection
Follows Park et al. (2023). Retrieval score = `0.5 × recency + 0.5 × (importance/10)` with exponential recency decay (24-hour half-life). When the cumulative importance accumulator exceeds a threshold (default 100), reflection fires: the LLM synthesizes the last 30 memories into 3 high-level insights, stored as permanent memory entries. A more capable model is used for reflection; a cheaper model handles routine `step()` calls.

### Persistence
All agent state (persona, memory stream, action log, signal log) is persisted to SQLite via SQLAlchemy. Each API request loads fresh from the database and writes back — stateless HTTP with durable state. Multiple agents can run concurrently via `agent_id` (UUID4).

## How to extend

BuildOcc is designed as a community platform. You can add new demographic profiles, activity schedulers, and memory architectures without forking the repository.

### Define a new demographic profile (stratum)

Subclass `BasePersona` and register it under a key. Your class will be discoverable via `OccupantAgent.from_stratum()` and `list_strata()`.

```python
import random
from occupant_agent import OccupantAgent
from occupant_agent.core import BasePersona, register_stratum, list_strata

@register_stratum("P5")
class LowIncomeElderlyAlone(BasePersona):
    """Single elderly adult, low income — for low-income housing DR research."""

    def __init__(self, seed=None, **kwargs):
        self._age = random.Random(seed).randint(65, 80)

    @property
    def stratum(self): return "P5"
    @property
    def age(self): return self._age
    @property
    def sex(self): return "female"
    @property
    def income_bracket(self): return 3          # low income (HEFAMINC 1–16)
    @property
    def work_from_home(self): return False
    @property
    def home_gym(self): return False
    @property
    def wfh_probability(self): return 0.0
    @property
    def comfort_band_c(self): return 2.2        # cost-sensitive → wide tolerance
    @property
    def appliances(self): return {"hvac", "thermostat", "tv", "refrigerator"}
    @property
    def schedule_priors(self): return {}
    @property
    def core_memory_text(self):
        return (f"I am a {self._age}-year-old woman living alone on a fixed income. "
                "I keep my thermostat low to save money.")
    def sample_wfh_today(self, rng): return False

# Now usable everywhere:
agent = OccupantAgent.from_stratum("P5", seed=42)
print(list_strata())  # ["O1", "O2", "O3", "O4", "P5"]
```

### Define a new activity scheduler

Subclass `BaseScheduler` to ground the agent in a different data source (Homer, MTUS, synthetic, etc.).

```python
from datetime import datetime
from occupant_agent.core import BaseScheduler, register_scheduler

@register_scheduler("homer")
class HomerScheduler(BaseScheduler):
    """Activity grounding from Homer dataset (21-participant HAR corpus)."""

    def __init__(self, stratum=None, seed=None, **kwargs):
        ...  # load Homer diary records

    def sample(self, timestep: datetime) -> str:
        ...  # return 6-digit ATUS code

    def category_weights(self, hour: int, timestep=None) -> dict[str, float]:
        ...  # return {category: probability}

# Inject at agent creation:
agent = OccupantAgent.from_stratum("O1", seed=42, scheduler="homer")
```

### Define a new memory architecture

Subclass `BaseMemoryStream` to replace the retrieval algorithm (e.g., sentence-embedding similarity, graph-structured memory).

```python
from datetime import datetime
from collections.abc import Callable
from occupant_agent.core import BaseMemoryStream

class EmbeddingMemory(BaseMemoryStream):
    """Retrieval weighted by semantic similarity (sentence-transformers)."""

    def retrieve(self, query_time: datetime, k: int = 5):
        ...  # rank by cosine similarity to current context

# Inject at agent construction:
mem = EmbeddingMemory()
agent = OccupantAgent(persona=persona, memory=mem)
```

### Platform-specific extensions via EnvironmentState

Use `extensions: dict` to pass platform-specific data alongside the core schema without modifying it:

```python
env = EnvironmentState(
    timestep=ts, zone_temp_c=23.0, outdoor_temp_c=29.0, tou_rate=0.18,
    devices=[...], rooms=[...],
    extensions={
        "home_assistant": {
            "sensor.living_room_co2": 650,
            "binary_sensor.front_door": "off",
        },
        "energyplus": {"zone_air_humidity_ratio": 0.009},
    }
)
```

The core agent ignores `extensions`; platform-specific plugins can read it.

### Package your extension for distribution

```python
# your_package/__init__.py
from occupant_agent.core import register_stratum
from .personas import LowIncomeElderlyAlone, PublicHousingResident

register_stratum("P5")(LowIncomeElderlyAlone)
register_stratum("P6")(PublicHousingResident)
```

After `pip install your-package`, any script that imports it gains the new strata.

### Built-in extension point summary

| Base class | Register with | Discovered by |
|---|---|---|
| `BasePersona` | `@register_stratum("P5")` | `OccupantAgent.from_stratum("P5")`, `list_strata()` |
| `BaseScheduler` | `@register_scheduler("homer")` | `OccupantAgent.from_stratum(..., scheduler="homer")`, `list_schedulers()` |
| `BaseMemoryStream` | Inject directly into `OccupantAgent(memory=...)` | — |

---

## Regenerating ATUS outputs

Pre-generated ATUS outputs are bundled in `occupant_agent/data/`. Raw ATUS microdata is not committed to the repository — download it and place in `data/atus/{year}/extracted/` before running these scripts. To regenerate outputs after a data update:

```bash
python3 scripts/atus/analyze.py
# Produces scripts/atus/outputs/time_at_activity.csv       (hour, category, weighted_pct, stratum, day_type)
#          scripts/atus/outputs/activity_frequency_{O1,O2,O3,O4}.csv
#          scripts/atus/outputs/schedule_peak_hours.csv    (weekday/weekend peak hours per stratum)
# Then sync:
cp scripts/atus/outputs/time_at_activity.csv occupant_agent/data/
cp scripts/atus/outputs/schedule_peak_hours.csv occupant_agent/data/
```

Download ATUS microdata from [BLS ATUS](https://www.bls.gov/tus/data.htm) (`.dat` files, uppercase column headers) or [IPUMS Time Use](https://www.ipums.org/timeuse) (`.csv` files, lowercase column headers — both formats are supported automatically).

## Data

Raw datasets go in `data/`. See [docs/datasets_and_resources.md](docs/datasets_and_resources.md) for access and coverage details.

| Directory | Dataset | Purpose |
|-----------|---------|---------|
| `data/atus/` | ATUS 2022–23 (BLS) | Agent grounding and scheduling |
| `data/casas/` | CASAS Aruba/Milan (Zenodo) | Behavioral validation (Phase 2) |
| `data/ecobee/` | DOE ecobee 2017 (OSTI) | Thermostat validation (Phase 2) |
| `data/pecan_street/` | Pecan Street Dataport | Energy validation (Phase 2) |
| `data/recs/` | EIA RECS | Appliance ownership priors |

## Project structure

```
occupant_agent/
    core/
        base_persona.py    — BasePersona ABC (extension contract for new strata)
        base_memory.py     — BaseMemoryStream ABC (extension contract for memory backends)
        base_scheduler.py  — BaseScheduler ABC (extension contract for activity sources)
        registry.py        — Plugin registry: @register_stratum, @register_scheduler
    agent/
        occupant.py     — OccupantAgent: step(), receive_signal(), from_stratum()
        memory.py       — MemoryStream(BaseMemoryStream): retrieval, reflection
        persona.py      — Persona dataclass, create_persona(), ROOM_DEFAULTS
    grounding/
        scheduler.py         — ActivityScheduler(BaseScheduler): ATUS-grounded sampling
        fixed_schedule.py    — FixedScheduleScheduler: rule-based ablation baseline
        activity_code_map.py — ATUS tier-3 code → occupancy + device state mapping
    llm/
        client.py       — call_llm(): Anthropic / OpenAI / Google Gemini / Ollama
    environment/
        state.py        — Pydantic models: EnvironmentState (+ extensions), AgentAction
        simulation.py   — SimulationEnvironment: device/room/setpoint state
    persistence/
        store.py        — AgentStore: SQLite via SQLAlchemy
    analysis/
        simulation_log.py — SimulationLog: per-step records, CSV/JSON export
        metrics.py        — compute_kl, compute_ks, compute_cvrmse, compute_mbe
    testing/
        mock_llm.py     — MockLLMAgent: deterministic test double (no API key needed)
        fixtures.py     — make_env(), make_room() — factory helpers for unit tests
        conformance.py  — assert_persona_contract(), assert_scheduler_contract()
    cli.py              — Entry point for buildocc CLI command
    data/               — Bundled ATUS outputs (time_at_activity.csv, ...)
    api/
        app.py          — FastAPI REST API (Layer 2)
    mcp_server/
        server.py       — MCP server (Layer 3, wraps REST API)
scripts/
    evaluate.py            — Config-driven evaluation harness (KL, KS, action distribution)
    validate_strata.py     — Validate each demographic stratum across a batch of seeds
    validate_signals.py    — Validate Type A/B/C signal response rates per stratum
    extract_zone_temps.py  — Extract zone temperatures from EnergyPlus output CSV
    atus/
        parse.py        — ATUS microdata → per-respondent episode records
        analyze.py      — Episode records → time_at_activity.csv + frequency CSVs
data/atus/              — Raw ATUS 2022+2023 microdata (not committed to Git)
examples/
    simulation_loop.py      — ATUS-grounded evening simulation with Type B signal
    signal_demo.py          — Demand response signal reference: all three types + stratum comparison
    energyplus_callback.py  — EnergyPlus Python API integration pattern
docs/
    datasets_and_resources.md — Dataset access, coverage, and caveats
    methodology_decisions.md  — Design decision log (D1–D13)
```

## Development plan

v1 is complete. Phase 2 (validation against CASAS and Pecan Street behavioral data) is next.

## Citation

```bibtex
@article{jung2026occupantagent,
  title   = {BuildOcc: A Large Language Model Occupant Agent Platform for Building Energy Research},
  author  = {Jung, Wooyoung},
  journal = {SoftwareX},
  year    = {2026},
  note    = {Under review}
}
```

## License

[Apache License 2.0](LICENSE)
