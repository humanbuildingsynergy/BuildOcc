"""
Command-line interface for BuildOcc.

Commands
--------
buildocc init [config.yaml]
    Write a template configuration file with sensible defaults.
    Users can then edit the file and run the simulation without writing Python.

buildocc run config.yaml [--steps N] [--mock]
    Run a simulation from a YAML configuration file.
    --mock skips the LLM call (no API key required) for quick testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Template configuration ────────────────────────────────────────────────────

_TEMPLATE = """\
# BuildOcc simulation configuration
# Run with: buildocc run <this file>
#
# Edit any field below; all values shown are the defaults.
# Lines starting with # are comments and are ignored.

# ── Agent ────────────────────────────────────────────────────────────────────
stratum: O1            # Demographic profile: O1 | O2 | O3 | O4
seed: 42               # Random seed for reproducibility
llm_provider: anthropic  # LLM backend: anthropic | openai | google | ollama

# ── Simulation ────────────────────────────────────────────────────────────────
steps: 8                              # Number of 15-minute timesteps to simulate
start_datetime: "2025-08-10 18:00"   # Simulation start (YYYY-MM-DD HH:MM)
thermostat_setpoint: 22.0            # Initial comfort setpoint (°C)

# ── Device inventory (appliances the agent can control) ───────────────────────
devices:
  - id: hvac
    on: true
    power_w: 3500
  - id: tv
    on: false
    power_w: 150
  - id: washer
    on: false
    power_w: 500
  - id: dryer
    on: false
    power_w: 5000
  - id: dishwasher
    on: false
    power_w: 1200
  - id: refrigerator
    on: true
    power_w: 150
  - id: microwave
    on: false
    power_w: 1100

# ── Rooms ─────────────────────────────────────────────────────────────────────
rooms:
  - living_room
  - kitchen
  - bedroom
  - laundry_room

# ── Zone (indoor) temperature source ─────────────────────────────────────────
# mode: constant  — use a fixed value every timestep (good for behavioral tests)
# mode: csv       — load from a pre-computed CSV file (requires csv_path below)
zone_temperature:
  mode: constant
  value: 25.0
  # csv_path: examples/data/zone_temps_sample.csv

# ── Outdoor temperature source ────────────────────────────────────────────────
# mode: summer_default  — sinusoidal summer profile peaking at 35°C at 6pm
# mode: constant        — use a fixed value (requires value below)
outdoor_temperature:
  mode: summer_default
  # value: 35.0

# ── Time-of-use electricity rate ──────────────────────────────────────────────
# mode: peak_default  — $0.22/kWh 4–9pm, $0.08/kWh otherwise
# mode: constant      — fixed rate (requires value below)
tou_rate:
  mode: peak_default
  # value: 0.12
"""


def cmd_init(args: argparse.Namespace) -> None:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"File already exists: {out}  (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    out.write_text(_TEMPLATE)
    print(f"Template written to: {out}")
    print(f"Edit it, then run:  buildocc run {out}")


def cmd_run(args: argparse.Namespace) -> None:
    try:
        import yaml
    except ImportError:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Invalid YAML in {cfg_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    import random
    from datetime import datetime, timedelta

    from occupant_agent.agent.occupant import OccupantAgent
    from occupant_agent.environment.simulation import (
        SimulationEnvironment,
        constant_zone_temp,
        peak_tou_rate,
        summer_day_temp,
        typical_household_devices,
        typical_household_rooms,
        zone_temp_from_csv,
    )
    from occupant_agent.environment.state import DeviceState, RoomState
    from occupant_agent.grounding.activity_code_map import lookup
    from occupant_agent.grounding.scheduler import ActivityScheduler

    # ── Config ────────────────────────────────────────────────────────────────
    stratum  = cfg.get("stratum", "O1")
    seed     = cfg.get("seed", 42)
    provider  = cfg.get("llm_provider", "anthropic")
    llm_model = cfg.get("llm_model")
    n_steps   = args.steps if args.steps is not None else cfg.get("steps", 8)
    if n_steps <= 0:
        print(f"--steps must be >= 1 (got {n_steps})", file=sys.stderr)
        sys.exit(1)
    mock     = args.mock
    setpoint = float(cfg.get("thermostat_setpoint", 22.0))
    base_time = datetime.strptime(cfg.get("start_datetime", "2025-08-10 18:00"), "%Y-%m-%d %H:%M")

    # ── Devices ───────────────────────────────────────────────────────────────
    raw_devices = cfg.get("devices", [])
    devices = (
        [DeviceState(device_id=d["id"], state=bool(d.get("on", False)), power_w=float(d.get("power_w", 0)))
         for d in raw_devices]
        if raw_devices else typical_household_devices()
    )

    # ── Rooms ─────────────────────────────────────────────────────────────────
    raw_rooms = cfg.get("rooms", [])
    rooms = (
        [RoomState(room_id=r, occupied=False) for r in raw_rooms]
        if raw_rooms else typical_household_rooms()
    )

    # ── Zone temperature callable ─────────────────────────────────────────────
    zt = cfg.get("zone_temperature", {})
    if zt.get("mode") == "csv":
        csv_path = zt.get("csv_path")
        if not csv_path:
            print("zone_temperature.csv_path is required when mode is 'csv'", file=sys.stderr)
            sys.exit(1)
        zone_temp_fn = zone_temp_from_csv(csv_path)
    else:
        zone_temp_fn = constant_zone_temp(float(zt.get("value", 25.0)))

    # ── Outdoor temperature callable ──────────────────────────────────────────
    ot = cfg.get("outdoor_temperature", {})
    if ot.get("mode", "summer_default") == "summer_default":
        outdoor_fn = summer_day_temp
    else:
        _ot_val = float(ot.get("value", 35.0))
        outdoor_fn = lambda ts: _ot_val  # noqa: E731

    # ── TOU rate callable ─────────────────────────────────────────────────────
    tou = cfg.get("tou_rate", {})
    if tou.get("mode", "peak_default") == "peak_default":
        tou_fn = peak_tou_rate
    else:
        _tou_val = float(tou.get("value", 0.12))
        tou_fn = lambda ts: _tou_val  # noqa: E731

    # ── Setup ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"BuildOcc  stratum={stratum}  seed={seed}  provider={'MOCK' if mock else provider}")
    print(f"{'='*65}\n")

    if mock:
        from occupant_agent.testing import MockLLMAgent
        agent = MockLLMAgent.from_stratum(stratum, seed=seed)
    else:
        agent = OccupantAgent.from_stratum(stratum, seed=seed, llm_provider=provider, llm_model=llm_model)

    print("PERSONA:")
    print("-" * 65)
    print(agent.persona.core_memory_text)
    print("-" * 65)

    scheduler = ActivityScheduler(stratum=stratum, seed=seed)

    sim = SimulationEnvironment(
        initial_devices=devices,
        initial_rooms=rooms,
        thermostat_setpoint=setpoint,
        outdoor_temp_fn=outdoor_fn,
        tou_rate_fn=tou_fn,
    )

    rng = random.Random(seed)
    wfh_today = agent.persona.sample_wfh_today(rng)
    current_date = base_time.date()

    # ── Simulation loop ───────────────────────────────────────────────────────
    for i in range(n_steps):
        timestep = base_time + timedelta(minutes=15 * i)

        if timestep.date() != current_date:
            current_date = timestep.date()
            wfh_today = agent.persona.sample_wfh_today(rng)

        atus_code = scheduler.sample(timestep)
        label = lookup(atus_code).description
        zone_c = zone_temp_fn(timestep)
        env = sim.observe(timestep, zone_c)

        rate_str = "PEAK" if env.tou_rate > 0.15 else "off-peak"
        print(f"\n{'─'*65}")
        print(f"STEP {i+1}/{n_steps}  {timestep.strftime('%H:%M')}  {label}")
        print(f"  zone {env.zone_temp_c}°C | outdoor {env.outdoor_temp_c}°C"
              f" | ${env.tou_rate:.2f}/kWh ({rate_str})")

        action = agent.step(env, atus_code=atus_code, wfh_today=wfh_today)

        print(f"  Action: {action.action_type}", end="")
        if action.target_id:
            print(f" → {action.target_id}", end="")
        if action.value is not None:
            print(f" = {action.value}", end="")
        print()
        if action.reasoning:
            print(f"  Reasoning: {action.reasoning}")

        sim.apply(action, timestep)

    print(f"\n{'='*65}")
    print(f"Done. {n_steps} timesteps | {len(agent.memory.entries)} memories accumulated.")
    reflections = [e for e in agent.memory.entries if e.memory_type == "reflection"]
    if reflections:
        print("\nINSIGHTS FROM REFLECTION:")
        for r in reflections:
            print(f"  • {r.content}")
    print(f"{'='*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="buildocc",
        description="BuildOcc — LLM occupant agent platform for building energy simulation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init subcommand
    p_init = sub.add_parser("init", help="Write a template configuration file")
    p_init.add_argument(
        "output", nargs="?", default="buildocc_config.yaml",
        help="Output path (default: buildocc_config.yaml)",
    )
    p_init.add_argument("--force", action="store_true", help="Overwrite existing file")

    # run subcommand
    p_run = sub.add_parser("run", help="Run a simulation from a YAML configuration file")
    p_run.add_argument("config", help="Path to YAML configuration file")
    p_run.add_argument("--steps", type=int, default=None,
                       help="Override number of timesteps from config")
    p_run.add_argument("--mock", action="store_true",
                       help="Use mock LLM — no API key required (for quick testing)")

    args = parser.parse_args()
    {"init": cmd_init, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
