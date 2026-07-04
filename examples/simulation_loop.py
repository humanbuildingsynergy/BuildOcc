"""
Standalone simulation loop — smoke test and usage reference for the full pipeline.

Demonstrates the complete ATUS-grounded LLM reasoning pipeline:
  1. ActivityScheduler samples ATUS codes from empirical time-at-activity distributions
  2. SimulationEnvironment maintains device-state continuity
  3. OccupantAgent.step() synthesizes persona, memory, and environment → action
  4. receive_signal() handles Type A/B/C signals with persona-coherent reasoning
  5. WFH flag is sampled once per simulated day and passed to step()

Zone temperature is supplied externally (here as a fixed test value). In a real
integration, it comes from the building energy platform (EnergyPlus, Home Assistant,
ecobee API, etc.) at each timestep.

Run from repo root (requires ANTHROPIC_API_KEY in .env):
    python3 examples/simulation_loop.py

Offline / no API key (deterministic mock responses):
    python3 examples/simulation_loop.py --mock --steps 4

Optional flags:
    --stratum O1|O2|O3|O4          (default: O1)
    --seed 42                      (default: 42, for reproducibility)
    --provider anthropic|openai|google|ollama
    --steps N                      (default: 8, one per 15-min interval)
    --start-hour H                 (default: 18, i.e. 6pm)
    --zone-temp C                  (default: 25.0°C — supply from your platform)
    --hardcode                     use hardcoded ATUS codes instead of scheduler
    --mock                         use MockLLMAgent (no API key needed; for CI/testing)
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.testing import MockLLMAgent
from occupant_agent.environment.simulation import (
    SimulationEnvironment,
    peak_tou_rate,
    persona_devices,
    summer_day_temp,
)
from occupant_agent.environment.state import RoomState
from occupant_agent.grounding.activity_code_map import lookup
from occupant_agent.grounding.scheduler import ActivityScheduler



# ── Hardcoded ATUS codes (fallback when scheduler unavailable) ─────────────────

_HARDCODED_CODES = [
    "180101",  # travel
    "020101",  # food prep
    "110101",  # eating
    "120303",  # computer leisure
    "120301",  # TV
    "020202",  # laundry
    "010101",  # sleeping
    "010101",
]


# ── Main simulation ───────────────────────────────────────────────────────────

def run_simulation(
    stratum: str = "O1",
    seed: int = 42,
    provider: str = "anthropic",
    n_steps: int = 8,
    start_hour: int = 18,
    zone_temp_c: float = 25.0,
    use_scheduler: bool = True,
    mock: bool = False,
) -> None:
    print(f"\n{'='*65}")
    mode = "MOCK (no LLM)" if mock else f"provider={provider}"
    print(f"BuildOcc Simulation — {stratum} (seed={seed}, {mode})")
    grounding = "ATUS scheduler" if use_scheduler else "hardcoded codes"
    print(f"Activity grounding: {grounding}")
    print(f"Zone temp supplied: {zone_temp_c}°C (fixed test value — replace with platform)")
    print(f"{'='*65}\n")

    # ── Agent ─────────────────────────────────────────────────────────────────
    if mock:
        agent = MockLLMAgent.from_stratum(stratum, seed=seed)
    else:
        agent = OccupantAgent.from_stratum(stratum, seed=seed, llm_provider=provider)

    print("PERSONA:")
    print("-" * 65)
    print(agent.persona.core_memory_text)
    print(f"\n  income_bracket={agent.persona.income_bracket}  "
          f"comfort_band={agent.persona.comfort_band_c:.1f}°C")
    print("-" * 65)

    # ── Activity scheduler ────────────────────────────────────────────────────
    scheduler: ActivityScheduler | None = None
    if use_scheduler:
        try:
            scheduler = ActivityScheduler(stratum=stratum, seed=seed)
            print(f"\nATUS scheduler loaded — sampling from empirical distributions.\n")
        except FileNotFoundError as e:
            print(f"\nWarning: {e}\nFalling back to hardcoded codes.\n")

    # ── Simulation environment ────────────────────────────────────────────────
    start_room = "living_room"
    sim = SimulationEnvironment(
        initial_devices=persona_devices(agent.persona),
        initial_rooms=[
            RoomState(room_id=r, occupied=(r == start_room))
            for r in agent.persona.room_ids
        ],
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )

    base_time = datetime(2024, 7, 15, start_hour, 0)
    signal_step = 2  # inject Type B signal at step index 2

    # WFH: sample once per simulated day (not per step)
    rng = random.Random(seed)
    wfh_today = agent.persona.sample_wfh_today(rng)
    current_date = base_time.date()
    if wfh_today:
        print(f"  Today is a WFH day for this agent.\n")
    elif agent.persona.wfh_probability > 0:
        print(f"  Today is an in-office day for this agent.\n")

    # ── Simulation loop ───────────────────────────────────────────────────────
    for i in range(n_steps):
        timestep = base_time + timedelta(minutes=15 * i)

        # Re-sample WFH if the simulated date rolls over to a new day
        if timestep.date() != current_date:
            current_date = timestep.date()
            wfh_today = agent.persona.sample_wfh_today(rng)
            status = "WFH" if wfh_today else "in-office"
            print(f"\n  [New day: {current_date} — {status}]")

        # Sample activity
        if scheduler:
            atus_code = scheduler.sample(timestep)
            label = lookup(atus_code).description
        else:
            atus_code = _HARDCODED_CODES[min(i, len(_HARDCODED_CODES) - 1)]
            label = lookup(atus_code).description

        # zone_temp_c comes from building platform; here we use a fixed test value
        env = sim.observe(timestep, zone_temp_c)

        rate_str = "PEAK" if env.tou_rate > 0.15 else "off-peak"
        print(f"\n{'─'*65}")
        print(f"STEP {i+1}/{n_steps}  {timestep.strftime('%H:%M')}  {label}")
        print(f"  ATUS {atus_code} | zone {env.zone_temp_c}°C / outdoor {env.outdoor_temp_c}°C"
              f" | ${env.tou_rate:.2f}/kWh ({rate_str})"
              f" | {sum(1 for d in env.devices if d.state is True)} device(s) ON")
        print(f"{'─'*65}")

        # Optionally deliver Type B signal at step 3
        if i == signal_step:
            sig_content = (
                "Your HVAC is running during peak hours (4–9pm) at $0.22/kWh — "
                "3× the off-peak rate. Raising the setpoint by 1°C saves ~$0.35 today."
            )
            print(f"\n  [SIGNAL TYPE B — Educational]")
            print(f"  {sig_content}")
            print("  Reasoning...")
            sig_response = agent.receive_signal("B", sig_content, env)
            print(f"  → {sig_response.response.upper()}: {sig_response.reasoning or ''}")

        # Agent step — pass wfh_today so the LLM knows the day type
        print("\n  Deciding what to do...")
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SIMULATION COMPLETE")
    print(f"  Timesteps run:        {n_steps}")
    print(f"  Memories accumulated: {len(agent.memory.entries)}")
    reflections = [e for e in agent.memory.entries if e.memory_type == "reflection"]
    print(f"  Reflections:          {len(reflections)}")

    if reflections:
        print("\nINSIGHTS FROM REFLECTION:")
        for r in reflections:
            print(f"  • {r.content}")

    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="BuildOcc simulation loop")
    parser.add_argument("--stratum", default="O1", choices=["O1", "O2", "O3", "O4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai", "google", "ollama"])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--start-hour", type=int, default=18, dest="start_hour")
    parser.add_argument("--zone-temp", type=float, default=25.0, dest="zone_temp_c",
                        help="Indoor zone temperature in °C (from your building platform)")
    parser.add_argument("--hardcode", action="store_true",
                        help="Use hardcoded ATUS codes instead of scheduler")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockLLMAgent — no API key needed (for CI / offline testing)")
    args = parser.parse_args()

    run_simulation(
        stratum=args.stratum,
        seed=args.seed,
        provider=args.provider,
        n_steps=args.steps,
        start_hour=args.start_hour,
        zone_temp_c=args.zone_temp_c,
        use_scheduler=not args.hardcode,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
