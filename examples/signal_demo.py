"""
Demand response signal demo — usage reference for OccupantAgent.receive_signal().

BuildOcc lets any external system — a building management system, an EnergyPlus
co-simulation script, or a researcher's analysis loop — deliver demand response
signals to a simulated occupant and observe how the agent responds.  This script
demonstrates three use cases:

  Part 1 — All three signal types (A, B, C) sent to the same agent
  Part 2 — The same signal sent to all four demographic strata (O1–O4)
  Part 3 — Signals with extra situational context (extra_context kwarg)

Signal typology
───────────────
  Type A  Direct command    "Turn off your HVAC for 30 minutes."
  Type B  Educational       Explains cost or mechanism behind the request.
  Type C  Social norm       Compares occupant to similar households.

The agent returns a SignalResponse with:
  response  — "accepted" | "rejected" | "deferred"
  reasoning — the agent's natural-language explanation

Run from repo root (requires ANTHROPIC_API_KEY in .env):
    python3 examples/signal_demo.py

Offline / no API key (deterministic mock responses):
    python3 examples/signal_demo.py --mock

Optional flags:
    --provider anthropic|openai|google|ollama   (default: anthropic)
    --seed 42                                   (default: 42)
    --warmup N                                  (default: 6 steps = 90 min)
    --mock                                      use MockLLMAgent (no API key needed)
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.testing import MockLLMAgent
from occupant_agent.environment.simulation import (
    SimulationEnvironment,
    peak_tou_rate,
    persona_devices,
    summer_day_temp,
    zone_temp_from_csv,
)
from occupant_agent.environment.state import RoomState
from occupant_agent.grounding.scheduler import ActivityScheduler

# Warmup starts at 3 pm so the agent enters peak hours (4–9 pm) with context
_WARMUP_START = datetime(2025, 8, 11, 15, 0)

# Three standardised signals — identical content used across all strata
SIGNALS: dict[str, dict[str, str]] = {
    "A": {
        "label": "Direct command",
        "content": (
            "Please turn off your HVAC for the next 30 minutes to support "
            "grid stability during today's peak demand event."
        ),
    },
    "B": {
        "label": "Educational / price signal",
        "content": (
            "Your HVAC is running during peak hours at $0.22/kWh — approximately "
            "3× the off-peak rate of $0.08/kWh. Raising your thermostat setpoint "
            "by 2°C for the next two hours saves an estimated $0.35 today."
        ),
    },
    "C": {
        "label": "Social norm nudge",
        "content": (
            "75% of similar households in your area have already reduced their "
            "HVAC use during today's peak demand event."
        ),
    },
}

STRATA_LABELS = {
    "O1": "Employed single adult",
    "O2": "Retired couple",
    "O3": "Employed parent",
    "O4": "Unemployed adult",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_sim(agent) -> SimulationEnvironment:
    """Build a simulation environment matched to the agent's persona."""
    start_room = "living_room"
    return SimulationEnvironment(
        initial_devices=persona_devices(agent.persona),
        initial_rooms=[
            RoomState(room_id=r, occupied=(r == start_room))
            for r in agent.persona.room_ids
        ],
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )


def _warmup(
    agent: OccupantAgent,
    sim: SimulationEnvironment,
    zone_temp_fn,
    n_steps: int,
    seed: int = 42,
) -> datetime:
    """Run n_steps to build agent memory; return the next timestep."""
    from occupant_agent.environment.state import AgentAction

    scheduler = ActivityScheduler(stratum=agent.persona.stratum, seed=seed)
    rng = random.Random(seed)
    wfh_today = agent.persona.sample_wfh_today(rng)
    timestep = _WARMUP_START
    step_delta = timedelta(minutes=15)

    print(f"  Warming up ({n_steps} steps, {_WARMUP_START.strftime('%H:%M')}–", end="")
    for _ in range(n_steps):
        env = sim.observe(timestep, zone_temp_fn(timestep))
        atus_code = scheduler.sample(timestep)
        try:
            action = agent.step(env, atus_code=atus_code, wfh_today=wfh_today)
        except Exception:
            action = AgentAction(action_type="do_nothing", reasoning="warmup")
        sim.apply(action, timestep)
        timestep += step_delta
    print(f"{timestep.strftime('%H:%M')})  memories={len(agent.memory.entries)}")
    return timestep


def _deliver(
    agent: OccupantAgent,
    sim: SimulationEnvironment,
    zone_temp_fn,
    timestep: datetime,
    signal_type: str,
    extra_context: str | None = None,
) -> None:
    sig = SIGNALS[signal_type]
    env = sim.observe(timestep, zone_temp_fn(timestep))
    rate = "PEAK" if env.tou_rate > 0.15 else "off-peak"
    print(f"\n  Signal {signal_type} ({sig['label']})  [{timestep.strftime('%H:%M')}, {rate}]")
    print(f"  ┌─ Content: {sig['content'][:90]}{'…' if len(sig['content']) > 90 else ''}")
    if extra_context:
        print(f"  ├─ Context: {extra_context}")

    response = agent.receive_signal(
        signal_type=signal_type,
        content=sig["content"],
        env=env,
        extra_context=extra_context,
    )
    icon = {"accepted": "✓", "rejected": "✗", "deferred": "~"}.get(response.response, "?")
    print(f"  └─ {icon} {response.response.upper()}: {response.reasoning or ''}")


# ── Part 1: all three signal types → single agent ────────────────────────────

def demo_signal_types(provider: str, seed: int, warmup: int, mock: bool = False) -> None:
    print("\n" + "=" * 70)
    print("PART 1 — All three signal types delivered to one agent (O1, seed=42)")
    print("=" * 70)

    zone_csv = Path(__file__).parent / "data" / "zone_temps_sample.csv"
    if zone_csv.exists():
        zone_temp_fn = zone_temp_from_csv(zone_csv)
    else:
        zone_temp_fn = lambda ts: summer_day_temp(ts) - 12.0

    agent = (MockLLMAgent.from_stratum("O1", seed=seed) if mock
             else OccupantAgent.from_stratum("O1", seed=seed, llm_provider=provider))
    sim   = _build_sim(agent)

    print(f"\nPersona: {agent.persona.core_memory_text.splitlines()[0]}")
    print(f"  income_bracket={agent.persona.income_bracket}  "
          f"comfort_band={agent.persona.comfort_band_c:.1f}°C")

    signal_ts = _warmup(agent, sim, zone_temp_fn, warmup, seed=seed)

    for sig_type in ("A", "B", "C"):
        _deliver(agent, sim, zone_temp_fn, signal_ts, sig_type)
        signal_ts += timedelta(minutes=15)

    print(f"\n  Total memories after signals: {len(agent.memory.entries)}")


# ── Part 2: same signal → all four strata ────────────────────────────────────

def demo_stratum_comparison(provider: str, seed: int, warmup: int, mock: bool = False) -> None:
    print("\n" + "=" * 70)
    print("PART 2 — Same signal (Type B) delivered to all four strata")
    print("         Differences in response reflect demographic persona variation.")
    print("=" * 70)

    zone_csv = Path(__file__).parent / "data" / "zone_temps_sample.csv"
    if zone_csv.exists():
        zone_temp_fn = zone_temp_from_csv(zone_csv)
    else:
        zone_temp_fn = lambda ts: summer_day_temp(ts) - 12.0

    results: list[tuple[str, str, str]] = []

    for stratum in ("O1", "O2", "O3", "O4"):
        print(f"\n  ── {stratum}: {STRATA_LABELS[stratum]} ──")
        agent = (MockLLMAgent.from_stratum(stratum, seed=seed) if mock
                 else OccupantAgent.from_stratum(stratum, seed=seed, llm_provider=provider))
        sim   = _build_sim(agent)

        sig_ts = _warmup(agent, sim, zone_temp_fn, warmup, seed=seed)
        env    = sim.observe(sig_ts, zone_temp_fn(sig_ts))
        sig    = SIGNALS["B"]

        response = agent.receive_signal("B", sig["content"], env)
        icon = {"accepted": "✓", "rejected": "✗", "deferred": "~"}.get(response.response, "?")
        print(f"  {icon} {response.response.upper()}: {(response.reasoning or '')[:140]}")
        results.append((stratum, response.response, response.reasoning or ""))

    print("\n  Summary:")
    print(f"  {'Stratum':<6}  {'Response':<10}  Description")
    print(f"  {'─'*6}  {'─'*10}  {'─'*30}")
    for stratum, resp, _ in results:
        print(f"  {stratum:<6}  {resp:<10}  {STRATA_LABELS[stratum]}")


# ── Part 3: extra_context kwarg ───────────────────────────────────────────────

def demo_extra_context(provider: str, seed: int, warmup: int, mock: bool = False) -> None:
    print("\n" + "=" * 70)
    print("PART 3 — extra_context: situational notes change agent reasoning")
    print("         Same agent (O2), same Type A signal, different context.")
    print("=" * 70)

    zone_csv = Path(__file__).parent / "data" / "zone_temps_sample.csv"
    if zone_csv.exists():
        zone_temp_fn = zone_temp_from_csv(zone_csv)
    else:
        zone_temp_fn = lambda ts: summer_day_temp(ts) - 12.0

    contexts = [
        None,
        "The occupant is currently hosting a dinner party with guests.",
        "The occupant has a newborn infant at home who needs stable temperature.",
    ]

    for ctx in contexts:
        agent  = (MockLLMAgent.from_stratum("O2", seed=seed) if mock
                  else OccupantAgent.from_stratum("O2", seed=seed, llm_provider=provider))
        sim    = _build_sim(agent)
        sig_ts = _warmup(agent, sim, zone_temp_fn, warmup, seed=seed)
        label  = f'"{ctx}"' if ctx else "(no extra context)"
        print(f"\n  Context: {label}")
        _deliver(agent, sim, zone_temp_fn, sig_ts, "A", extra_context=ctx)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BuildOcc demand response signal demo")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai", "google", "ollama"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=6,
                        help="Warmup steps before each signal (default 6 = 90 min)")
    parser.add_argument("--part", type=int, choices=[1, 2, 3],
                        help="Run only one part (default: all three)")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockLLMAgent — no API key needed (for CI / offline testing)")
    args = parser.parse_args()

    if args.part in (None, 1):
        demo_signal_types(args.provider, args.seed, args.warmup, mock=args.mock)
    if args.part in (None, 2):
        demo_stratum_comparison(args.provider, args.seed, args.warmup, mock=args.mock)
    if args.part in (None, 3):
        demo_extra_context(args.provider, args.seed, args.warmup, mock=args.mock)

    print("\n" + "=" * 70)
    print("Done. See examples/simulation_loop.py for the full step() loop.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
