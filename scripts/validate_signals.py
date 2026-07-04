"""
Validation Study — Analysis 2: Signal Compliance by Type and Stratum.

Delivers standardized Type A/B/C demand response signals to all four strata
after a brief warmup period. Records accept/reject/defer responses and
aggregates them into a compliance matrix suitable for Table 4 in the
SoftwareX manuscript.

Signal typology (Gyamfi & Krumdieck 2011; Albadi & El-Saadany 2008):
    A — Direct command  ("Turn off your HVAC for 30 minutes.")
    B — Educational     ("Your HVAC costs 3× more during peak hours.")
    C — Social norm     ("75% of similar households have reduced HVAC use.")

Usage
─────
    python scripts/validate_signals.py --provider anthropic --seeds 5

    # Dry run (no API key)
    python scripts/validate_signals.py --mock --seeds 2

Outputs (in scripts/experiments/outputs/)
──────────────────────────────────────────
    signal_responses.csv  — per-trial: stratum, seed, signal_type, response, reasoning
    signal_summary.csv    — 4×3 table: stratum × signal_type → accept/reject/defer %
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

STRATA = ["O1", "O2", "O3", "O4"]
ZONE_CSV = _ROOT / "examples" / "data" / "zone_temps_sample.csv"
WARMUP_STEPS = 10
STEP_MINUTES = 15

# Standardized signal content — identical across all strata and seeds
SIGNALS: dict[str, str] = {
    "A": (
        "Turn off your HVAC for the next 30 minutes to support grid stability "
        "during today's peak demand event."
    ),
    "B": (
        "Your HVAC is currently running during peak hours at $0.22/kWh — "
        "approximately 3× the off-peak rate of $0.08/kWh. "
        "Raising your thermostat setpoint by 2°C saves an estimated $0.35 today."
    ),
    "C": (
        "75% of similar households in your area have reduced their HVAC use "
        "during today's peak demand event."
    ),
}

# Warmup starts at 4 pm so the agent has peak-rate context before signals arrive
WARMUP_START = datetime(2025, 8, 11, 16, 0)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Signal compliance validation")
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds per stratum")
    p.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "openai", "google", "ollama"],
    )
    p.add_argument("--mock", action="store_true", help="Use MockLLMAgent (no API key needed)")
    p.add_argument(
        "--output-dir",
        default=str(_ROOT / "scripts" / "experiments" / "outputs"),
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files without prompting")
    return p


def _warmup(agent, sim, zone_temp_fn, seed: int) -> datetime:
    """Run WARMUP_STEPS steps to build agent memory; return the next timestep."""
    from occupant_agent.environment.state import AgentAction
    from occupant_agent.grounding.scheduler import ActivityScheduler

    scheduler = ActivityScheduler(stratum=agent.persona.stratum, seed=seed)
    rng = random.Random(seed)
    wfh_today = agent.persona.sample_wfh_today(rng)

    timestep = WARMUP_START
    step_delta = timedelta(minutes=STEP_MINUTES)

    for _ in range(WARMUP_STEPS):
        env = sim.observe(timestep, zone_temp_fn(timestep))
        atus_code = scheduler.sample(timestep)
        try:
            action = agent.step(env, atus_code=atus_code, wfh_today=wfh_today)
        except Exception:
            action = AgentAction(action_type="do_nothing", reasoning="warmup error")
        sim.apply(action, timestep)
        timestep += step_delta

    return timestep


def _build_sim(persona, rooms=None):
    from occupant_agent.environment.simulation import (
        SimulationEnvironment, peak_tou_rate, summer_day_temp, persona_devices,
    )
    from occupant_agent.environment.state import RoomState
    default_rooms = [
        RoomState(room_id="living_room", occupied=True),
        RoomState(room_id="bedroom",     occupied=False),
    ]
    return SimulationEnvironment(
        initial_devices=persona_devices(persona),
        initial_rooms=rooms if rooms is not None else default_rooms,
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )


def _run_one_seed(
    stratum: str,
    seed: int,
    provider: str,
    mock: bool,
    zone_temp_fn,
) -> list[dict]:
    """Warm up a fresh agent per signal type to avoid ordering confounds."""
    from occupant_agent.agent.occupant import OccupantAgent
    from occupant_agent.testing import MockLLMAgent

    cls = MockLLMAgent if mock else OccupantAgent
    rows: list[dict] = []

    for sig_type, content in SIGNALS.items():
        # Independent agent + sim for each signal so prior signal responses
        # do not appear in memory when the next signal is evaluated.
        agent = cls.from_stratum(stratum, seed=seed, llm_provider=provider)
        from occupant_agent.environment.state import RoomState
        persona_rooms = [
            RoomState(room_id=r, occupied=(r == "living_room"))
            for r in agent.persona.room_ids
        ]
        sim   = _build_sim(agent.persona, rooms=persona_rooms)

        signal_ts = _warmup(agent, sim, zone_temp_fn, seed)
        env = sim.observe(signal_ts, zone_temp_fn(signal_ts))

        try:
            resp = agent.receive_signal(
                signal_type=sig_type,
                content=content,
                env=env,
            )
            response  = resp.response
            reasoning = (resp.reasoning or "").replace("\n", " ")[:300]
        except Exception as exc:
            response  = "error"
            reasoning = str(exc)[:200]

        rows.append({
            "stratum":     stratum,
            "seed":        seed,
            "signal_type": sig_type,
            "response":    response,
            "reasoning":   reasoning,
        })

    return rows


_RESPONSE_MAP = {
    "accepted": "accept", "accept": "accept",
    "rejected": "reject", "reject": "reject",
    "deferred": "defer",  "defer":  "defer",
}


def _normalize_response(raw: str) -> str:
    return _RESPONSE_MAP.get(raw.lower().strip(), "error")


def _summarize(all_rows: list[dict]) -> list[dict]:
    """Compute accept/reject/defer rates per (stratum, signal_type)."""
    counts: dict[tuple, dict] = defaultdict(
        lambda: {"accept": 0, "reject": 0, "defer": 0, "error": 0, "total": 0}
    )

    for row in all_rows:
        key  = (row["stratum"], row["signal_type"])
        resp = _normalize_response(row["response"])
        if resp == "error":
            print(f"  [WARN] Unrecognised response '{row['response']}' for "
                  f"{row['stratum']} signal {row['signal_type']} — counted as error",
                  file=sys.stderr)
        counts[key][resp]    += 1
        counts[key]["total"] += 1

    summary: list[dict] = []
    for stratum in STRATA:
        for sig_type in ("A", "B", "C"):
            c     = counts[(stratum, sig_type)]
            valid = c["total"] - c["error"]
            if valid == 0:
                print(
                    f"  [WARN] All trials for {stratum} Signal {sig_type} errored — rates undefined",
                    file=sys.stderr,
                )
            valid = max(valid, 1)
            summary.append({
                "stratum":    stratum,
                "signal_type": sig_type,
                "accept_pct": round(c["accept"] / valid * 100, 1),
                "reject_pct": round(c["reject"] / valid * 100, 1),
                "defer_pct":  round(c["defer"]  / valid * 100, 1),
                "n_trials":   c["total"],
            })
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guard against silent overwrites
    existing = [f for f in ["signal_responses.csv", "signal_summary.csv"]
                if (output_dir / f).exists()]
    if existing and not args.force:
        print(f"[validate_signals] ERROR: output file(s) already exist: {existing}\n"
              f"  Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    from occupant_agent.environment.simulation import summer_day_temp, zone_temp_from_csv
    if ZONE_CSV.exists():
        zone_temp_fn = zone_temp_from_csv(ZONE_CSV)
        print(f"[validate_signals] Using EnergyPlus zone temps from {ZONE_CSV.name}")
    else:
        print(f"[validate_signals] WARNING: {ZONE_CSV} not found — falling back to summer_day_temp",
              file=sys.stderr)
        zone_temp_fn = lambda ts: summer_day_temp(ts) - 12.0

    seeds = list(range(args.seeds))
    total_agents = len(STRATA) * len(seeds)
    total_llm = total_agents * len(SIGNALS) * (WARMUP_STEPS + 1)
    print(f"[validate_signals] {len(STRATA)} strata × {len(seeds)} seeds = "
          f"{total_agents} agent runs, "
          f"~{total_llm} LLM calls ({len(SIGNALS)} signals × ({WARMUP_STEPS} warmup + 1) each)")

    all_rows: list[dict] = []
    run_num = 0
    for stratum in STRATA:
        for seed in seeds:
            run_num += 1
            print(f"[validate_signals] [{run_num}/{total_agents}] {stratum} seed={seed} ...",
                  flush=True)
            rows = _run_one_seed(
                stratum=stratum,
                seed=seed,
                provider=args.provider,
                mock=args.mock,
                zone_temp_fn=zone_temp_fn,
            )
            for row in rows:
                print(f"  Signal {row['signal_type']}: {row['response']}")
            all_rows.extend(rows)

    # Write signal_responses.csv
    responses_path = output_dir / "signal_responses.csv"
    if all_rows:
        fieldnames = ["stratum", "seed", "signal_type", "response", "reasoning"]
        with open(responses_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[validate_signals] Wrote {len(all_rows)} rows → {responses_path}")

    # Write signal_summary.csv
    summary = _summarize(all_rows)
    summary_path = output_dir / "signal_summary.csv"
    if summary:
        fieldnames = list(summary[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary)
        print(f"[validate_signals] Summary → {summary_path}")
        print()
        print("Signal Compliance Matrix (accept %):")
        print(f"  {'Stratum':<8} {'Type A':>8} {'Type B':>8} {'Type C':>8}")
        for stratum in STRATA:
            row_a = next(r for r in summary if r["stratum"] == stratum and r["signal_type"] == "A")
            row_b = next(r for r in summary if r["stratum"] == stratum and r["signal_type"] == "B")
            row_c = next(r for r in summary if r["stratum"] == stratum and r["signal_type"] == "C")
            print(f"  {stratum:<8} {row_a['accept_pct']:>7.0f}% {row_b['accept_pct']:>7.0f}% "
                  f"{row_c['accept_pct']:>7.0f}%")

    # Write run metadata sidecar
    import occupant_agent
    metadata = {
        "script":           "scripts/validate_signals.py",
        "timestamp_utc":    datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "provider":         args.provider,
        "mock":             args.mock,
        "seeds":            args.seeds,
        "warmup_steps":     WARMUP_STEPS,
        "package_version": getattr(occupant_agent, "__version__", "unknown"),
        "python":           platform.python_version(),
        "zone_csv":         str(ZONE_CSV) if ZONE_CSV.exists() else "fallback:summer_day_temp",
    }
    meta_path = output_dir / "signals_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[validate_signals] Metadata → {meta_path}")


if __name__ == "__main__":
    main()
