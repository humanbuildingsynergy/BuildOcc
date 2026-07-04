"""
Validation Study — Analysis 1: Persona Behavioral Differentiation.

Runs all four ATUS strata (O1–O4) with N seeds over M weekdays using the
EnergyPlus zone temperature CSV. All strata see identical environment
conditions; only the persona varies. Outputs per-timestep action logs and
an aggregated summary suitable for Table 3 in the SoftwareX manuscript.

Usage
─────
    python scripts/validate_strata.py --provider anthropic --seeds 3 --days 1

    # Dry run with MockLLMAgent (no API key needed)
    python scripts/validate_strata.py --mock --seeds 1 --days 1

Outputs (in scripts/experiments/outputs/)
──────────────────────────────────────────
    strata_actions.csv  — per-timestep: stratum, seed, timestep, action_type,
                          target_id, value, atus_code, zone_temp_c,
                          outdoor_temp_c, tou_rate, thermostat_setpoint_c
    strata_summary.csv  — per-stratum: do_nothing_pct, adjust_thermostat_pct,
                          toggle_device_pct, move_room_pct, mean_peak_setpoint_c
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
STEP_MINUTES = 15
ZONE_CSV = _ROOT / "examples" / "data" / "zone_temps_sample.csv"

# TOU peak band for mean-setpoint-during-peak computation
PEAK_START = 16
PEAK_END = 21


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Persona behavioral differentiation validation")
    p.add_argument("--seeds", type=int, default=3, help="Number of random seeds per stratum")
    p.add_argument("--days", type=int, default=1,
                   help="Simulation days per seed (default 1 = one weekday, 96 steps, matches manuscript)")
    p.add_argument(
        "--start-date", default="2025-08-11",
        help="ISO 8601 start date (2025-08-11 Monday falls in the EnergyPlus CSV range)",
    )
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


def _run_one(
    stratum: str,
    seed: int,
    days: int,
    start_date: datetime,
    provider: str,
    mock: bool,
    zone_temp_fn,
) -> list[dict]:
    """Run one (stratum, seed) pair; return per-timestep row dicts."""
    from occupant_agent.agent.occupant import OccupantAgent
    from occupant_agent.environment.simulation import peak_tou_rate, summer_day_temp
    from occupant_agent.environment.state import AgentAction, RoomState
    from occupant_agent.environment.simulation import SimulationEnvironment
    from occupant_agent.grounding.scheduler import ActivityScheduler
    from occupant_agent.testing import MockLLMAgent

    cls = MockLLMAgent if mock else OccupantAgent
    agent = cls.from_stratum(stratum, seed=seed, llm_provider=provider)
    scheduler = ActivityScheduler(stratum=stratum, seed=seed)

    # Use stratum-specific room layout from persona; start in living_room
    start_room = "living_room"
    initial_rooms = [
        RoomState(room_id=r, occupied=(r == start_room))
        for r in agent.persona.room_ids
    ]

    # Device set adapts to each stratum's appliance ownership via persona_devices()
    from occupant_agent.environment.simulation import persona_devices
    sim = SimulationEnvironment(
        initial_devices=persona_devices(agent.persona),
        initial_rooms=initial_rooms,
        thermostat_setpoint=22.0,
        outdoor_temp_fn=summer_day_temp,
        tou_rate_fn=peak_tou_rate,
    )

    rows: list[dict] = []
    timestep = start_date
    step_delta = timedelta(minutes=STEP_MINUTES)
    total_steps = days * 24 * (60 // STEP_MINUTES)

    rng = random.Random(seed)
    wfh_today = agent.persona.sample_wfh_today(rng)
    current_date = timestep.date()

    for _ in range(total_steps):
        if timestep.date() != current_date:
            current_date = timestep.date()
            wfh_today = agent.persona.sample_wfh_today(rng)

        zone_temp_c = zone_temp_fn(timestep)
        env = sim.observe(timestep, zone_temp_c)
        atus_code = scheduler.sample(timestep)

        try:
            action = agent.step(env, atus_code=atus_code, wfh_today=wfh_today)
        except Exception as exc:
            print(f"  [WARN] {stratum} seed={seed}: step() error: {exc}", file=sys.stderr)
            action = AgentAction(action_type="do_nothing", reasoning="error fallback")

        rows.append({
            "stratum":              stratum,
            "seed":                 seed,
            "timestep":             timestep.isoformat(),
            "action_type":          action.action_type,
            "target_id":            action.target_id,
            "value":                action.value,
            "atus_code":            atus_code,
            "zone_temp_c":          env.zone_temp_c,
            "outdoor_temp_c":       env.outdoor_temp_c,
            "tou_rate":             env.tou_rate,
            "thermostat_setpoint_c": env.thermostat_setpoint_c,
        })

        sim.apply(action, timestep)
        timestep += step_delta

    return rows


def _summarize(all_rows: list[dict]) -> list[dict]:
    """Aggregate per-stratum action-type percentages and mean peak-hour setpoints."""
    action_types = ["do_nothing", "adjust_thermostat", "toggle_device", "move_room"]
    data: dict[str, dict] = defaultdict(lambda: {
        "counts": {at: 0 for at in action_types},
        "unknown": 0,
        "total":   0,
        "peak_setpoints": [],
    })

    for row in all_rows:
        s  = row["stratum"]
        at = row["action_type"]
        data[s]["total"] += 1
        if at in action_types:
            data[s]["counts"][at] += 1
        else:
            data[s]["unknown"] += 1
            print(f"  [WARN] Unknown action_type '{at}' for {s} — not counted in summary",
                  file=sys.stderr)

        ts = datetime.fromisoformat(row["timestep"])
        if PEAK_START <= ts.hour < PEAK_END and row["thermostat_setpoint_c"] is not None:
            data[s]["peak_setpoints"].append(float(row["thermostat_setpoint_c"]))

    summary = []
    for stratum in STRATA:
        d = data[stratum]
        total = d["total"] or 1
        peak_sp = d["peak_setpoints"]
        summary.append({
            "stratum":                stratum,
            "total_steps":            d["total"],
            "do_nothing_pct":         round(d["counts"]["do_nothing"]         / total * 100, 1),
            "adjust_thermostat_pct":  round(d["counts"]["adjust_thermostat"]  / total * 100, 1),
            "toggle_device_pct":      round(d["counts"]["toggle_device"]       / total * 100, 1),
            "move_room_pct":          round(d["counts"]["move_room"]           / total * 100, 1),
            "unknown_pct":            round(d["unknown"]                       / total * 100, 1),
            "mean_peak_setpoint_c":   round(sum(peak_sp) / len(peak_sp), 2) if peak_sp else None,
        })
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    start_date = datetime.fromisoformat(args.start_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guard against silent overwrites
    existing = [f for f in ["strata_actions.csv", "strata_summary.csv"]
                if (output_dir / f).exists()]
    if existing and not args.force:
        print(f"[validate_strata] ERROR: output file(s) already exist: {existing}\n"
              f"  Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    from occupant_agent.environment.simulation import summer_day_temp, zone_temp_from_csv
    if ZONE_CSV.exists():
        zone_temp_fn = zone_temp_from_csv(ZONE_CSV)
        print(f"[validate_strata] Using EnergyPlus zone temps from {ZONE_CSV.name}")
    else:
        print(f"[validate_strata] WARNING: {ZONE_CSV} not found — falling back to summer_day_temp",
              file=sys.stderr)
        zone_temp_fn = lambda ts: summer_day_temp(ts) - 12.0  # rough indoor estimate

    seeds = list(range(args.seeds))
    total_runs = len(STRATA) * len(seeds)
    steps_per_run = args.days * 24 * (60 // STEP_MINUTES)
    print(f"[validate_strata] {len(STRATA)} strata × {len(seeds)} seeds × "
          f"{args.days} day(s) × {steps_per_run} steps = "
          f"{total_runs * steps_per_run} LLM calls")

    all_rows: list[dict] = []
    run_num = 0
    for stratum in STRATA:
        for seed in seeds:
            run_num += 1
            print(f"[validate_strata] [{run_num}/{total_runs}] {stratum} seed={seed} ...",
                  flush=True)
            rows = _run_one(
                stratum=stratum,
                seed=seed,
                days=args.days,
                start_date=start_date,
                provider=args.provider,
                mock=args.mock,
                zone_temp_fn=zone_temp_fn,
            )
            all_rows.extend(rows)
            print(f"  → {len(rows)} steps logged")

    # Write strata_actions.csv
    actions_path = output_dir / "strata_actions.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(actions_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[validate_strata] Wrote {len(all_rows)} rows → {actions_path}")

    # Write strata_summary.csv
    summary = _summarize(all_rows)
    summary_path = output_dir / "strata_summary.csv"
    if summary:
        fieldnames = list(summary[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary)
        print(f"[validate_strata] Summary → {summary_path}")
        print()
        header = f"{'Stratum':<8} {'do_nothing':>12} {'thermostat':>12} {'toggle':>10} {'move_room':>11} {'peak_sp':>10}"
        print(header)
        print("-" * len(header))
        for row in summary:
            print(f"  {row['stratum']:<6} {row['do_nothing_pct']:>11.1f}% "
                  f"{row['adjust_thermostat_pct']:>11.1f}% "
                  f"{row['toggle_device_pct']:>9.1f}% "
                  f"{row['move_room_pct']:>10.1f}% "
                  f"{str(row['mean_peak_setpoint_c']) + '°C':>10}")

    # Write run metadata sidecar
    import occupant_agent
    metadata = {
        "script":         "scripts/validate_strata.py",
        "timestamp_utc":  datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "provider":       args.provider,
        "mock":           args.mock,
        "seeds":          args.seeds,
        "days":           args.days,
        "start_date":     args.start_date,
        "package_version": getattr(occupant_agent, "__version__", "unknown"),
        "python":         platform.python_version(),
        "zone_csv":       str(ZONE_CSV) if ZONE_CSV.exists() else "fallback:summer_day_temp",
    }
    meta_path = output_dir / "strata_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[validate_strata] Metadata → {meta_path}")


if __name__ == "__main__":
    main()
