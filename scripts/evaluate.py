"""
Evaluation harness for OccupantAgent — Tier 1: Activity Schedule KL Validation.

Runs a single agent configuration for N days and computes behavioral metrics
(KL-divergence, KS-test) against ATUS reference distributions.

To reproduce Table 2 in the manuscript (180 days × 96 timesteps per stratum):

    python scripts/evaluate.py --stratum O1 --seed 42 --days 180 --scheduler atus
    python scripts/evaluate.py --stratum O1 --seed 42 --days 180 --scheduler fixed

    # All strata (atus + fixed ablation)
    for s in O1 O2 O3 O4; do
        python scripts/evaluate.py --stratum $s --seed 42 --days 180 --scheduler atus
        python scripts/evaluate.py --stratum $s --seed 42 --days 180 --scheduler fixed
    done

    # Skip ATUS KL/KS (no ATUS data); overwrite an existing result file
    python scripts/evaluate.py --stratum O1 --days 30 --no-atus-ref --force

Output
──────
Writes to scripts/experiments/outputs/eval_{stratum}_{scheduler}_seed{seed}.json:
    {
      "run_id": "O1_atus_seed42",
      "n_steps": 672,
      "metrics": {
        "aggregate_kl_divergence": 0.123,
        "ks_statistic": 0.045,
        "action_distribution": { "do_nothing": 0.87, "toggle_device": 0.10, ... }
      },
      "per_hour_kl": { "0": 0.05, "1": 0.03, ... }
    }
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.analysis.metrics import compute_kl, compute_ks, compute_kl_by_hour
from occupant_agent.analysis.simulation_log import SimulationLog
from occupant_agent.environment.simulation import SimulationEnvironment, summer_day_temp, peak_tou_rate, persona_devices
from occupant_agent.environment.state import AgentAction, RoomState
from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
from occupant_agent.grounding.scheduler import ActivityScheduler, _get_category


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OccupantAgent evaluation harness")
    p.add_argument("--stratum", default="O1", choices=["O1", "O2", "O3", "O4"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=180,
                   help="Simulation days (default 180 matches Table 2 in the manuscript)")
    p.add_argument(
        "--scheduler", default="atus", choices=["atus", "fixed"],
        help="Activity scheduler: 'atus' (ATUS-grounded) or 'fixed' (ablation baseline)"
    )
    p.add_argument(
        "--llm-provider", default="anthropic",
        choices=["anthropic", "openai", "google", "ollama"]
    )
    p.add_argument(
        "--start-date", default="2025-08-11",
        help="Simulation start date ISO 8601 (use a Monday; default 2025-08-11 matches manuscript)"
    )
    p.add_argument(
        "--output-dir",
        default=str(_ROOT / "scripts" / "experiments" / "outputs"),
    )
    p.add_argument(
        "--step-minutes", type=int, default=15,
        help="Timestep size in minutes (must match agent design; default 15)"
    )
    p.add_argument(
        "--no-atus-ref", action="store_true",
        help="Skip KL/KS computation against ATUS reference (useful when data is unavailable)"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output file (default: abort if file already exists)"
    )
    return p


def _load_atus_reference(
    stratum: str,
    data_dir: Path,
) -> dict[int, dict[str, float]] | None:
    """
    Load hourly time-at-activity reference from ATUS outputs.
    Returns {hour: {category: weighted_pct}} or None if data unavailable.
    """
    try:
        import pandas as pd
        path = data_dir / "time_at_activity.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        sub = df[df["stratum"] == stratum]
        if sub.empty:
            return None
        if "day_type" in sub.columns:
            sub = sub[sub["day_type"] == "weekday"]
        if sub.empty:
            return None
        ref: dict[int, dict[str, float]] = {}
        for _, row in sub.iterrows():
            h = int(row["hour"])
            cat = str(row["category"])
            pct = float(row["weighted_pct"])
            ref.setdefault(h, {})[cat] = pct
        return ref
    except Exception:
        return None


_ZONE_CSV = _ROOT / "examples" / "data" / "zone_temps_sample.csv"


def _run_simulation(
    stratum: str,
    seed: int,
    days: int,
    scheduler_name: str,
    llm_provider: str,
    start_date: datetime,
    step_minutes: int,
) -> SimulationLog:
    """Run the simulation and return a populated SimulationLog."""
    from occupant_agent.environment.simulation import zone_temp_from_csv

    if _ZONE_CSV.exists():
        zone_temp_fn = zone_temp_from_csv(_ZONE_CSV)
    else:
        print(f"[evaluate] WARNING: {_ZONE_CSV} not found — falling back to summer_day_temp",
              file=sys.stderr)
        zone_temp_fn = lambda ts: summer_day_temp(ts) - 12.0

    # Build agent. atus_sched below drives code sampling; no internal scheduler needed.
    agent = OccupantAgent.from_stratum(
        stratum,
        seed=seed,
        llm_provider=llm_provider,
    )

    # ATUS scheduler for code sampling (separate from agent's built-in scheduler).
    # When agent.scheduler is set, step() auto-samples; we still need the code for logging.
    atus_sched: ActivityScheduler | None = None
    if scheduler_name == "atus":
        atus_sched = ActivityScheduler(stratum=stratum, seed=seed)

    # Set up simulation environment using stratum-specific device and room layout.
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

    log = SimulationLog(stratum=stratum, seed=seed, scheduler=scheduler_name)
    timestep = start_date
    step_delta = timedelta(minutes=step_minutes)
    total_steps = days * 24 * (60 // step_minutes)

    # FixedScheduleScheduler only needed when atus_sched is None; create once.
    _fsched = FixedScheduleScheduler() if atus_sched is None else None

    # Sample wfh_today once per simulated day; re-sample when the date rolls over.
    import random as _random
    _rng = _random.Random(seed)
    wfh_today = agent.persona.sample_wfh_today(_rng)
    current_date = timestep.date()

    for _ in range(total_steps):
        if timestep.date() != current_date:
            current_date = timestep.date()
            wfh_today = agent.persona.sample_wfh_today(_rng)

        env = sim.observe(timestep, zone_temp_fn(timestep))

        # Determine ATUS code for this step.
        if atus_sched is not None:
            atus_code = atus_sched.sample(timestep)
        else:
            atus_code = _fsched.sample(timestep)  # type: ignore[union-attr]

        try:
            action = agent.step(env, atus_code=atus_code, wfh_today=wfh_today)
        except Exception as exc:
            print(f"  [WARN] step() raised {type(exc).__name__}: {exc} at {timestep}", file=sys.stderr)
            action = AgentAction(action_type="do_nothing", reasoning="Error fallback")

        category = _get_category(atus_code) if atus_code else None

        log.record(
            timestep=timestep,
            action=action,
            env=env,
            atus_code=atus_code,
            activity_category=category,
            memory_count=len(agent.memory.entries),
            wfh_today=wfh_today,
        )

        sim.apply(action, timestep)
        timestep += step_delta

    return log


def _compute_metrics(
    log: SimulationLog,
    atus_ref: dict[int, dict[str, float]] | None,
) -> tuple[dict, dict]:
    """Compute all metrics from the simulation log."""
    records = log.to_dicts()

    # Action distribution.
    action_counts: dict[str, int] = {}
    for r in records:
        at = r["action_type"]
        action_counts[at] = action_counts.get(at, 0) + 1
    total = len(records) or 1
    action_dist = {k: v / total for k, v in action_counts.items()}

    # Per-hour activity category counts — weekday only to match the weekday-only
    # ATUS reference loaded by _load_atus_reference().  Including weekend records
    # in a weekday-filtered reference comparison inflates KL/KS for employed strata
    # (O1, O3) where work rates differ sharply across day types.
    hour_cat_counts: dict[int, dict[str, float]] = {}
    for r in records:
        ts = datetime.fromisoformat(r["timestep"])
        if ts.weekday() >= 5:  # skip Saturday (5) and Sunday (6)
            continue
        h = ts.hour
        cat = r.get("activity_category") or "other"
        hour_cat_counts.setdefault(h, {}).setdefault(cat, 0)
        hour_cat_counts[h][cat] += 1

    metrics: dict = {"action_distribution": action_dist}

    if atus_ref is not None:
        # Overall KL: aggregate all hours.
        categories = ["sleeping", "work", "food_prep", "laundry", "tv", "eating", "exercise", "other"]
        sim_total = {c: 0.0 for c in categories}
        ref_total = {c: 0.0 for c in categories}
        for h in range(24):
            for c in categories:
                sim_total[c] += hour_cat_counts.get(h, {}).get(c, 0.0)
                ref_total[c] += atus_ref.get(h, {}).get(c, 0.0)

        p_vec = [sim_total[c] for c in categories]
        q_vec = [ref_total[c] for c in categories]
        try:
            metrics["aggregate_kl_divergence"] = compute_kl(p_vec, q_vec)
            metrics["ks_statistic"] = compute_ks(p_vec, q_vec)
        except ValueError as e:
            metrics["aggregate_kl_divergence"] = None
            metrics["ks_statistic"] = None
            metrics["metric_error"] = str(e)

        # Per-hour KL.
        per_hour_kl = compute_kl_by_hour(hour_cat_counts, atus_ref)
        valid_kl = [v for v in per_hour_kl.values() if not math.isnan(v)]
        metrics["mean_per_hour_kl"] = sum(valid_kl) / len(valid_kl) if valid_kl else None
    else:
        metrics["aggregate_kl_divergence"] = None
        metrics["ks_statistic"] = None
        per_hour_kl = {}

    return metrics, {str(h): v for h, v in (per_hour_kl or {}).items()}


def main() -> None:
    args = _build_arg_parser().parse_args()

    start_date = datetime.fromisoformat(args.start_date)
    print(f"[evaluate] stratum={args.stratum} scheduler={args.scheduler} "
          f"seed={args.seed} days={args.days} provider={args.llm_provider}")

    # Load ATUS reference if available.
    data_dir = _ROOT / "occupant_agent" / "data"
    atus_ref = None if args.no_atus_ref else _load_atus_reference(args.stratum, data_dir)
    if atus_ref is None and not args.no_atus_ref:
        print("[evaluate] ATUS reference not found — KL/KS will be skipped.", file=sys.stderr)

    log = _run_simulation(
        stratum=args.stratum,
        seed=args.seed,
        days=args.days,
        scheduler_name=args.scheduler,
        llm_provider=args.llm_provider,
        start_date=start_date,
        step_minutes=args.step_minutes,
    )
    print(f"[evaluate] Completed {len(log)} steps.")

    metrics, per_hour_kl = _compute_metrics(log, atus_ref)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")

    # Write output.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.stratum}_{args.scheduler}_seed{args.seed}"
    output_path = output_dir / f"eval_{run_id}.json"

    if output_path.exists() and not args.force:
        print(
            f"[evaluate] ERROR: {output_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = {
        "run_id": run_id,
        "stratum": args.stratum,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "days": args.days,
        "n_steps": len(log),
        "metrics": metrics,
        "per_hour_kl": per_hour_kl,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[evaluate] Results written to {output_path}")

    # Also write the full trace.
    trace_path = output_dir / f"trace_{run_id}.json"
    log.to_json(trace_path)
    print(f"[evaluate] Trace written to {trace_path}")


if __name__ == "__main__":
    main()
