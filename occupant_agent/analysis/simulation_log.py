"""
SimulationLog — accumulates per-timestep records during a simulation run.

Usage
─────
    log = SimulationLog(stratum="O1", seed=42, scheduler="atus")
    for ts in timesteps:
        env = sim.observe(ts, zone_temp)
        action = agent.step(env, atus_code=scheduler.sample(ts))
        log.record(ts, atus_code=atus_code, action=action, env=env)

    # Export for analysis
    df = log.to_dataframe()
    log.to_csv("run_P1_seed42.csv")
    log.to_json("run_P1_seed42.json")

The log is self-contained: it records stratum, seed, and scheduler so that
multiple runs can be aggregated without ambiguity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from occupant_agent.environment.state import AgentAction, EnvironmentState


@dataclass
class StepRecord:
    """A single recorded timestep."""

    timestep: str              # ISO 8601
    atus_code: str | None
    activity_category: str | None
    occupancy: str | None
    zone_temp_c: float
    outdoor_temp_c: float
    tou_rate: float
    thermostat_setpoint_c: float | None
    action_type: str
    target_id: str | None
    value: Any
    reasoning: str | None
    memory_count: int
    wfh_today: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SimulationLog:
    """
    Accumulates per-timestep records during a simulation run for post-hoc analysis.

    Thread-unsafe — designed for single-process simulation loops.
    """

    def __init__(
        self,
        stratum: str,
        seed: int | None = None,
        scheduler: str = "atus",
        run_id: str | None = None,
    ) -> None:
        self.stratum = stratum
        self.seed = seed
        self.scheduler_name = scheduler
        self.run_id = run_id or f"{stratum}_seed{seed}"
        self._records: list[StepRecord] = []

    def record(
        self,
        timestep: datetime,
        action: AgentAction,
        env: EnvironmentState,
        atus_code: str | None = None,
        activity_category: str | None = None,
        occupancy: str | None = None,
        memory_count: int = 0,
        wfh_today: bool | None = None,
    ) -> None:
        """
        Append one timestep record.

        Args:
            timestep:          Simulation datetime.
            action:            AgentAction returned by OccupantAgent.step().
            env:               EnvironmentState used for that step.
            atus_code:         ATUS code sampled for this step (None if not provided).
            activity_category: Resolved category name (sleeping, work, ...) or None.
            occupancy:         "home" | "away" | "ambiguous" resolved from ATUS code.
            memory_count:      len(agent.memory.entries) after this step.
            wfh_today:         WFH flag passed to agent.step() this day.
        """
        self._records.append(StepRecord(
            timestep=timestep.isoformat(),
            atus_code=atus_code,
            activity_category=activity_category,
            occupancy=occupancy,
            zone_temp_c=env.zone_temp_c,
            outdoor_temp_c=env.outdoor_temp_c,
            tou_rate=env.tou_rate,
            thermostat_setpoint_c=env.thermostat_setpoint_c,
            action_type=action.action_type,
            target_id=action.target_id,
            value=action.value,
            reasoning=action.reasoning,
            memory_count=memory_count,
            wfh_today=wfh_today,
        ))

    def record_signal(
        self,
        timestep: datetime,
        signal_type: str,
        content: str,
        response: str,
        reasoning: str | None = None,
    ) -> None:
        """Record a signal event as a special action row."""
        self._records.append(StepRecord(
            timestep=timestep.isoformat(),
            atus_code=None,
            activity_category=None,
            occupancy=None,
            zone_temp_c=float("nan"),
            outdoor_temp_c=float("nan"),
            tou_rate=float("nan"),
            thermostat_setpoint_c=None,
            action_type=f"signal_{signal_type}_{response}",
            target_id=None,
            value=content,
            reasoning=reasoning,
            memory_count=0,
            wfh_today=None,
        ))

    def to_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self._records]

    def to_json(self, path: str | Path) -> None:
        """Write the full log (with metadata) to a JSON file."""
        payload = {
            "run_id": self.run_id,
            "stratum": self.stratum,
            "seed": self.seed,
            "scheduler": self.scheduler_name,
            "n_steps": len(self._records),
            "records": self.to_dicts(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def to_csv(self, path: str | Path) -> None:
        """Write records to CSV (requires pandas)."""
        import pandas as pd
        df = pd.DataFrame(self.to_dicts())
        df.to_csv(path, index=False)

    def to_dataframe(self):
        """Return records as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.to_dicts())

    @property
    def records(self) -> list[StepRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"SimulationLog(run_id={self.run_id!r}, n_steps={len(self._records)})"
