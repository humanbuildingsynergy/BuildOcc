"""
OccupantAgent analysis module.

Provides tools for logging simulation traces and computing validation metrics.

Modules
───────
simulation_log  — SimulationLog: accumulates per-step records during a run
metrics         — compute_kl, compute_ks, compute_cvrmse, compute_mbe
"""

from occupant_agent.analysis.metrics import (
    compute_cvrmse,
    compute_kl,
    compute_kl_by_hour,
    compute_ks,
    compute_mbe,
)
from occupant_agent.analysis.simulation_log import SimulationLog

__all__ = [
    "SimulationLog",
    "compute_kl",
    "compute_kl_by_hour",
    "compute_ks",
    "compute_cvrmse",
    "compute_mbe",
]
