"""
BaseScheduler — abstract contract for activity grounding sources.

Subclass this to ground the agent in a different data source:
  - Homer dataset (replicating AgentSense approach — for ablation §4.5)
  - MTUS (Multinational Time Use Study — for cross-country comparison)
  - Synthetic schedules (for controlled experiments)
  - Fixed schedules (for unconstrained-LLM baseline — see FixedScheduleScheduler)

The default implementation (ActivityScheduler) uses ATUS 2022+2023 microdata
with population weights and weekday/weekend split.

Example
───────
    from occupant_agent.core import BaseScheduler, register_scheduler
    from datetime import datetime

    @register_scheduler("homer")
    class HomerScheduler(BaseScheduler):
        \"\"\"Activity grounding from Homer dataset (21 participants).\"\"\"

        def sample(self, timestep: datetime) -> str:
            # Sample from Homer diary records
            ...

        def category_weights(self, hour: int, timestep=None) -> dict[str, float]:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class BaseScheduler(ABC):
    """
    Abstract base class for activity grounding sources.

    Subclass this — do NOT modify scheduler.py — to use a different
    empirical dataset, a synthetic schedule, or a fixed baseline.

    Both methods are required; OccupantAgent.step() calls sample() when
    no atus_code is provided by the caller.
    """

    @abstractmethod
    def sample(self, timestep: datetime) -> str:
        """
        Sample an ATUS activity code for the given simulation timestep.

        Args:
            timestep: Current simulation datetime.

        Returns:
            6-digit ATUS tier-3 code string (e.g., '010101', '050101').
            Must be a valid ATUS code; activity_code_map.lookup() will be
            called on it to resolve occupancy and device context.
        """

    @abstractmethod
    def category_weights(
        self,
        hour: int,
        timestep: datetime | None = None,
    ) -> dict[str, float]:
        """
        Return the probability distribution over activity categories at hour H.

        Args:
            hour:     Integer 0–23.
            timestep: Optional; if provided, weekday/weekend is auto-detected.

        Returns:
            Dict mapping category name → probability (must sum to ~1.0).
            Category names: sleeping, work, food_prep, laundry, tv,
                            eating, exercise, other.
        """
