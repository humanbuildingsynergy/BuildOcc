"""
FixedScheduleScheduler — rule-based baseline scheduler for ablation studies.

This scheduler does NOT use ATUS data. It assigns activity codes based on a
simplified hour-of-day rule: sleep at night, work during business hours, and
a generic "other" code otherwise.

Purpose
───────
Ablation baseline: comparing OccupantAgent with ATUS grounding (ActivityScheduler)
vs. without ATUS grounding (FixedScheduleScheduler) isolates the contribution of
the empirical time-use grounding to behavioral realism.

Reference: see Phase 2 / Paper 1 ablation experiment design.

Usage
─────
    from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
    agent = OccupantAgent.from_stratum("O1", seed=42, scheduler=FixedScheduleScheduler())
"""

from __future__ import annotations

from datetime import datetime

from occupant_agent.core.base_scheduler import BaseScheduler
from occupant_agent.core.registry import register_scheduler

# ── Hourly activity map (rule-based, no survey data) ─────────────────────────
# Maps hour → (category, canonical ATUS tier-3 code)
# Rules: sleep 22:00-07:00, work 09:00-17:00 Mon-Fri, eating at meal times,
#        everything else → other.

_HOUR_RULES: dict[int, tuple[str, str]] = {
    22: ("sleeping", "010101"),
    23: ("sleeping", "010101"),
    0:  ("sleeping", "010101"),
    1:  ("sleeping", "010101"),
    2:  ("sleeping", "010101"),
    3:  ("sleeping", "010101"),
    4:  ("sleeping", "010101"),
    5:  ("sleeping", "010101"),
    6:  ("sleeping", "010101"),
    7:  ("other",    "010201"),  # grooming / waking up
    8:  ("eating",   "110101"),  # breakfast
    9:  ("work",     "050101"),
    10: ("work",     "050101"),
    11: ("work",     "050101"),
    12: ("eating",   "110101"),  # lunch
    13: ("work",     "050101"),
    14: ("work",     "050101"),
    15: ("work",     "050101"),
    16: ("work",     "050101"),
    17: ("other",    "010201"),  # commute / transition
    18: ("eating",   "110101"),  # dinner
    19: ("tv",       "120301"),
    20: ("tv",       "120301"),
    21: ("tv",       "120301"),
}

_WEEKEND_HOUR_RULES: dict[int, tuple[str, str]] = {
    22: ("sleeping", "010101"),
    23: ("sleeping", "010101"),
    0:  ("sleeping", "010101"),
    1:  ("sleeping", "010101"),
    2:  ("sleeping", "010101"),
    3:  ("sleeping", "010101"),
    4:  ("sleeping", "010101"),
    5:  ("sleeping", "010101"),
    6:  ("sleeping", "010101"),
    7:  ("sleeping", "010101"),
    8:  ("eating",   "110101"),  # late breakfast
    9:  ("other",    "010201"),
    10: ("food_prep","020101"),
    11: ("other",    "010201"),
    12: ("eating",   "110101"),
    13: ("other",    "010201"),
    14: ("exercise", "130101"),
    15: ("other",    "010201"),
    16: ("tv",       "120301"),
    17: ("food_prep","020101"),
    18: ("eating",   "110101"),
    19: ("tv",       "120301"),
    20: ("tv",       "120301"),
    21: ("other",    "010201"),
}


@register_scheduler("fixed")
class FixedScheduleScheduler(BaseScheduler):
    """
    Rule-based activity scheduler — ablation baseline (no ATUS data).

    Uses a deterministic hour-of-day → activity rule with weekday/weekend split.
    No random sampling; stratum and seed are ignored.

    Instantiate with no arguments:
        scheduler = FixedScheduleScheduler()
    """

    def __init__(self, **kwargs) -> None:
        pass

    def sample(self, timestep: datetime) -> str:
        """Return the canonical ATUS code for this hour (rule-based, no sampling)."""
        rules = _WEEKEND_HOUR_RULES if timestep.weekday() >= 5 else _HOUR_RULES
        _, code = rules.get(timestep.hour, ("other", "010201"))
        return code

    def category_weights(
        self,
        hour: int,
        timestep: datetime | None = None,
    ) -> dict[str, float]:
        """
        Return a deterministic weight distribution (1.0 for the assigned category).
        Used by the evaluation harness for KL-divergence comparison.
        """
        is_weekend = timestep.weekday() >= 5 if timestep else False
        rules = _WEEKEND_HOUR_RULES if is_weekend else _HOUR_RULES
        category, _ = rules.get(hour, ("other", "010201"))

        from occupant_agent.grounding.scheduler import _CATEGORY_NAMES
        return {cat: (1.0 if cat == category else 0.0) for cat in _CATEGORY_NAMES}
