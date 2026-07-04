"""
ATUS-grounded activity scheduler: samples 6-digit ATUS codes from empirical
time-at-activity distributions for a given demographic stratum.

This closes the loop between the ATUS analysis outputs (scripts/atus/outputs/)
and the simulation pipeline. Without this module, callers must inject ATUS codes
manually; with it, the simulation is self-contained and fully ATUS-driven.

How it works
─────────────
At each 15-minute timestep the scheduler:

  1. Determines the activity category using time_at_activity.csv:
       P(category | hour, stratum) ∝ weighted_pct(category, hour, stratum)

     weighted_pct is the population-weighted fraction of respondents who are
     IN that activity AT that clock hour (episode-overlap based, not start time).
     This is the correct distribution for "what is the person doing now?"

  2. Samples a specific tier-3 code from the pool of codes in that category,
     weighted by pct_diary_time from activity_frequency_{stratum}.csv.

Eight activity categories:
  sleeping  → codes starting with "0101"
  work      → codes starting with "05"
  food_prep → codes starting with "0201"
  laundry   → code "020202" exactly
  tv        → code "120301" exactly
  eating    → codes starting with "1101"
  exercise  → codes starting with "13"
  other     → everything else (computer leisure, socializing, grooming, travel…)

Data sources (run scripts/atus/analyze.py to regenerate):
  occupant_agent/data/time_at_activity.csv          — hourly time-in-activity rates
  occupant_agent/data/activity_frequency_{s}.csv    — code prevalences

Reference:
  ATUS 2022–23, BLS. Activity codes from ATUS 2023 Lexicon
  (https://www.bls.gov/tus/lexiconwex2023.pdf).
  time_at_activity.csv method: episode [start, stop] overlap at H:30,
  aggregated with population weights (tufinlwgt). See analyze.time_at_activity().
"""

from __future__ import annotations

import random
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Literal

import pandas as pd

from occupant_agent.core.base_scheduler import BaseScheduler
from occupant_agent.core.registry import register_scheduler

# ── Category definitions ──────────────────────────────────────────────────────
# Filters match scripts/atus/analyze.py CATEGORIES exactly.

Stratum = Literal["O1", "O2", "O3", "O4"]

_CATEGORY_NAMES = ["sleeping", "work", "food_prep", "laundry", "tv", "eating", "exercise", "other"]

def _matches_category(code: str, cat: str) -> bool:
    """Return True if the ATUS code belongs to the named tod category."""
    code = code.zfill(6)
    if cat == "sleeping":
        return code.startswith("0101")
    if cat == "work":
        return code.startswith("05")
    if cat == "food_prep":
        return code.startswith("0201")
    if cat == "laundry":
        return code == "020202"
    if cat == "tv":
        return code == "120301"
    if cat == "eating":
        return code.startswith("1101")
    if cat == "exercise":
        return code.startswith("13")
    return False  # "other" is the residual

def _get_category(code: str) -> str:
    """Identify which tod category a 6-digit ATUS code belongs to."""
    for cat in _CATEGORY_NAMES[:-1]:  # exclude "other"
        if _matches_category(code, cat):
            return cat
    return "other"

# Fallback codes used when a tod category has no codes in the top-50 frequency
# file (e.g., laundry is never in top 50 but IS in the tod distribution).
# Each fallback is the canonical tier-3 code for that activity.
_CATEGORY_FALLBACKS: dict[str, str] = {
    "sleeping":  "010101",
    "work":      "050101",
    "food_prep": "020101",
    "laundry":   "020202",  # the main fallback case
    "tv":        "120301",
    "eating":    "110101",
    "exercise":  "130101",
    "other":     "010201",  # grooming — neutral catch-all
}


# ── Scheduler ─────────────────────────────────────────────────────────────────

@register_scheduler("atus")
class ActivityScheduler(BaseScheduler):
    """
    Samples ATUS activity codes from empirical time-of-day distributions.

    Usage:
        scheduler = ActivityScheduler(stratum="O1", seed=42)
        code = scheduler.sample(timestep)
        # e.g. "010101" (Sleeping) at 3am, "050101" (Work) at 9am

    The same seed produces the same sequence of codes for reproducibility.
    Pass seed=None for stochastic simulation.
    """

    _DEFAULT_OUTPUTS = Path(__file__).resolve().parent.parent / "data"

    def __init__(
        self,
        stratum: Stratum,
        seed: int | None = None,
        outputs_dir: Path | str | None = None,
    ) -> None:
        self.stratum = stratum
        self._rng = random.Random(seed)
        self._outputs = Path(outputs_dir) if outputs_dir else self._DEFAULT_OUTPUTS
        self._validate_outputs()

    # ── Public API ─────────────────────────────────────────────────────────

    def sample(self, timestep: datetime) -> str:
        """
        Sample an ATUS activity code for the given simulation timestep.

        Uses the weekday/weekend split from time_at_activity.csv:
          weekday = Mon–Fri (timestep.weekday() ∈ {0,1,2,3,4})
          weekend = Sat–Sun (timestep.weekday() ∈ {5,6})

        Args:
            timestep: Simulation datetime.

        Returns:
            6-digit ATUS tier-3 code string (e.g., "010101", "050101").
        """
        day_type = "weekend" if timestep.weekday() >= 5 else "weekday"
        cat = self._sample_category(timestep.hour, day_type)
        return self._sample_code(cat)

    def category_weights(self, hour: int, timestep: datetime | None = None) -> dict[str, float]:
        """
        Return the normalized probability distribution over activity categories
        at a given hour of day. Pass timestep to auto-detect weekday/weekend.

        Args:
            hour:     Integer 0–23.
            timestep: Optional datetime; if provided, weekday/weekend is inferred.
                      If None, returns the weekday distribution.

        Returns:
            Dict mapping category name → probability (sums to 1.0).
        """
        day_type = "weekend" if (timestep and timestep.weekday() >= 5) else "weekday"
        raw = self._weights_by_hour.get(day_type, {}).get(
            hour, {cat: 1.0 for cat in _CATEGORY_NAMES}
        )
        total = sum(raw.values())
        if total == 0:
            if 0 <= hour <= 3:
                return {cat: (1.0 if cat == "sleeping" else 0.0) for cat in _CATEGORY_NAMES}
            return {cat: 1 / len(_CATEGORY_NAMES) for cat in _CATEGORY_NAMES}
        return {cat: w / total for cat, w in raw.items()}

    # ── Lazy data loading ──────────────────────────────────────────────────

    @cached_property
    def _taa(self) -> pd.DataFrame:
        """time_at_activity.csv filtered to this stratum."""
        df = pd.read_csv(self._outputs / "time_at_activity.csv")
        return df[df["stratum"] == self.stratum].copy()

    @cached_property
    def _freq(self) -> pd.DataFrame:
        """activity_frequency_{stratum}.csv with zero-padded codes."""
        df = pd.read_csv(self._outputs / f"activity_frequency_{self.stratum}.csv")
        df["trcode"] = df["trcode"].astype(str).str.zfill(6)
        df["category"] = df["trcode"].apply(_get_category)
        return df

    @cached_property
    def _code_pools(self) -> dict[str, list[tuple[str, float]]]:
        """
        Per-category pool of (code, pct_diary_time) pairs for code sampling.
        Categories with no codes in the frequency file use the hardcoded fallback.
        """
        pools: dict[str, list[tuple[str, float]]] = {}
        for cat in _CATEGORY_NAMES:
            rows = self._freq[self._freq["category"] == cat]
            if len(rows) == 0:
                pools[cat] = [(_CATEGORY_FALLBACKS[cat], 1.0)]
            else:
                pools[cat] = list(
                    zip(rows["trcode"].tolist(), rows["pct_diary_time"].tolist())
                )
        return pools

    @cached_property
    def _weights_by_hour(self) -> dict[str, dict[int, dict[str, float]]]:
        """
        {day_type: {hour: {category: weighted_pct}}} from time_at_activity.csv.

        day_type is "weekday" or "weekend". If the CSV pre-dates the day_type
        split (no day_type column), all rows are treated as "weekday" so the
        scheduler degrades gracefully to the previous blended behaviour.
        """
        result: dict[str, dict[int, dict[str, float]]] = {
            "weekday": {h: {} for h in range(24)},
            "weekend": {h: {} for h in range(24)},
        }

        has_day_type = "day_type" in self._taa.columns

        for _, row in self._taa.iterrows():
            h = int(row["hour"])
            cat = row["category"]
            dt = str(row["day_type"]) if has_day_type else "weekday"
            if 0 <= h <= 23 and cat in _CATEGORY_NAMES and dt in result:
                result[dt][h][cat] = float(row["weighted_pct"])

        # If no day_type column, mirror weekday → weekend for backward compat
        if not has_day_type:
            result["weekend"] = {h: dict(result["weekday"][h]) for h in range(24)}

        # Fill zeros for any missing category × hour combinations
        for dt in ("weekday", "weekend"):
            for h in range(24):
                for cat in _CATEGORY_NAMES:
                    result[dt][h].setdefault(cat, 0.0)

        return result

    # ── Internal sampling ──────────────────────────────────────────────────

    def _sample_category(self, hour: int, day_type: str = "weekday") -> str:
        """
        Weighted random choice over activity categories at hour H and day type.
        Weights are directly from time_at_activity.csv.
        """
        weights = self._weights_by_hour.get(day_type, {}).get(
            hour, {cat: 1.0 for cat in _CATEGORY_NAMES}
        )
        cats = list(weights.keys())
        wts = list(weights.values())
        if sum(wts) == 0:
            # Hours 0-3 are all-zero in the bundled CSV (ATUS extended-hour
            # encoding limitation). Real ATUS data shows >80% sleeping at these
            # hours, so use sleeping as the fallback rather than uniform random.
            return "sleeping" if 0 <= hour <= 3 else self._rng.choice(cats)
        return self._rng.choices(cats, weights=wts, k=1)[0]

    def _sample_code(self, category: str) -> str:
        """Weighted random choice of a tier-3 ATUS code within a category."""
        pool = self._code_pools.get(category, [(_CATEGORY_FALLBACKS.get(category, "010201"), 1.0)])
        codes, wts = zip(*pool)
        return self._rng.choices(list(codes), weights=list(wts), k=1)[0]

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate_outputs(self) -> None:
        taa_path = self._outputs / "time_at_activity.csv"
        freq_path = self._outputs / f"activity_frequency_{self.stratum}.csv"
        if not taa_path.exists():
            raise FileNotFoundError(
                f"time_at_activity.csv not found at {self._outputs}. "
                "Run scripts/atus/analyze.py first."
            )
        if not freq_path.exists():
            raise FileNotFoundError(
                f"activity_frequency_{self.stratum}.csv not found at {self._outputs}. "
                "Run scripts/atus/analyze.py first."
            )
