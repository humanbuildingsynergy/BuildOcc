from __future__ import annotations

import math
from datetime import datetime

import pytest

from occupant_agent.grounding.scheduler import ActivityScheduler

# Fixed test datetimes
MON_3AM = datetime(2024, 1, 15, 3, 0)   # Monday 03:00 — used for scheduler call tests
MON_4AM = datetime(2024, 1, 15, 4, 0)   # Monday 04:00 — sleeping weight ~90% in bundled data
MON_9AM = datetime(2024, 1, 15, 9, 0)   # Monday 09:00 — work/active window
SAT_9AM = datetime(2024, 1, 20, 9, 0)   # Saturday 09:00 — weekend


# ---------------------------------------------------------------------------
# Basic sample() contract
# ---------------------------------------------------------------------------

def test_sample_returns_six_digit_code():
    s = ActivityScheduler("O1", seed=42)
    code = s.sample(MON_9AM)
    assert isinstance(code, str)
    assert len(code) == 6
    assert code.isdigit()


def test_sample_reproducible_with_seed():
    times = [MON_3AM, MON_9AM, SAT_9AM]
    codes_a = [ActivityScheduler("O1", seed=42).sample(t) for t in times]
    codes_b = [ActivityScheduler("O1", seed=42).sample(t) for t in times]
    assert codes_a == codes_b


def test_different_seeds_differ():
    # Draw enough samples to make a collision astronomically unlikely
    codes_42 = [ActivityScheduler("O1", seed=42 + i).sample(MON_9AM) for i in range(10)]
    codes_99 = [ActivityScheduler("O1", seed=99 + i).sample(MON_9AM) for i in range(10)]
    assert codes_42 != codes_99


# ---------------------------------------------------------------------------
# Sleep behaviour at 3 am
# ---------------------------------------------------------------------------

def test_sample_midnight_mostly_sleeping():
    # Bundled time_at_activity.csv has sleeping weight ~90% at hour 4 for O1.
    sleeping_count = 0
    n = 20
    for seed in range(n):
        code = ActivityScheduler("O1", seed=seed).sample(MON_4AM)
        if code.startswith("0101"):
            sleeping_count += 1
    # With ~90% sleeping weight at 4am, expect at least 12/20 sleeping codes
    assert sleeping_count > n // 2, (
        f"Expected most 4am samples to be sleeping codes, got {sleeping_count}/{n}"
    )


def test_sample_hours_0_to_3_always_sleeping():
    # Hours 0-3 are all-zero in the bundled CSV (ATUS extended-hour encoding
    # limitation). The scheduler falls back to "sleeping" for those hours rather
    # than uniform random, matching the >80% sleeping rate in real ATUS data.
    s = ActivityScheduler("O1", seed=42)
    for h in range(4):
        ts = datetime(2024, 1, 15, h, 0)
        code = s.sample(ts)
        assert code.startswith("0101"), (
            f"Expected sleeping code at hour {h}, got {code!r}"
        )


def test_category_weights_hours_0_to_3_sleeping_dominant():
    # category_weights() should reflect the sleeping fallback for hours 0-3.
    s = ActivityScheduler("O1", seed=0)
    for h in range(4):
        weights = s.category_weights(hour=h)
        assert weights["sleeping"] == 1.0, (
            f"Expected sleeping=1.0 at hour {h}, got {weights['sleeping']}"
        )
        assert all(weights[cat] == 0.0 for cat in weights if cat != "sleeping"), (
            f"Expected all non-sleeping weights=0.0 at hour {h}"
        )


# ---------------------------------------------------------------------------
# Weekday / weekend split (graceful degradation when no day_type column)
# ---------------------------------------------------------------------------

def test_weekday_weekend_split_no_crash():
    # Whether the CSV has a day_type column or not, sampling must not raise
    s = ActivityScheduler("O1", seed=42)
    mon_code = s.sample(MON_9AM)
    sat_code = s.sample(SAT_9AM)
    assert isinstance(mon_code, str) and len(mon_code) == 6
    assert isinstance(sat_code, str) and len(sat_code) == 6


def test_weekday_weekend_distributions_differ():
    # Bundled data now has day_type column — weekday and weekend weights differ at peak work hours
    s = ActivityScheduler("O1", seed=0)
    weekday_w = s.category_weights(hour=9, timestep=MON_9AM)
    weekend_w = s.category_weights(hour=9, timestep=SAT_9AM)
    # At 9am Monday, work probability is high for O1; on Saturday it is near zero
    assert weekday_w.get("work", 0) > weekend_w.get("work", 0)


# ---------------------------------------------------------------------------
# category_weights()
# ---------------------------------------------------------------------------

def test_category_weights_sum_to_one():
    s = ActivityScheduler("O1", seed=42)
    weights = s.category_weights(hour=9)
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)


def test_category_weights_all_nonnegative():
    s = ActivityScheduler("O1", seed=42)
    for hour in range(24):
        weights = s.category_weights(hour=hour)
        assert all(v >= 0 for v in weights.values()), (
            f"Negative weight at hour {hour}: {weights}"
        )


def test_category_weights_returns_all_categories():
    expected = {"sleeping", "work", "food_prep", "laundry", "tv", "eating", "exercise", "other"}
    s = ActivityScheduler("O1", seed=42)
    assert set(s.category_weights(hour=9).keys()) == expected


# ---------------------------------------------------------------------------
# Invalid stratum raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_invalid_stratum_raises():
    with pytest.raises(FileNotFoundError):
        ActivityScheduler("P5")


# ---------------------------------------------------------------------------
# All valid strata load without error
# ---------------------------------------------------------------------------

def test_all_strata_load():
    for stratum in ("O1", "O2", "O3", "O4"):
        s = ActivityScheduler(stratum, seed=0)
        code = s.sample(MON_9AM)
        assert isinstance(code, str) and len(code) == 6, (
            f"Stratum {stratum} returned unexpected code: {code!r}"
        )
