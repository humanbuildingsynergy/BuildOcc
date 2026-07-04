"""
Conformance tests for BasePersona and BaseScheduler extension authors.

Call these from your test suite to verify that your subclass correctly
implements the OccupantAgent extension contract before publishing it.

Usage
─────
    # In your test file (pytest):
    from occupant_agent.testing import assert_persona_contract, assert_scheduler_contract

    def test_my_persona_contract():
        assert_persona_contract(MyPersona(seed=0))

    def test_my_scheduler_contract():
        assert_scheduler_contract(MyScheduler(stratum="O1", seed=0))

Each function raises AssertionError with a descriptive message on the
first violation, so pytest output clearly identifies what contract
was broken and why.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from occupant_agent.core.base_persona import BasePersona
from occupant_agent.core.base_scheduler import BaseScheduler

# Valid 6-digit ATUS code categories for checking scheduler output
_KNOWN_CATEGORIES = {
    "sleeping", "work", "food_prep", "laundry",
    "tv", "eating", "exercise", "other",
}

# Timestamps covering weekday and weekend, all hours
_WEEKDAY = datetime(2024, 7, 15, 0, 0)   # Monday
_WEEKEND = datetime(2024, 7, 20, 0, 0)   # Saturday


def assert_persona_contract(persona: Any, *, stratum: str | None = None) -> None:
    """
    Assert that `persona` fully satisfies the BasePersona contract.

    Checks every abstract property for correct type and value range.
    Raises AssertionError with a descriptive message on the first failure.

    Args:
        persona:  Instance of a BasePersona subclass (or Persona dataclass).
        stratum:  Expected stratum key — asserted if provided.

    Example:
        def test_my_persona():
            assert_persona_contract(MyPersona(seed=0), stratum="P5")
    """
    cls = type(persona).__name__

    # isinstance check (covers both formal subclasses and virtual registrations)
    assert isinstance(persona, BasePersona), (
        f"{cls} is not an instance of BasePersona. "
        "Subclass BasePersona or call BasePersona.register(YourClass)."
    )

    # stratum
    _check_attr(persona, "stratum", str, cls)
    assert persona.stratum, f"{cls}.stratum must be a non-empty string."
    if stratum is not None:
        assert persona.stratum == stratum, (
            f"{cls}.stratum == {persona.stratum!r}, expected {stratum!r}."
        )

    # identity
    _check_attr(persona, "age", int, cls)
    assert persona.age > 0, f"{cls}.age must be positive, got {persona.age}."

    _check_attr(persona, "sex", str, cls)
    assert persona.sex in {"male", "female"}, (
        f"{cls}.sex must be 'male' or 'female', got {persona.sex!r}."
    )

    _check_attr(persona, "income_bracket", int, cls)
    assert 1 <= persona.income_bracket <= 16, (
        f"{cls}.income_bracket must be 1–16 (HEFAMINC scale), "
        f"got {persona.income_bracket}."
    )

    # behavioral traits
    _check_attr(persona, "work_from_home", bool, cls)
    _check_attr(persona, "home_gym", bool, cls)

    _check_attr(persona, "wfh_probability", float, cls)
    assert 0.0 <= persona.wfh_probability <= 1.0, (
        f"{cls}.wfh_probability must be in [0, 1], got {persona.wfh_probability}."
    )

    _check_attr(persona, "comfort_band_c", float, cls)
    assert persona.comfort_band_c > 0, (
        f"{cls}.comfort_band_c must be positive (°C), got {persona.comfort_band_c}."
    )

    # appliances
    appliances = getattr(persona, "appliances", None)
    assert isinstance(appliances, set), (
        f"{cls}.appliances must be a set, got {type(appliances).__name__}."
    )
    for a in appliances:
        assert isinstance(a, str), (
            f"{cls}.appliances contains non-string element: {a!r}."
        )

    # schedule_priors — dict is allowed to be empty for custom grounding sources
    priors = getattr(persona, "schedule_priors", None)
    assert isinstance(priors, dict), (
        f"{cls}.schedule_priors must be a dict, got {type(priors).__name__}."
    )

    # core_memory_text
    _check_attr(persona, "core_memory_text", str, cls)
    assert persona.core_memory_text, (
        f"{cls}.core_memory_text must be a non-empty string."
    )

    # sample_wfh_today
    assert callable(getattr(persona, "sample_wfh_today", None)), (
        f"{cls} must implement sample_wfh_today(rng)."
    )
    rng = random.Random(0)
    result = persona.sample_wfh_today(rng)
    assert isinstance(result, bool), (
        f"{cls}.sample_wfh_today() must return bool, got {type(result).__name__}."
    )


def assert_scheduler_contract(
    scheduler: Any,
    *,
    n_samples: int = 96,
) -> None:
    """
    Assert that `scheduler` fully satisfies the BaseScheduler contract.

    Tests sample() and category_weights() across all 24 hours for both
    weekday and weekend timestamps.

    Args:
        scheduler: Instance of a BaseScheduler subclass.
        n_samples: Number of timesteps to sample per day type (default 96 = one day).

    Example:
        def test_my_scheduler():
            assert_scheduler_contract(MyScheduler(stratum="O1", seed=0))
    """
    cls = type(scheduler).__name__

    assert isinstance(scheduler, BaseScheduler), (
        f"{cls} is not an instance of BaseScheduler. Subclass BaseScheduler."
    )

    # sample() — weekday and weekend, full day
    step = timedelta(minutes=15)
    for label, base in (("weekday", _WEEKDAY), ("weekend", _WEEKEND)):
        ts = base
        for i in range(n_samples):
            code = scheduler.sample(ts)
            assert isinstance(code, str), (
                f"{cls}.sample() must return str, got {type(code).__name__} "
                f"at {ts} ({label})."
            )
            assert len(code) == 6, (
                f"{cls}.sample() must return a 6-digit ATUS code, "
                f"got {code!r} (len={len(code)}) at {ts} ({label})."
            )
            assert code.isdigit(), (
                f"{cls}.sample() must return a numeric string, got {code!r} "
                f"at {ts} ({label})."
            )
            ts += step

    # category_weights() — all 24 hours, both day types
    for label, ts_day in (("weekday", _WEEKDAY), ("weekend", _WEEKEND)):
        for h in range(24):
            weights = scheduler.category_weights(h, ts_day + timedelta(hours=h))
            assert isinstance(weights, dict), (
                f"{cls}.category_weights() must return dict, "
                f"got {type(weights).__name__} at hour {h} ({label})."
            )
            assert weights, (
                f"{cls}.category_weights() returned empty dict at hour {h} ({label})."
            )
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, (
                f"{cls}.category_weights() values must sum to ~1.0, "
                f"got sum={total:.6f} at hour {h} ({label})."
            )
            for cat, w in weights.items():
                assert isinstance(cat, str), (
                    f"{cls}.category_weights() keys must be str, got {type(cat).__name__}."
                )
                assert isinstance(w, (int, float)), (
                    f"{cls}.category_weights() values must be numeric, "
                    f"got {type(w).__name__} for category {cat!r}."
                )
                assert w >= 0, (
                    f"{cls}.category_weights() values must be non-negative, "
                    f"got {w} for {cat!r} at hour {h} ({label})."
                )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_attr(obj: Any, attr: str, expected_type: type, cls_name: str) -> None:
    """Assert that obj.attr exists and is of expected_type."""
    val = getattr(obj, attr, _MISSING)
    assert val is not _MISSING, (
        f"{cls_name} must implement the '{attr}' property."
    )
    assert isinstance(val, expected_type), (
        f"{cls_name}.{attr} must be {expected_type.__name__}, "
        f"got {type(val).__name__} ({val!r})."
    )


class _Missing:
    pass

_MISSING = _Missing()
