from __future__ import annotations
import random
import pytest
from occupant_agent.agent.persona import create_persona, Persona, _load_peak_hours


# ── 1. Basic creation ─────────────────────────────────────────────────────────

def test_create_persona_p1_seed():
    p = create_persona("O1", seed=42)
    assert p.stratum == "O1"
    assert 25 <= p.age <= 44


# ── 2. Reproducibility ────────────────────────────────────────────────────────

def test_create_persona_reproducible():
    p1 = create_persona("O1", seed=7)
    p2 = create_persona("O1", seed=7)
    assert p1.age == p2.age
    assert p1.sex == p2.sex
    assert p1.income_bracket == p2.income_bracket


# ── 3. Different seeds produce different personas ─────────────────────────────

def test_create_persona_different_seeds_differ():
    p1 = create_persona("O1", seed=1)
    p2 = create_persona("O1", seed=2)
    # At least one demographic field must differ across 100 possible ages,
    # 2 sexes, and 8 income brackets — practically guaranteed to differ.
    differ = (
        p1.age != p2.age
        or p1.sex != p2.sex
        or p1.income_bracket != p2.income_bracket
    )
    assert differ


# ── 4. All strata create successfully ─────────────────────────────────────────

def test_all_strata_create():
    for stratum in ("O1", "O2", "O3", "O4"):
        p = create_persona(stratum, seed=0)
        assert isinstance(p, Persona)
        assert p.stratum == stratum


# ── 5. Age range O1 ───────────────────────────────────────────────────────────

def test_age_range_p1():
    for seed in range(20):
        p = create_persona("O1", seed=seed)
        assert 25 <= p.age <= 44, f"seed={seed}: age={p.age} out of range"


# ── 6. Age range O2 ───────────────────────────────────────────────────────────

def test_age_range_p2():
    for seed in range(10):
        p = create_persona("O2", seed=seed)
        assert 65 <= p.age <= 85, f"seed={seed}: age={p.age} out of range"


# ── 7. Core appliances always present ─────────────────────────────────────────

def test_core_appliances_always_present():
    required = {"hvac", "thermostat", "tv", "refrigerator"}
    for stratum in ("O1", "O2", "O3", "O4"):
        p = create_persona(stratum, seed=99)
        assert required <= p.appliances, (
            f"{stratum}: missing {required - p.appliances}"
        )


# ── 8. core_memory_text is a non-empty string > 100 chars ────────────────────

def test_core_memory_text_nonempty():
    p = create_persona("O1", seed=42)
    assert isinstance(p.core_memory_text, str)
    assert len(p.core_memory_text) > 100


# ── 9. core_memory_text contains stratum-appropriate keywords ─────────────────

def test_core_memory_text_contains_stratum_hint():
    p1 = create_persona("O1", seed=42)
    text1 = p1.core_memory_text.lower()
    assert "professional" in text1 or "work" in text1

    p2 = create_persona("O2", seed=42)
    text2 = p2.core_memory_text.lower()
    assert "retire" in text2


# ── 10. schedule_priors has required keys ─────────────────────────────────────

def test_schedule_priors_has_required_keys():
    required_keys = {"sleep", "work", "food_prep", "laundry", "tv", "exercise", "eating"}
    for stratum in ("O1", "O2", "O3", "O4"):
        p = create_persona(stratum, seed=0)
        assert required_keys <= p.schedule_priors.keys(), (
            f"{stratum}: missing {required_keys - p.schedule_priors.keys()}"
        )


# ── 11. schedule_priors hour values in [0, 23] ───────────────────────────────

def test_schedule_priors_hour_range():
    for stratum in ("O1", "O2", "O3", "O4"):
        p = create_persona(stratum, seed=0)
        for activity, entry in p.schedule_priors.items():
            hour = entry[0]
            assert 0 <= hour <= 23, (
                f"{stratum}/{activity}: hour={hour} out of [0, 23]"
            )


# ── 12. _load_peak_hours returns dict or None ─────────────────────────────────

def test_load_peak_hours_returns_dict():
    result = _load_peak_hours("O1")
    # Returns dict (possibly empty) when CSV exists, or None when it doesn't.
    assert result is None or isinstance(result, dict)
    if isinstance(result, dict) and result:
        # If populated, all values must be valid hours.
        for cat, hour in result.items():
            assert 0 <= int(hour) <= 23, f"hour={hour} out of range for category {cat}"


# ── 13. income_bracket override ───────────────────────────────────────────────

def test_income_bracket_override():
    p = create_persona("O1", seed=42, income_bracket=16)
    assert p.income_bracket == 16


# ── 14. WFH probability ───────────────────────────────────────────────────────

def test_wfh_probability_p1():
    p1 = create_persona("O1", seed=42)
    assert p1.wfh_probability > 0

    p2 = create_persona("O2", seed=42)
    assert p2.wfh_probability == 0.0


# ── 15. sample_wfh_today returns bool ────────────────────────────────────────

def test_sample_wfh_today():
    p = create_persona("O1", seed=42)
    rng = random.Random(0)
    result = p.sample_wfh_today(rng)
    assert isinstance(result, bool)


# ── 16. comfort_band_c varies inversely with income ──────────────────────────

def test_comfort_band_c_varies_with_income():
    """Lower income → wider comfort band (more temperature tolerance to save money)."""
    low_income  = create_persona("O1", seed=42, income_bracket=2)
    high_income = create_persona("O1", seed=42, income_bracket=15)
    assert low_income.comfort_band_c > high_income.comfort_band_c


def test_comfort_band_c_in_core_memory_text():
    """comfort_band_c is reflected in the LLM prompt so it affects reasoning."""
    p = create_persona("O1", seed=42)
    assert str(int(p.comfort_band_c)) in p.core_memory_text
