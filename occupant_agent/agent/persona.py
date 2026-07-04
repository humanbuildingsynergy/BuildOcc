"""
Persona generator: produces a structured persona and natural-language core memory
for the LLM occupant agent, grounded in ATUS-derived schedule priors.

The persona is the agent's reasoning context — it is prepended to every LLM
call so the model reasons in character. It encodes:
  - Demographic identity (from ATUS stratum)
  - Schedule priors (activity timing from ATUS frequency analysis)
  - Appliance ownership (from RECS priors)
  - Behavioral traits (cost-consciousness, tech comfort — for signal reasoning)
  - Resolved persona flags (work_from_home, home_gym — from TEWHERE analysis)

Design principle: the LLM reads this once per session and reasons from it.
Richer context → more consistent, demographically realistic decisions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Stratum type ─────────────────────────────────────────────────────────────

Stratum = Literal["O1", "O2", "O3", "O4"]

# ── ATUS-derived schedule priors (from scripts/atus/analyze.py output) ───────
# Each entry: (typical_peak_hour, typical_duration_min, note)
# Source: ATUS 2022+2023 weighted episode-overlap distributions (16,684 respondents).
# Peak hours computed from time_at_activity.csv via analyze.schedule_peak_hours()
# with weekday/weekend split. food_prep peak hours for employed strata (O1, O3)
# are kept at the intuitive weekday-dinner value (h17/h18) because the weekday
# ATUS data shows a shallow h10 peak for food_prep in employed strata — likely
# breakfast/brunch prep — while the evening dinner peak is distributed across
# h16–h19 without a sharp argmax. h17 is used as a reasonable representative.

SCHEDULE_PRIORS: dict[Stratum, dict[str, tuple[int, int, str]]] = {
    "O1": {
        "sleep":    (23, 420, "typically asleep by 11pm, wakes ~6am"),
        "work":     (8,  480, "work starts around 8am on office days"),
        "food_prep":(17, 30,  "quick weekday dinner prep around 5pm, longer on weekends"),
        "laundry":  (19, 60,  "concentrated at 7pm — primary DR target"),
        "tv":       (18, 90,  "evening screen time from around 6pm"),   # data: h18 (prev h20)
        "exercise": (19, 45,  "post-work exercise peaking around 7pm"), # data: h19 (prev h18)
        "eating":   (12, 30,  "lunch peak at noon; dinner around 7pm"), # data: h12
    },
    "O2": {
        "sleep":    (22, 480, "earlier bedtime ~10pm, wakes ~6am"),
        "work":     (7,  120, "part-time or volunteer only; irregular"),
        "food_prep":(10, 45,  "morning and midday cooking around 10am; longer sessions"), # data: h10 (prev h8)
        "laundry":  (17, 60,  "afternoon laundry around 5pm, less concentrated than O1"), # data: h17
        "tv":       (15, 150, "substantial afternoon TV starting around 3pm"),             # data: h15
        "exercise": (9,  45,  "morning exercise around 9am; mostly walking"),             # data: h9 (prev h8)
        "eating":   (17, 45,  "dinner is the main social meal around 5pm; longer"),       # data: h17 (prev h18)
    },
    "O3": {
        "sleep":    (22, 420, "later bedtime due to parenting duties; ~10pm"),
        "work":     (7,  480, "early start on office days; work drives the family's morning routine"),
        "food_prep":(17, 45,  "family dinner prep around 5pm; larger meals"),
        "laundry":  (17, 75,  "afternoon laundry around 5pm; family load volume"),        # data: h17
        "tv":       (15, 90,  "family TV starting around 3pm; afternoon and evening"),    # data: h15
        "exercise": (17, 30,  "brief post-work exercise around 5pm when possible"),       # data: h17
        "eating":   (18, 30,  "family dinner around 6pm is the main meal"),               # data: h18 (prev h12)
    },
    "O4": {
        "sleep":    (2,  480, "very late bedtime; wakes ~10am"),
        "work":     (12, 60,  "occasional gig work peaking around noon; irregular hours"), # data: h12 (prev h15)
        "food_prep":(10, 45,  "home most of day; cooking peaks around 10am"),             # data: h10 (prev h8)
        "laundry":  (18, 60,  "evening laundry around 6pm"),                              # data: h18 (prev h19)
        "tv":       (16, 180, "extended TV block from around 4pm"),                       # data: h16 (prev h18)
        "exercise": (12, 45,  "midday exercise around noon when motivated"),               # data: h12 (prev h11)
        "eating":   (12, 45,  "lunch is the biggest meal around noon; dinner also eaten"), # data: h12 (prev h18)
    },
}

# ── RECS appliance ownership priors ──────────────────────────────────────────
# Probability that a stratum member owns each appliance.
# Source: EIA RECS 2020, Table CE4; adjusted by income stratum for O1 vs. O4.

APPLIANCE_PRIORS: dict[Stratum, dict[str, float]] = {
    "O1": {
        "dishwasher":      0.65,
        "washer":          0.75,
        "dryer":           0.72,
        "smart_thermostat":0.35,
        "ev_charger":      0.08,
        "pool_pump":       0.05,
        "home_gym":        0.15,  # treadmill/stationary bike (RECS + TEWHERE)
    },
    "O2": {
        "dishwasher":      0.72,
        "washer":          0.88,
        "dryer":           0.85,
        "smart_thermostat":0.28,
        "ev_charger":      0.05,
        "pool_pump":       0.12,
        "home_gym":        0.10,
    },
    "O3": {
        "dishwasher":      0.78,
        "washer":          0.90,
        "dryer":           0.88,
        "smart_thermostat":0.40,
        "ev_charger":      0.10,
        "pool_pump":       0.08,
        "home_gym":        0.18,
    },
    "O4": {
        "dishwasher":      0.40,
        "washer":          0.55,
        "dryer":           0.50,
        "smart_thermostat":0.15,
        "ev_charger":      0.02,
        "pool_pump":       0.02,
        "home_gym":        0.05,
    },
}

# ── RECS-grounded default room sets ──────────────────────────────────────────
# Based on EIA RECS 2020 housing unit characteristics by housing type.
# O1 (single adult): apartment/small house — fewer dedicated rooms.
# O2 (retired couple): larger single-family home — dining room, spare bedroom.
# O3 (family with children): family home — kids' bedroom, laundry room.
# O4 (unemployed, alone or roommates): small apartment or shared house.
# home_office is added dynamically for strata with WFH probability > 0.

ROOM_DEFAULTS: dict[str, list[str]] = {
    "O1": ["living_room", "kitchen", "bedroom"],
    "O2": ["living_room", "kitchen", "master_bedroom", "dining_room", "spare_bedroom"],
    "O3": ["living_room", "kitchen", "master_bedroom", "kids_bedroom", "laundry_room"],
    "O4": ["living_room", "kitchen", "bedroom"],
}

# WFH probabilities (empirical from ATUS 2022-23 TEWHERE analysis, D4)
WFH_PRIORS: dict[Stratum, float] = {
    "O1": 0.243,
    "O2": 0.0,    # not employed
    "O3": 0.257,
    "O4": 0.0,    # not employed
}

# ── Persona dataclass ─────────────────────────────────────────────────────────

@dataclass
class Persona:
    """
    Structured persona for one simulated occupant.

    This is the agent's stable identity — it does not change within a simulation
    run unless a high-importance memory is promoted to core memory (Phase 3).

    Fields:
      stratum:          demographic stratum (O1–O4)
      age:              sampled age within stratum range
      sex:              sampled sex
      income_bracket:   1–16 ordinal (EIA RECS / ATUS CPS HEFAMINC scale)
      state_fips:       US state FIPS code (for climate zone / TOU rate lookup)
      work_from_home:   True on WFH days (sampled per-day from WFH prior)
      home_gym:         True if persona owns home exercise equipment
      appliances:       set of device_ids this persona owns
      schedule_priors:  ATUS-derived activity timing summary (for LLM context)
      core_memory_text: natural-language description passed to LLM at every step
    """

    stratum:           Stratum
    age:               int
    sex:               Literal["male", "female"]
    income_bracket:    int            # 1–16 (HEFAMINC scale)
    state_fips:        int            # 48 = Texas (Pecan Street), etc.
    work_from_home:    bool
    home_gym:          bool
    comfort_band_c:    float          # °C deviation from setpoint before acting
    appliances:        set[str]
    schedule_priors:   dict[str, tuple[int, int, str]]
    core_memory_text:  str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.core_memory_text:
            self.core_memory_text = _build_core_memory(self)

    @property
    def prompt_suffix(self) -> str:
        return ""

    @property
    def wfh_probability(self) -> float:
        return WFH_PRIORS[self.stratum]

    def sample_wfh_today(self, rng: random.Random | None = None) -> bool:
        """Sample whether today is a WFH day (call once per simulated day)."""
        r = rng or random
        return r.random() < self.wfh_probability

    @property
    def room_ids(self) -> list[str]:
        """
        RECS-grounded default room list for this stratum.
        home_office is included when the persona works from home.
        Use as initial_rooms for SimulationEnvironment if you don't have
        a building model that specifies rooms.
        """
        rooms = list(ROOM_DEFAULTS[self.stratum])
        if self.work_from_home and "home_office" not in rooms:
            rooms.append("home_office")
        return rooms


# ── Income helpers ────────────────────────────────────────────────────────────

def _income_to_comfort_band(income_bracket: int) -> float:
    """
    Temperature comfort band (°C from setpoint) by income bracket.

    Lower-income households tolerate wider indoor temperature swings; higher-income
    households act sooner on small deviations. Directionally consistent with EIA
    RECS 2020 (Table HC6.1), which shows lower-income households report wider
    actual indoor temperature variation. The specific thresholds (0.8–2.2°C)
    are calibrated to produce behaviorally plausible decision rates across strata,
    not read directly from a RECS table.
    """
    if income_bracket <= 4:
        return 2.2   # very low income: tolerate wide swings
    elif income_bracket <= 8:
        return 1.7   # low-moderate
    elif income_bracket <= 12:
        return 1.1   # moderate
    else:
        return 0.8   # high income: sensitive to small deviations


# ── Factory ───────────────────────────────────────────────────────────────────

def create_persona(
    stratum: Stratum,
    seed: int | None = None,
    state_fips: int = 48,           # default: Texas (Pecan Street cohort)
    income_bracket: int | None = None,
) -> Persona:
    """
    Sample a persona from the given ATUS stratum.

    All stochastic choices use a seeded RNG for reproducibility.
    Each unique (stratum, seed) pair produces a distinct but deterministic persona.

    Args:
        stratum:        O1–O4 demographic stratum
        seed:           RNG seed (None = random)
        state_fips:     US state for climate/TOU context (default 48 = Texas)
        income_bracket: override income bracket (1–16); sampled if None
    """
    rng = random.Random(seed)

    # Sample age within stratum range
    age_ranges = {"O1": (25, 44), "O2": (65, 85), "O3": (35, 54), "O4": (25, 44)}
    age = rng.randint(*age_ranges[stratum])

    sex: Literal["male", "female"] = rng.choice(["male", "female"])  # type: ignore

    # Sample income bracket (rough ATUS/RECS prior per stratum)
    income_bracket_priors: dict[Stratum, tuple[int, int]] = {
        "O1": (5, 12),   # moderate income range
        "O2": (4, 10),   # retirees: lower-moderate
        "O3": (7, 14),   # employed parents: higher income
        "O4": (1, 6),    # unemployed: lower income
    }
    lo, hi = income_bracket_priors[stratum]
    inc = income_bracket if income_bracket is not None else rng.randint(lo, hi)

    # Income-modulated appliance ownership: position within stratum range scales
    # base RECS priors up (high income) or down (low income) by up to ±30%.
    income_position = (inc - lo) / max(hi - lo, 1)   # 0.0 = bottom, 1.0 = top
    appliance_scale = 0.7 + 0.6 * income_position    # 0.70× to 1.30×

    base_priors = APPLIANCE_PRIORS[stratum]
    appliances = {
        device
        for device, prob in base_priors.items()
        if rng.random() < min(1.0, prob * appliance_scale)
    }

    # Always include core appliances
    appliances |= {"thermostat", "hvac", "tv", "microwave", "refrigerator",
                   "lighting_living", "lighting_bedroom", "lighting_kitchen"}

    home_gym = "home_gym" in appliances
    appliances.discard("home_gym")  # flag → not a device ID

    # WFH: sampled once as a persona trait (days are sampled per-day in simulation)
    work_from_home = rng.random() < WFH_PRIORS[stratum]

    comfort_band_c = _income_to_comfort_band(inc)  # °C

    priors_for_stratum = SCHEDULE_PRIORS[stratum]

    return Persona(
        stratum=stratum,
        age=age,
        sex=sex,
        income_bracket=inc,
        state_fips=state_fips,
        work_from_home=work_from_home,
        home_gym=home_gym,
        comfort_band_c=comfort_band_c,
        appliances=appliances,
        schedule_priors=priors_for_stratum,
    )


# ── Peak-hour CSV loader ──────────────────────────────────────────────────────

def _load_peak_hours(
    stratum: Stratum,
    day_type: str = "weekday",
    outputs_dir: Path | None = None,
    min_pct: float = 1.0,
) -> dict[str, int] | None:
    """
    Load per-category peak hours from schedule_peak_hours.csv.

    Returns {category: peak_hour} for the given stratum, or None if the CSV
    doesn't exist. Categories with peak_pct < min_pct are excluded — they are
    too rare to reliably identify a peak hour from the blended distribution
    (e.g., laundry for O1 shows 0.05% at all hours; the argmax is noise).

    When the CSV has a day_type column (after re-running analyze.py with real
    ATUS microdata + C2 fix), returns the requested day_type peaks — weekday
    peaks are most relevant for core_memory_text for employed strata (O1, O3).

    Callers must apply domain knowledge: food_prep peak hours for O1/O3 in
    blended CSVs (no day_type column) are biased toward noon by weekend lunch
    cooking and should not override the hardcoded h17 weekday estimate.
    """
    try:
        import pandas as pd
        path = (outputs_dir or _OUTPUTS_DIR) / "schedule_peak_hours.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        sub = df[(df["stratum"] == stratum) & (df["peak_pct"] >= min_pct)]
        if "day_type" in df.columns:
            sub = sub[sub["day_type"] == day_type]
        if sub.empty:
            return {}
        return dict(zip(sub["category"].tolist(), sub["peak_hour"].astype(int).tolist()))
    except Exception:
        return None


# ── Core memory text builder ──────────────────────────────────────────────────

def _build_core_memory(p: Persona) -> str:
    """
    Produce the natural-language core memory string passed to the LLM at every
    step(). This is the primary reasoning context — it should be complete enough
    that the LLM can make consistent, demographically realistic decisions without
    any other background information.

    Design principles:
    - Write in second person ("You are...") so the LLM reasons as the character.
    - Include ATUS-derived schedule priors as natural language, not numbers.
    - Flag behavioral traits that affect signal reasoning (cost-consciousness,
      tech comfort) — these are what make Type A/B/C signal responses differ.
    - Keep under ~600 tokens so it fits in context alongside the full memory
      stream and environment state at each step().
    """

    stratum_descriptions = {
        "O1": (
            f"a {p.age}-year-old {p.sex} professional living alone. "
            "You work full-time and value your independence and privacy."
        ),
        "O2": (
            f"a {p.age}-year-old {p.sex} retiree living with your spouse/partner. "
            "You are home most of the day and follow a fairly regular daily routine."
        ),
        "O3": (
            f"a {p.age}-year-old {p.sex} employed parent living with your family including children under 18. "
            "Your schedule is driven by work hours and family responsibilities."
        ),
        "O4": (
            f"a {p.age}-year-old {p.sex} currently not employed, living alone or with roommates. "
            "Your schedule is flexible but irregular."
        ),
    }

    # (cost attitude, signal preference) — both go into the LLM prompt
    cost_desc: dict[range, tuple[str, str]] = {
        range(1, 5): (
            "very cost-conscious — you watch every dollar on utilities",
            "A message explaining specific dollar savings gets your attention immediately. "
            "Vague efficiency tips do not.",
        ),
        range(5, 9): (
            "moderately cost-conscious — you notice energy bills and try to keep them reasonable",
            "Concrete savings amounts motivate you more than abstract efficiency advice.",
        ),
        range(9, 13): (
            "mildly cost-conscious — you're aware of costs but comfort usually wins",
            "You respond to clear explanations but tend to prioritize convenience.",
        ),
        range(13, 17): (
            "not particularly cost-conscious — comfort and convenience take priority",
            "Small dollar savings rarely motivate you; social comparisons and physical "
            "discomfort are more persuasive.",
        ),
    }
    if not any(p.income_bracket in k for k in cost_desc):
        raise ValueError(
            f"income_bracket {p.income_bracket!r} is out of range 1–16; "
            "use create_persona() or pass a value between 1 and 16 inclusive."
        )
    cost_str, signal_pref = next(v for k, v in cost_desc.items() if p.income_bracket in k)

    wfh_str = (
        "You work from home roughly one day per week on average. "
        if p.wfh_probability > 0
        else ""
    )

    sched = p.schedule_priors

    # Load CSV-aligned peak hours; fall back to SCHEDULE_PRIORS.
    # food_prep for employed strata (O1, O3) is exempted: blended CSV shows a
    # noon artifact from weekend lunch cooking; hardcoded h17 is more accurate.
    _peak = _load_peak_hours(p.stratum) or {}
    _employed = p.stratum in ("O1", "O3")

    def _h(cat: str) -> int:
        """Peak hour from CSV (with food_prep exemption for employed strata)."""
        if cat == "food_prep" and _employed:
            return sched[cat][0]
        return _peak.get(cat, sched[cat][0])

    sleep_start = sched["sleep"][0]
    sleep_dur_h = sched["sleep"][1] // 60

    appliance_list = sorted(
        a for a in p.appliances
        if a not in {"thermostat", "hvac", "lighting_living",
                     "lighting_bedroom", "lighting_kitchen"}
    )

    gym_str = (
        "You have exercise equipment at home (treadmill or stationary bike) "
        "and sometimes work out at home rather than going to the gym. "
        if p.home_gym else
        "You exercise outside or at a gym rather than at home. "
    )

    def cap(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    text = f"""You are {stratum_descriptions[p.stratum]}

DAILY SCHEDULE (typical):
- You usually go to sleep around {sleep_start}:00 and sleep about {sleep_dur_h} hours.
- {cap(sched['work'][2])}.
- {wfh_str}You tend to prepare food around {_h('food_prep')}:00 ({sched['food_prep'][2]}).
- You typically do laundry around {_h('laundry')}:00 ({sched['laundry'][2]}).
- You eat main meals around {_h('eating')}:00 ({sched['eating'][2]}).
- You watch TV or use screens starting around {_h('tv')}:00 ({sched['tv'][2]}).
- You typically exercise around {_h('exercise')}:00. {cap(sched['exercise'][2])}.

HOME AND APPLIANCES:
- You own: {', '.join(appliance_list) if appliance_list else 'basic appliances only'}.
- {gym_str}
- Your thermostat is your primary comfort control. You start to feel uncomfortable when
  the indoor temperature drifts more than {p.comfort_band_c:.1f}°C from your setpoint.

ENERGY AND COST:
- You are {cost_str}.
- {signal_pref}
- You are aware that electricity prices vary by time of day but may not always act on it.

PERSONALITY (for decision-making):
- You make practical decisions based on convenience and habit, not optimization.
- You are willing to change a habit if the effort is low and the benefit is clear.
- You find repeated or nagging suggestions annoying — one good explanation is enough.
- You occasionally override suggested actions when you're tired, busy, or uncomfortable.
""".strip()

    return text


# ── Built-in stratum registration ─────────────────────────────────────────────
# Registers O1-O4 factories with the plugin registry so that
# OccupantAgent.from_stratum("O1") works via the registry path and third-party
# extensions can call list_strata() and see the built-ins alongside their own.
# Uses functools.partial so the factory signature is (seed=None, state_fips=48).

import functools as _functools  # noqa: E402

from occupant_agent.core.base_persona import BasePersona as _BasePersona  # noqa: E402
from occupant_agent.core.registry import register_stratum as _register_stratum  # noqa: E402

# Register the built-in Persona dataclass as a virtual subclass of BasePersona.
# ABC.register() bypasses the abstract-method check, making isinstance(p, BasePersona)
# return True for existing Persona instances. Persona already implements every
# property BasePersona declares — the dataclass fields satisfy the interface.
_BasePersona.register(Persona)

# Register O1–O4 factories so OccupantAgent.from_stratum("O1") goes through the
# plugin registry and third-party strata can be discovered alongside the built-ins.
for _s in ("O1", "O2", "O3", "O4"):
    _register_stratum(_s)(_functools.partial(create_persona, stratum=_s))
