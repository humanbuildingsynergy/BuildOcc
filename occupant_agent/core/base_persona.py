"""
BasePersona — abstract contract for occupant demographic profiles.

Subclass this to define a new grounding source (custom survey, synthetic
population, Homer dataset, etc.) without modifying the core agent logic.

All attributes consumed by OccupantAgent.step() and receive_signal() are
declared here. Concrete implementations must provide all of them.

Example
───────
    from occupant_agent.core import BasePersona, register_stratum
    import random

    @register_stratum("P5")
    class LowIncomeElderlyAlone(BasePersona):
        \"\"\"Single elderly adult, low income, not retired (working poor).\"\"\"

        def __init__(self, seed: int | None = None) -> None:
            rng = random.Random(seed)
            self._age = rng.randint(65, 80)

        @property
        def stratum(self) -> str:
            return "P5"

        @property
        def age(self) -> int:
            return self._age

        @property
        def sex(self) -> str:
            return "female"

        @property
        def income_bracket(self) -> int:
            return 3  # low income (1–16 scale, ATUS/CPS)

        @property
        def work_from_home(self) -> bool:
            return False

        @property
        def home_gym(self) -> bool:
            return False

        @property
        def wfh_probability(self) -> float:
            return 0.0

        @property
        def comfort_band_c(self) -> float:
            return 2.2  # wider band — cost-sensitive

        @property
        def appliances(self) -> set[str]:
            return {"hvac", "thermostat", "tv", "refrigerator"}

        @property
        def schedule_priors(self) -> dict[str, tuple]:
            return {
                "sleep": (22, 2.0), "work": (9, 2.0), "food_prep": (17, 1.5),
                "laundry": (10, 2.0), "tv": (19, 2.0), "exercise": (8, 1.5),
                "eating": (12, 1.0),
            }

        @property
        def core_memory_text(self) -> str:
            return (
                f"I am a {self._age}-year-old woman living alone on a tight budget. "
                "I keep my home between 20°C and 22°C to save money. "
                "I am retired but still work part-time. I watch TV most evenings."
            )

        def sample_wfh_today(self, rng: random.Random) -> bool:
            return False
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class BasePersona(ABC):
    """
    Abstract base class for occupant demographic profiles.

    Defines the interface consumed by OccupantAgent. Subclass this —
    do NOT modify persona.py — to introduce a new demographic profile,
    a new population survey grounding, or a synthetic persona.

    Registration makes the subclass discoverable:
        @register_stratum("P5")
        class MyPersona(BasePersona): ...
        OccupantAgent.from_stratum("P5")  # works
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def stratum(self) -> str:
        """Short identifier (e.g., 'O1', 'P5', 'custom_wfh')."""

    @property
    @abstractmethod
    def age(self) -> int:
        """Age in years."""

    @property
    @abstractmethod
    def sex(self) -> str:
        """'male' or 'female'."""

    @property
    @abstractmethod
    def income_bracket(self) -> int:
        """Income bracket 1–16 (ATUS/CPS HEFAMINC scale)."""

    # ── Behavioral traits ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def work_from_home(self) -> bool:
        """Whether this persona's base profile includes WFH."""

    @property
    @abstractmethod
    def home_gym(self) -> bool:
        """Whether this persona has home exercise equipment."""

    @property
    @abstractmethod
    def wfh_probability(self) -> float:
        """Per-day probability of working from home (0.0 if not employed)."""

    @property
    @abstractmethod
    def comfort_band_c(self) -> float:
        """
        Acceptable temperature deviation from setpoint (°C) before the
        agent acts. Higher = more tolerant (typically lower income).
        """

    @property
    @abstractmethod
    def appliances(self) -> set[str]:
        """Set of device_id strings present in this household."""

    @property
    @abstractmethod
    def schedule_priors(self) -> dict[str, tuple]:
        """
        Per-activity (peak_hour, std_dev_hours) priors used in core_memory_text.
        Keys must include: sleep, work, food_prep, laundry, tv, exercise, eating.
        """

    # ── LLM interface ─────────────────────────────────────────────────────────

    @property
    def prompt_suffix(self) -> str:
        """
        Optional text appended to every system prompt for this persona.
        Override in custom strata to inject stratum-specific reasoning context
        without subclassing OccupantAgent.
        Default: empty string (no suffix).
        """
        return ""

    @property
    @abstractmethod
    def core_memory_text(self) -> str:
        """
        Natural-language persona description injected as the LLM system prompt.
        Should describe: daily routine, device preferences, thermal comfort
        tolerance, and economic framing of energy decisions.
        """

    @abstractmethod
    def sample_wfh_today(self, rng: random.Random) -> bool:
        """
        Sample whether today is a WFH day for this persona.
        Called once per simulated day; result passed to OccupantAgent.step().
        """
