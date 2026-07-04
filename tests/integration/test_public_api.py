"""
Public API surface tests.

Every symbol documented in the README must be importable and the plugin
registry must expose exactly the built-in strata and schedulers.
No API key, network access, or I/O required.
"""

from __future__ import annotations


def test_top_level_imports() -> None:
    from occupant_agent import (  # noqa: F401
        ActivityScheduler,
        AgentStore,
        DeviceState,
        MemoryEntry,
        MemoryStream,
        OccupantAgent,
        Persona,
        RoomState,
        SimulationEnvironment,
        constant_zone_temp,
        peak_tou_rate,
        summer_day_temp,
        zone_temp_from_csv,
    )


def test_core_imports() -> None:
    from occupant_agent.core import (
        list_schedulers,
        list_strata,
    )
    strata = list_strata()
    for s in ("O1", "O2", "O3", "O4"):
        assert s in strata, f"Built-in stratum {s!r} missing from registry"

    schedulers = list_schedulers()
    for sc in ("atus", "fixed"):
        assert sc in schedulers, f"Built-in scheduler {sc!r} missing from registry"


def test_analysis_imports() -> None:
    from occupant_agent.analysis import (  # noqa: F401
        SimulationLog,
        compute_cvrmse,
        compute_kl,
        compute_ks,
        compute_mbe,
    )


def test_testing_imports() -> None:
    from occupant_agent.testing import (  # noqa: F401
        MockLLMAgent,
        assert_persona_contract,
        assert_scheduler_contract,
    )


def test_entry_points_callable() -> None:
    from occupant_agent.api.app import app, run
    from occupant_agent.cli import main
    from occupant_agent.mcp_server.server import main_sync

    assert callable(main), "buildocc CLI entry point not callable"
    assert callable(run), "buildocc-api entry point not callable"
    assert callable(main_sync), "buildocc-mcp entry point not callable"
    assert app is not None, "FastAPI app object missing"


def test_all_strata_instantiate() -> None:
    from occupant_agent.testing import MockLLMAgent

    for stratum in ("O1", "O2", "O3", "O4"):
        agent = MockLLMAgent.from_stratum(stratum, seed=0)
        assert agent.persona.stratum == stratum
        assert agent.persona.age > 0


def test_plugin_registration() -> None:
    import random

    from occupant_agent import OccupantAgent
    from occupant_agent.core import BasePersona, list_strata, register_stratum

    @register_stratum("_TEST_P99")
    class _TestPersona(BasePersona):
        def __init__(self, seed=None, **kwargs):
            self._age = random.Random(seed).randint(20, 30)

        @property
        def stratum(self): return "_TEST_P99"
        @property
        def age(self): return self._age
        @property
        def sex(self): return "male"
        @property
        def income_bracket(self): return 5
        @property
        def work_from_home(self): return False
        @property
        def home_gym(self): return False
        @property
        def wfh_probability(self): return 0.0
        @property
        def comfort_band_c(self): return 2.0
        @property
        def appliances(self): return {"hvac", "tv"}
        @property
        def schedule_priors(self): return {}
        @property
        def core_memory_text(self): return "Test persona."
        def sample_wfh_today(self, rng): return False

    assert "_TEST_P99" in list_strata()
    agent = OccupantAgent.from_stratum("_TEST_P99", seed=0)
    assert agent.persona.stratum == "_TEST_P99"
