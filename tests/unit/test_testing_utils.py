"""
Tests for occupant_agent.testing — verifies that the test utilities work
correctly, and that the built-in strata and schedulers pass their own
conformance checks.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from occupant_agent.testing import (
    MockLLMAgent,
    assert_persona_contract,
    assert_scheduler_contract,
    make_device,
    make_env,
    make_memory_stream,
    make_offpeak_env,
    make_peak_env,
    make_persona,
    make_room,
)


# ── make_env ──────────────────────────────────────────────────────────────────

class TestMakeEnv:
    def test_defaults(self):
        env = make_env()
        assert env.zone_temp_c == 22.0
        assert env.tou_rate == 0.09
        assert len(env.devices) == 1
        assert env.devices[0].device_id == "hvac"
        assert len(env.rooms) == 1

    def test_custom_temp(self):
        env = make_env(zone_temp_c=80.0)
        assert env.zone_temp_c == 80.0

    def test_custom_timestep(self):
        ts = datetime(2024, 1, 10, 9, 0)
        env = make_env(timestep=ts)
        assert env.timestep == ts

    def test_extensions_passthrough(self):
        env = make_env(extensions={"ha": {"sensor.temp": 75}})
        assert env.extensions["ha"]["sensor.temp"] == 75

    def test_peak_preset(self):
        env = make_peak_env()
        assert env.tou_rate == 0.22
        assert env.outdoor_temp_c == 34.0

    def test_offpeak_preset(self):
        env = make_offpeak_env()
        assert env.tou_rate == 0.06

    def test_custom_devices(self):
        env = make_env(devices=[make_device("washer", state=False, power_w=500)])
        assert env.devices[0].device_id == "washer"

    def test_custom_rooms(self):
        env = make_env(rooms=[make_room("bedroom", occupied=False)])
        assert env.rooms[0].room_id == "bedroom"
        assert not env.rooms[0].occupied


# ── make_persona ──────────────────────────────────────────────────────────────

class TestMakePersona:
    def test_default_stratum(self):
        p = make_persona()
        assert p.stratum == "O1"

    def test_stratum_p2(self):
        p = make_persona("O2")
        assert p.stratum == "O2"

    def test_field_override(self):
        p = make_persona("O1", comfort_band_c=1.5)
        assert p.comfort_band_c == 1.5

    def test_reproducible(self):
        p1 = make_persona("O3", seed=7)
        p2 = make_persona("O3", seed=7)
        assert p1.age == p2.age
        assert p1.income_bracket == p2.income_bracket


# ── make_memory_stream ────────────────────────────────────────────────────────

class TestMakeMemoryStream:
    def test_entry_count(self):
        ms = make_memory_stream(n_observations=4, n_reflections=2)
        assert len(ms.entries) == 6

    def test_types(self):
        ms = make_memory_stream(n_observations=2, n_reflections=1)
        types = [e.memory_type for e in ms.entries]
        assert types.count("observation") == 2
        assert types.count("reflection") == 1

    def test_retrieve_returns_entries(self):
        ms = make_memory_stream(n_observations=5)
        results = ms.retrieve(datetime(2024, 7, 15, 18, 0), k=3)
        assert len(results) == 3


# ── MockLLMAgent ──────────────────────────────────────────────────────────────

class TestMockLLMAgent:
    def test_step_returns_action(self):
        agent = MockLLMAgent.from_stratum("O1", seed=0)
        env = make_env()
        action = agent.step(env)
        assert action.action_type in {
            "do_nothing", "toggle_device", "adjust_thermostat", "move_room"
        }

    def test_step_adds_memory(self):
        agent = MockLLMAgent.from_stratum("O1", seed=0)
        env = make_env()
        agent.step(env)
        assert len(agent.memory.entries) == 1

    def test_step_custom_response(self):
        agent = MockLLMAgent.from_stratum("O1", seed=0, step_response={
            "action_type": "toggle_device",
            "target_id": "hvac",
            "value": False,
            "reasoning": "Saving peak energy.",
        })
        env = make_env()
        action = agent.step(env)
        assert action.action_type == "toggle_device"
        assert action.target_id == "hvac"

    def test_receive_signal_returns_response(self):
        agent = MockLLMAgent.from_stratum("O2", seed=0)
        env = make_env()
        resp = agent.receive_signal("B", "Shift laundry to after 9pm.", env)
        assert resp.response in {"accepted", "rejected", "deferred"}

    def test_custom_signal_response(self):
        agent = MockLLMAgent.from_stratum("O1", seed=0,
            signal_response={"response": "rejected", "reasoning": "Not convenient."})
        resp = agent.receive_signal("A", "Turn off HVAC.", make_env())
        assert resp.response == "rejected"

    def test_reflection_fires_on_threshold(self):
        agent = MockLLMAgent.from_stratum("O1", seed=0, step_response={
            "action_type": "do_nothing",
            "reasoning": "OK.",
            "_memory_note": "High importance event.",
            "_importance": 10,
        })
        env = make_env()
        # 10 importance per step; threshold = 100 → fires after 10 steps
        for _ in range(11):
            agent.step(env)
        reflections = [e for e in agent.memory.entries if e.memory_type == "reflection"]
        assert len(reflections) == 3  # reflect() creates 3 insights

    def test_no_network_calls(self):
        # If any real LLM call were made, it would raise (no API key in CI).
        # This test passing proves MockLLMAgent is self-contained.
        import os
        orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            agent = MockLLMAgent.from_stratum("O1", seed=0)
            agent.step(make_env())
        finally:
            if orig:
                os.environ["ANTHROPIC_API_KEY"] = orig


# ── Conformance: built-in strata ──────────────────────────────────────────────

class TestPersonaConformance:
    @pytest.mark.parametrize("stratum", ["O1", "O2", "O3", "O4"])
    def test_builtin_strata_pass_contract(self, stratum):
        p = make_persona(stratum, seed=42)
        assert_persona_contract(p, stratum=stratum)

    def test_custom_persona_passes_contract(self):
        import random
        from occupant_agent.core.base_persona import BasePersona

        class MinimalPersona(BasePersona):
            def __init__(self, seed=None, **kw):
                self._age = 35
            @property
            def stratum(self): return "test"
            @property
            def age(self): return self._age
            @property
            def sex(self): return "male"
            @property
            def income_bracket(self): return 8
            @property
            def work_from_home(self): return False
            @property
            def home_gym(self): return False
            @property
            def wfh_probability(self): return 0.2
            @property
            def comfort_band_c(self): return 2.0
            @property
            def appliances(self): return {"hvac", "tv"}
            @property
            def schedule_priors(self): return {}
            @property
            def core_memory_text(self): return "I am a test persona."
            def sample_wfh_today(self, rng): return False

        assert_persona_contract(MinimalPersona())

    def test_bad_wfh_probability_caught(self):
        import random
        from occupant_agent.core.base_persona import BasePersona

        class BadPersona(BasePersona):
            def __init__(self, **kw): pass
            @property
            def stratum(self): return "bad"
            @property
            def age(self): return 30
            @property
            def sex(self): return "female"
            @property
            def income_bracket(self): return 5
            @property
            def work_from_home(self): return False
            @property
            def home_gym(self): return False
            @property
            def wfh_probability(self): return 1.5   # invalid
            @property
            def comfort_band_c(self): return 2.0
            @property
            def appliances(self): return set()
            @property
            def schedule_priors(self): return {}
            @property
            def core_memory_text(self): return "Bad."
            def sample_wfh_today(self, rng): return False

        with pytest.raises(AssertionError, match="wfh_probability"):
            assert_persona_contract(BadPersona())


# ── Conformance: built-in schedulers ─────────────────────────────────────────

class TestSchedulerConformance:
    def test_atus_scheduler_passes_contract(self):
        from occupant_agent.grounding.scheduler import ActivityScheduler
        assert_scheduler_contract(ActivityScheduler(stratum="O1", seed=0))

    def test_fixed_scheduler_passes_contract(self):
        from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
        assert_scheduler_contract(FixedScheduleScheduler())

    @pytest.mark.parametrize("stratum", ["O1", "O2", "O3", "O4"])
    def test_atus_all_strata(self, stratum):
        from occupant_agent.grounding.scheduler import ActivityScheduler
        assert_scheduler_contract(ActivityScheduler(stratum=stratum, seed=0))

    def test_bad_scheduler_caught(self):
        from occupant_agent.core.base_scheduler import BaseScheduler

        class BadScheduler(BaseScheduler):
            def sample(self, ts): return "INVALID"  # not 6 digits
            def category_weights(self, hour, ts=None): return {"sleeping": 1.0}

        with pytest.raises(AssertionError, match="6-digit"):
            assert_scheduler_contract(BadScheduler())
