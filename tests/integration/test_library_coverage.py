"""
Library coverage tests.

Exercises every public API surface that users access directly from Python,
covering gaps left by the e2e and REST API tests:
  - zone_temp_from_csv / outdoor_temp_from_csv / constant_zone_temp / summer_day_temp / peak_tou_rate
  - SimulationLog (record, record_signal, to_dicts, to_json, to_csv, len)
  - compute_kl / compute_ks / compute_cvrmse / compute_mbe
  - assert_persona_contract / assert_scheduler_contract
  - FixedScheduleScheduler (registered baseline)
  - typical_household_devices / typical_household_rooms / persona_devices
  - Testing module fixtures (make_env, make_peak_env, make_offpeak_env, make_persona,
    make_memory_stream, make_device, make_room)
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TS = datetime(2025, 8, 12, 10, 0, tzinfo=UTC)
_STEP = timedelta(minutes=15)


# ── Zone / outdoor temp helpers ───────────────────────────────────────────────

def test_constant_zone_temp():
    from occupant_agent import constant_zone_temp
    fn = constant_zone_temp(23.5)
    assert fn(_TS) == pytest.approx(23.5)
    assert fn(_TS + _STEP) == pytest.approx(23.5)


def test_summer_day_temp_range():
    from occupant_agent import summer_day_temp
    # peak around 18:00 local, night minimum ~22°C
    temps = [summer_day_temp(datetime(2025, 8, 12, h, 0, tzinfo=UTC)) for h in range(24)]
    assert all(15.0 < t < 45.0 for t in temps), "summer_day_temp produced out-of-range values"
    assert max(temps) > min(temps) + 5, "summer_day_temp shows no diurnal variation"


def test_peak_tou_rate_values():
    from occupant_agent import peak_tou_rate
    # 16:00 – 21:00 → peak rate
    peak_ts = datetime(2025, 8, 12, 17, 0, tzinfo=UTC)
    off_ts = datetime(2025, 8, 12, 10, 0, tzinfo=UTC)
    assert peak_tou_rate(peak_ts) > peak_tou_rate(off_ts)
    assert peak_tou_rate(off_ts) > 0


def test_zone_temp_from_csv():
    from occupant_agent import zone_temp_from_csv
    csv = REPO_ROOT / "examples" / "data" / "zone_temps_sample.csv"
    if not csv.exists():
        pytest.skip("zone_temps_sample.csv not present")
    fn = zone_temp_from_csv(str(csv))
    ts = datetime(2025, 8, 12, 10, 0, tzinfo=UTC)
    temp = fn(ts)
    assert isinstance(temp, float)
    assert 15.0 < temp < 40.0


def test_zone_temp_from_csv_nearest_neighbor():
    """Unmatched timesteps fall back to nearest neighbor."""
    from occupant_agent import zone_temp_from_csv
    csv = REPO_ROOT / "examples" / "data" / "zone_temps_sample.csv"
    if not csv.exists():
        pytest.skip("zone_temps_sample.csv not present")
    fn = zone_temp_from_csv(str(csv))
    # An unaligned timestamp (off by 3 minutes) should still return a float
    ts = datetime(2025, 8, 12, 10, 3, tzinfo=UTC)
    temp = fn(ts)
    assert isinstance(temp, float)
    assert 15.0 < temp < 40.0


# ── Analysis: metrics ─────────────────────────────────────────────────────────

def test_compute_kl_identical():
    from occupant_agent.analysis import compute_kl
    p = [0.25, 0.25, 0.25, 0.25]
    assert compute_kl(p, p) == pytest.approx(0.0, abs=1e-9)


def test_compute_kl_different():
    from occupant_agent.analysis import compute_kl
    p = [0.9, 0.1]
    q = [0.1, 0.9]
    assert compute_kl(p, q) > 0


def test_compute_kl_length_mismatch():
    from occupant_agent.analysis import compute_kl
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_kl([0.5, 0.5], [0.3, 0.3, 0.4])


def test_compute_kl_all_zero_raises():
    from occupant_agent.analysis import compute_kl
    with pytest.raises(ValueError):
        compute_kl([0.0, 0.0], [0.5, 0.5])


def test_compute_ks_identical():
    from occupant_agent.analysis import compute_ks
    p = [0.1, 0.2, 0.3, 0.4]
    stat = compute_ks(p, p)
    assert stat == pytest.approx(0.0, abs=1e-9)


def test_compute_ks_different():
    from occupant_agent.analysis import compute_ks
    p = [0.9, 0.05, 0.03, 0.02]
    q = [0.02, 0.03, 0.05, 0.9]
    stat = compute_ks(p, q)
    assert 0.0 < stat <= 1.0


def test_compute_cvrmse_perfect():
    from occupant_agent.analysis import compute_cvrmse
    vals = [1.0, 2.0, 3.0, 4.0]
    assert compute_cvrmse(vals, vals) == pytest.approx(0.0, abs=1e-9)


def test_compute_cvrmse_nonzero():
    from occupant_agent.analysis import compute_cvrmse
    measured = [10.0, 20.0, 30.0]
    simulated = [11.0, 19.0, 31.0]
    result = compute_cvrmse(measured, simulated)
    assert result > 0
    assert not math.isnan(result)


def test_compute_mbe_perfect():
    from occupant_agent.analysis import compute_mbe
    vals = [5.0, 10.0, 15.0]
    assert compute_mbe(vals, vals) == pytest.approx(0.0, abs=1e-9)


def test_compute_mbe_sign():
    from occupant_agent.analysis import compute_mbe
    measured = [10.0, 10.0, 10.0]
    simulated = [11.0, 11.0, 11.0]  # consistently over-predicts
    result = compute_mbe(measured, simulated)
    assert result > 0, "Positive bias (over-prediction) should give positive MBE"


# ── Analysis: SimulationLog ───────────────────────────────────────────────────

def _make_env():
    from occupant_agent.testing import make_env
    return make_env(timestep=_TS)


def _make_action(action_type="do_nothing"):
    from occupant_agent.environment.state import AgentAction
    return AgentAction(action_type=action_type, target_id=None, value=None, reasoning="test")


def test_simulation_log_record_and_len():
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O1", seed=0)
    env = _make_env()
    action = _make_action()
    log.record(_TS, action, env, atus_code="030101", memory_count=3)
    assert len(log) == 1


def test_simulation_log_multiple_records():
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O1", seed=0)
    env = _make_env()
    for i in range(4):
        log.record(_TS + i * _STEP, _make_action(), env, memory_count=i)
    assert len(log) == 4


def test_simulation_log_to_dicts():
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O2", seed=7)
    log.record(_TS, _make_action("adjust_thermostat"), _make_env(), atus_code="010101")
    dicts = log.to_dicts()
    assert len(dicts) == 1
    assert dicts[0]["action_type"] == "adjust_thermostat"
    assert dicts[0]["atus_code"] == "010101"


def test_simulation_log_to_json(tmp_path):
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O1", seed=0, run_id="test_run")
    log.record(_TS, _make_action(), _make_env())
    out = tmp_path / "log.json"
    log.to_json(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) > 0


def test_simulation_log_to_csv(tmp_path):
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O3", seed=0)
    log.record(_TS, _make_action(), _make_env(), wfh_today=True)
    out = tmp_path / "log.csv"
    log.to_csv(out)
    assert out.exists()
    content = out.read_text()
    assert "action_type" in content


def test_simulation_log_record_signal():
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O1", seed=0)
    log.record_signal(_TS, "A", "Save energy now.", "accepted", "Complying.")
    assert len(log) == 1
    dicts = log.to_dicts()
    assert "signal_A_accepted" in dicts[0]["action_type"]


def test_simulation_log_records_property():
    from occupant_agent.analysis import SimulationLog
    log = SimulationLog(stratum="O1", seed=0)
    log.record(_TS, _make_action(), _make_env())
    assert len(log.records) == 1


# ── Testing utilities ─────────────────────────────────────────────────────────

def test_make_env_returns_environment_state():
    from occupant_agent.environment.state import EnvironmentState
    from occupant_agent.testing import make_env
    env = make_env()
    assert isinstance(env, EnvironmentState)
    assert env.zone_temp_c > 0
    assert isinstance(env.devices, list)
    assert isinstance(env.rooms, list)


def test_make_peak_env():
    from occupant_agent.testing import make_peak_env
    env = make_peak_env()
    assert env.tou_rate > 0.15  # Peak rate


def test_make_offpeak_env():
    from occupant_agent.testing import make_offpeak_env
    env = make_offpeak_env()
    assert env.tou_rate <= 0.15


def test_make_persona_defaults():
    from occupant_agent.testing import make_persona
    p = make_persona()
    assert p.stratum in ("O1", "O2", "O3", "O4")
    assert p.age > 0


def test_make_persona_overrides():
    from occupant_agent.testing import make_persona
    p = make_persona(stratum="O2", age=70)
    assert p.stratum == "O2"
    assert p.age == 70


def test_make_memory_stream():
    from occupant_agent.testing import make_memory_stream
    ms = make_memory_stream(n_observations=4, n_reflections=1)
    assert ms.count() == 5


def test_make_device():
    from occupant_agent.environment.state import DeviceState
    from occupant_agent.testing import make_device
    d = make_device("hvac", state=True, power_w=3500)
    assert isinstance(d, DeviceState)
    assert d.device_id == "hvac"
    assert d.state is True


def test_make_room():
    from occupant_agent.environment.state import RoomState
    from occupant_agent.testing import make_room
    r = make_room("bedroom", occupied=False)
    assert isinstance(r, RoomState)
    assert r.room_id == "bedroom"
    assert r.occupied is False


# ── Conformance contracts ─────────────────────────────────────────────────────

def test_assert_persona_contract_o1():
    from occupant_agent.testing import MockLLMAgent, assert_persona_contract
    agent = MockLLMAgent.from_stratum("O1", seed=0)
    assert_persona_contract(agent.persona, stratum="O1")


def test_assert_persona_contract_all_strata():
    from occupant_agent.testing import MockLLMAgent, assert_persona_contract
    for stratum in ("O1", "O2", "O3", "O4"):
        agent = MockLLMAgent.from_stratum(stratum, seed=0)
        assert_persona_contract(agent.persona)


def test_assert_scheduler_contract_atus():
    from occupant_agent import ActivityScheduler
    from occupant_agent.testing import assert_scheduler_contract
    scheduler = ActivityScheduler(stratum="O1", seed=0)
    assert_scheduler_contract(scheduler)


def test_assert_scheduler_contract_fixed():
    from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
    from occupant_agent.testing import assert_scheduler_contract
    scheduler = FixedScheduleScheduler()
    assert_scheduler_contract(scheduler)


# ── FixedScheduleScheduler ────────────────────────────────────────────────────

def test_fixed_schedule_returns_valid_codes():
    from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
    sched = FixedScheduleScheduler()
    for h in range(24):
        ts = datetime(2025, 8, 12, h, 0, tzinfo=UTC)
        code = sched.sample(ts)
        assert code is None or (isinstance(code, str) and len(code) == 6), (
            f"Invalid code at hour {h}: {code!r}"
        )


def test_fixed_schedule_registered_in_registry():
    from occupant_agent.core import list_schedulers
    assert "fixed" in list_schedulers()


def test_fixed_schedule_day_night_differ():
    from occupant_agent.grounding.fixed_schedule import FixedScheduleScheduler
    sched = FixedScheduleScheduler()
    midnight = datetime(2025, 8, 12, 0, 0, tzinfo=UTC)
    midday = datetime(2025, 8, 12, 12, 0, tzinfo=UTC)
    assert sched.sample(midnight) != sched.sample(midday), (
        "Fixed schedule should assign different activities at midnight vs midday"
    )


# ── SimulationEnvironment helpers ─────────────────────────────────────────────

def test_typical_household_devices_not_empty():
    from occupant_agent.environment.simulation import typical_household_devices
    devices = typical_household_devices()
    assert len(devices) > 0
    ids = {d.device_id for d in devices}
    assert "hvac" in ids


def test_typical_household_rooms_not_empty():
    from occupant_agent.environment.simulation import typical_household_rooms
    rooms = typical_household_rooms()
    assert len(rooms) > 0
    ids = {r.room_id for r in rooms}
    assert "living_room" in ids


def test_persona_devices_matches_appliances():
    from occupant_agent.environment.simulation import persona_devices
    from occupant_agent.testing import MockLLMAgent
    agent = MockLLMAgent.from_stratum("O1", seed=0)
    devices = persona_devices(agent.persona)
    assert len(devices) > 0
    for d in devices:
        assert d.device_id in agent.persona.appliances


def test_typical_household_rooms_first_is_occupied():
    """Convention: the first room in the default list starts occupied."""
    from occupant_agent.environment.simulation import typical_household_rooms
    rooms = typical_household_rooms()
    assert rooms[0].occupied, "Convention: the first room in the list starts occupied"


# ── Plugin registry (completeness) ───────────────────────────────────────────

def test_built_in_strata_in_registry():
    from occupant_agent.core import list_strata
    strata = list_strata()
    for s in ("O1", "O2", "O3", "O4"):
        assert s in strata, f"Built-in stratum {s!r} missing"


def test_built_in_schedulers_in_registry():
    from occupant_agent.core import list_schedulers
    schedulers = list_schedulers()
    for sc in ("atus", "fixed"):
        assert sc in schedulers, f"Built-in scheduler {sc!r} missing"


def test_get_stratum_factory_callable():
    from occupant_agent.core import get_stratum
    for s in ("O1", "O2", "O3", "O4"):
        factory = get_stratum(s)
        assert callable(factory)


def test_get_scheduler_factory_callable():
    from occupant_agent.core import get_scheduler
    for sc in ("atus", "fixed"):
        factory = get_scheduler(sc)
        assert callable(factory)
