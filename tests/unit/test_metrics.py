"""
Unit tests for occupant_agent.analysis.metrics and SimulationLog.

Two-tier validation coverage:
  Tier 1 (behavioral): compute_kl, compute_ks, compute_kl_by_hour
  Tier 2 (energy):     compute_cvrmse, compute_mbe
  Logging:             SimulationLog record / export behaviour
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from occupant_agent.analysis import (
    SimulationLog,
    compute_cvrmse,
    compute_kl,
    compute_kl_by_hour,
    compute_ks,
    compute_mbe,
)
from occupant_agent.environment.state import (
    AgentAction,
    DeviceState,
    EnvironmentState,
    RoomState,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_env(
    zone_temp_c: float = 22.0,
    thermostat_setpoint_c: float | None = 21.5,
) -> EnvironmentState:
    return EnvironmentState(
        timestep=datetime(2024, 7, 15, 9, 0),
        zone_temp_c=zone_temp_c,
        outdoor_temp_c=30.0,
        tou_rate=0.08,
        thermostat_setpoint_c=thermostat_setpoint_c,
        devices=[DeviceState(device_id="hvac", state=True, power_w=3500)],
        rooms=[RoomState(room_id="living_room", occupied=True)],
    )


def _make_action(action_type: str = "do_nothing") -> AgentAction:
    return AgentAction(action_type=action_type, reasoning="test")


def _make_log(n: int = 1) -> SimulationLog:
    """Return a SimulationLog with n identical records already added."""
    log = SimulationLog(stratum="O1", seed=42)
    env = _make_env()
    action = _make_action()
    ts = datetime(2024, 7, 15, 9, 0)
    for i in range(n):
        log.record(
            ts,
            action=action,
            env=env,
            atus_code="010101",
            activity_category="sleeping",
        )
    return log


# ── compute_kl ────────────────────────────────────────────────────────────────
# Verifies KL divergence: normalization, epsilon smoothing, error cases,
# and a numerically known result.

def test_kl_identical_distributions_is_zero():
    """Identical P and Q produce KL = 0."""
    result = compute_kl([0.3, 0.5, 0.2], [0.3, 0.5, 0.2])
    assert result == pytest.approx(0.0, abs=1e-12)


def test_kl_disjoint_distributions_finite_with_epsilon():
    """P has mass where Q=0; epsilon regularisation prevents crash and yields finite value."""
    result = compute_kl([1.0, 0.0], [0.0, 1.0])
    assert math.isfinite(result)
    assert result > 0


def test_kl_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_kl([0.5, 0.5], [0.3, 0.3, 0.4])


def test_kl_all_zero_p_raises():
    with pytest.raises(ValueError, match="all-zero"):
        compute_kl([0.0, 0.0], [0.5, 0.5])


def test_kl_all_zero_q_raises():
    with pytest.raises(ValueError, match="all-zero"):
        compute_kl([0.5, 0.5], [0.0, 0.0])


def test_kl_known_case_log2():
    """P=[1,0], Q=[0.5,0.5] → KL = log(2) ≈ 0.6931."""
    result = compute_kl([1, 0], [0.5, 0.5])
    assert result == pytest.approx(math.log(2), rel=1e-6)


def test_kl_unnormalized_inputs_normalized_internally():
    """Unnormalised counts produce the same KL as their normalised equivalents."""
    result_counts = compute_kl([10, 0, 0], [5, 0, 5])
    result_probs = compute_kl([1, 0, 0], [0.5, 0, 0.5])
    assert result_counts == pytest.approx(result_probs, rel=1e-9)


# ── compute_ks ────────────────────────────────────────────────────────────────
# Verifies KS statistic: identical → 0, maximally separated → 1,
# error on mismatch, range property for arbitrary valid inputs.

def test_ks_identical_distributions_is_zero():
    """Identical P and Q yield KS statistic = 0."""
    result = compute_ks([0.5, 0.3, 0.2], [0.5, 0.3, 0.2])
    assert result == pytest.approx(0.0, abs=1e-12)


def test_ks_maximally_separated_is_one():
    """P=[1,0,0], Q=[0,0,1]: CDF gap reaches 1.0."""
    result = compute_ks([1, 0, 0], [0, 0, 1])
    assert result == pytest.approx(1.0)


def test_ks_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_ks([0.5, 0.5], [0.3, 0.3, 0.4])


def test_ks_result_in_unit_interval():
    """KS statistic lies in [0, 1] for all valid PMF inputs."""
    import random
    rng = random.Random(0)
    for _ in range(30):
        n = rng.randint(2, 12)
        p = [rng.random() for _ in range(n)]
        q = [rng.random() for _ in range(n)]
        result = compute_ks(p, q)
        assert 0.0 <= result <= 1.0
        assert math.isfinite(result)


# ── compute_kl_by_hour ────────────────────────────────────────────────────────
# Verifies per-hour KL dispatch: key alignment, missing-hour fallback,
# all-zero sim → nan, and a numerically correct result.

def test_kl_by_hour_keys_match_simulated():
    """Returned dict has exactly the hours present in simulated_counts."""
    sim = {6: {"sleeping": 100, "work": 0}, 9: {"sleeping": 0, "work": 50}}
    ref = {6: {"sleeping": 80, "work": 20}, 9: {"sleeping": 10, "work": 90}}
    result = compute_kl_by_hour(sim, ref)
    assert set(result.keys()) == {6, 9}


def test_kl_by_hour_missing_ref_hour_returns_finite():
    """Hour present in sim but absent from ref uses epsilon → finite KL, no crash."""
    sim = {8: {"sleeping": 10, "work": 5}}
    ref = {}
    result = compute_kl_by_hour(sim, ref)
    assert 8 in result
    assert math.isfinite(result[8])


def test_kl_by_hour_all_zero_sim_returns_nan():
    """All-zero sim counts for a given hour → result is nan, not an exception."""
    sim = {9: {"sleeping": 0, "work": 0}}
    ref = {9: {"sleeping": 50, "work": 50}}
    result = compute_kl_by_hour(sim, ref)
    assert math.isnan(result[9])


def test_kl_by_hour_known_value():
    """Correct KL for a single-hour case matching the compute_kl known result."""
    sim = {10: {"A": 1, "B": 0}}
    ref = {10: {"A": 0.5, "B": 0.5}}
    result = compute_kl_by_hour(sim, ref)
    assert result[10] == pytest.approx(math.log(2), rel=1e-6)


# ── compute_cvrmse ────────────────────────────────────────────────────────────
# Verifies CVRMSE: perfect prediction, known numeric case, zero-mean guard,
# empty sequence guard, length mismatch guard.

def test_cvrmse_perfect_prediction_is_zero():
    """Simulated exactly equals measured → CVRMSE = 0."""
    result = compute_cvrmse([3.0, 5.0, 2.0], [3.0, 5.0, 2.0])
    assert result == pytest.approx(0.0, abs=1e-12)


def test_cvrmse_known_case():
    """measured=[2,2], simulated=[1,3]: MSE=1, RMSE=1, mean=2, CVRMSE=0.5."""
    result = compute_cvrmse([2, 2], [1, 3])
    assert result == pytest.approx(0.5, rel=1e-9)


def test_cvrmse_zero_mean_measured_raises():
    with pytest.raises(ValueError, match="zero"):
        compute_cvrmse([0.0, 0.0], [1.0, 2.0])


def test_cvrmse_empty_sequences_raises():
    with pytest.raises(ValueError):
        compute_cvrmse([], [])


def test_cvrmse_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_cvrmse([1.0, 2.0], [1.0])


def test_cvrmse_result_is_nonnegative():
    """CVRMSE is always >= 0."""
    result = compute_cvrmse([10.0, 20.0, 30.0], [11.0, 18.0, 33.0])
    assert result >= 0.0


# ── compute_mbe ───────────────────────────────────────────────────────────────
# Verifies MBE: zero bias, sign convention for over- and under-prediction,
# zero-mean guard, empty sequence guard.

def test_mbe_no_bias_is_zero():
    """Simulated == measured → MBE = 0."""
    result = compute_mbe([4.0, 8.0, 12.0], [4.0, 8.0, 12.0])
    assert result == pytest.approx(0.0, abs=1e-12)


def test_mbe_over_prediction_is_positive():
    """Simulated consistently above measured → positive MBE."""
    result = compute_mbe([10.0, 10.0], [12.0, 12.0])
    assert result > 0.0


def test_mbe_under_prediction_is_negative():
    """Simulated consistently below measured → negative MBE."""
    result = compute_mbe([10.0, 10.0], [8.0, 8.0])
    assert result < 0.0


def test_mbe_zero_mean_measured_raises():
    with pytest.raises(ValueError, match="zero"):
        compute_mbe([0.0, 0.0], [1.0, 2.0])


def test_mbe_empty_sequences_raises():
    with pytest.raises(ValueError):
        compute_mbe([], [])


def test_mbe_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_mbe([1.0, 2.0, 3.0], [1.0, 2.0])


# ── SimulationLog.record / to_dicts ──────────────────────────────────────────
# Verifies that record() appends accessible entries with the expected fields.

def test_log_record_adds_entry():
    """record() appends one entry accessible through to_dicts()."""
    log = SimulationLog(stratum="O1", seed=0)
    assert len(log) == 0
    log.record(
        datetime(2024, 7, 15, 9, 0),
        action=_make_action(),
        env=_make_env(),
        activity_category="sleeping",
    )
    assert len(log) == 1
    assert len(log.to_dicts()) == 1


def test_log_to_dicts_required_fields():
    """to_dicts() entries contain timestep, action_type, activity_category, thermostat_setpoint_c."""
    log = _make_log(n=1)
    row = log.to_dicts()[0]
    assert "timestep" in row
    assert "action_type" in row
    assert "activity_category" in row
    assert "thermostat_setpoint_c" in row


def test_log_to_dicts_field_values():
    """Field values round-trip correctly through to_dicts()."""
    log = SimulationLog(stratum="O2", seed=7)
    ts = datetime(2024, 8, 1, 14, 30)
    log.record(
        ts,
        action=AgentAction(action_type="toggle_device", target_id="tv", value=True),
        env=_make_env(thermostat_setpoint_c=23.0),
        atus_code="020201",
        activity_category="work",
        occupancy="home",
    )
    row = log.to_dicts()[0]
    assert row["timestep"] == ts.isoformat()
    assert row["action_type"] == "toggle_device"
    assert row["activity_category"] == "work"
    assert row["thermostat_setpoint_c"] == pytest.approx(23.0)


def test_log_len_tracks_record_count():
    """len(log) equals the number of record() calls made."""
    log = _make_log(n=5)
    assert len(log) == 5


# ── SimulationLog — empty log behaviour ──────────────────────────────────────

def test_log_empty_to_dicts_returns_empty_list():
    """to_dicts() on a freshly created log returns []."""
    log = SimulationLog(stratum="O1", seed=0)
    assert log.to_dicts() == []


def test_log_empty_len_is_zero():
    """len() of a new log is 0."""
    assert len(SimulationLog(stratum="O1", seed=0)) == 0


# ── SimulationLog.to_csv ──────────────────────────────────────────────────────
# Verifies CSV export: roundtrip with DictReader, and empty-log behaviour.

def test_log_to_csv_roundtrip():
    """to_csv() writes a file that csv.DictReader can parse with correct values."""
    log = _make_log(n=2)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    log.to_csv(path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["action_type"] == "do_nothing"
    assert rows[0]["activity_category"] == "sleeping"


def test_log_to_csv_empty_log_writes_file():
    """to_csv() on an empty log writes a file (pandas produces an empty CSV)."""
    log = SimulationLog(stratum="O1", seed=0)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    log.to_csv(path)
    assert Path(path).exists()
    # Empty DataFrame → no data rows
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


# ── SimulationLog.to_json ─────────────────────────────────────────────────────
# Verifies JSON export roundtrip: valid JSON, correct metadata, records preserved.

def test_log_to_json_roundtrip():
    """to_json() writes valid JSON that round-trips all records and metadata."""
    log = _make_log(n=3)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    log.to_json(path)

    with open(path) as f:
        data = json.load(f)

    assert data["stratum"] == "O1"
    assert data["seed"] == 42
    assert data["n_steps"] == 3
    assert len(data["records"]) == 3
    assert data["records"][0]["action_type"] == "do_nothing"


def test_log_to_json_includes_run_id():
    """to_json() payload contains the run_id key."""
    log = SimulationLog(stratum="O1", seed=99)
    log.record(
        datetime(2024, 7, 15, 9, 0),
        action=_make_action(),
        env=_make_env(),
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    log.to_json(path)

    with open(path) as f:
        data = json.load(f)

    assert "run_id" in data
