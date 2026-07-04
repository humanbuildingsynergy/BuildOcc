"""
REST API endpoint tests via FastAPI TestClient.

All LLM calls are patched — no API key or network access required.
Covers: initialize, step, signal, state, list, delete, error paths.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from occupant_agent.api.app import app

# ── Shared fixtures ───────────────────────────────────────────────────────────

_STEP_RESPONSE = {
    "action_type": "do_nothing",
    "target_id": None,
    "value": None,
    "reasoning": "No action needed.",
    "_memory_note": "Uneventful step.",
    "_importance": 2,
}

_SIGNAL_RESPONSE = {
    "response": "accepted",
    "reasoning": "Complying with the request.",
    "_importance": 5,
}

_ENV = {
    "timestep": "2025-08-12T10:00:00Z",
    "zone_temp_c": 24.5,
    "outdoor_temp_c": 32.0,
    "tou_rate": 0.12,
    "thermostat_setpoint_c": 22.0,
    "devices": [{"device_id": "hvac", "state": True, "power_w": 3500}],
    "rooms": [{"room_id": "living_room", "occupied": True}],
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def agent_id(client):
    r = client.post("/agents/initialize", json={"stratum": "O1", "seed": 42})
    assert r.status_code == 200
    aid = r.json()["agent_id"]
    yield aid
    client.delete(f"/agents/{aid}")


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── /agents/initialize ────────────────────────────────────────────────────────

def test_initialize_returns_agent_id(client):
    r = client.post("/agents/initialize", json={"stratum": "O1", "seed": 0})
    assert r.status_code == 200
    data = r.json()
    assert "agent_id" in data
    assert data["stratum"] == "O1"
    client.delete(f"/agents/{data['agent_id']}")


@pytest.mark.parametrize("stratum", ["O1", "O2", "O3", "O4"])
def test_initialize_all_strata(client, stratum):
    r = client.post("/agents/initialize", json={"stratum": stratum, "seed": 7})
    assert r.status_code == 200, f"Initialize failed for {stratum}: {r.text}"
    client.delete(f"/agents/{r.json()['agent_id']}")


def test_initialize_invalid_stratum(client):
    r = client.post("/agents/initialize", json={"stratum": "INVALID", "seed": 0})
    assert r.status_code in (400, 422, 500)


# ── /agents/{id}/step ─────────────────────────────────────────────────────────

def test_step_returns_action(client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_STEP_RESPONSE):
        r = client.post(f"/agents/{agent_id}/step", json={
            "environment": _ENV,
            "atus_code": "030101",
            "wfh_today": False,
        })
    assert r.status_code == 200
    action = r.json()
    assert action["action_type"] in {
        "do_nothing", "adjust_thermostat", "toggle_device", "move_room"
    }


def test_step_increments_action_count(client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_STEP_RESPONSE):
        client.post(f"/agents/{agent_id}/step", json={"environment": _ENV})
        client.post(f"/agents/{agent_id}/step", json={"environment": _ENV})

    r = client.get(f"/agents/{agent_id}/state")
    assert r.json()["action_count"] >= 2


def test_step_missing_environment_422(client, agent_id):
    r = client.post(f"/agents/{agent_id}/step", json={"atus_code": "030101"})
    assert r.status_code == 422


def test_step_unknown_agent_404(client):
    r = client.post("/agents/does-not-exist/step", json={"environment": _ENV})
    assert r.status_code == 404


def test_step_llm_error_returns_500(client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm",
               side_effect=RuntimeError("API key missing")):
        r = client.post(f"/agents/{agent_id}/step", json={"environment": _ENV})
    assert r.status_code == 500
    assert "API key missing" in r.json()["detail"]


# ── /agents/{id}/signal ───────────────────────────────────────────────────────

@pytest.mark.parametrize("sig_type", ["A", "B", "C"])
def test_signal_all_types(client, agent_id, sig_type):
    with patch("occupant_agent.agent.occupant.call_llm", return_value=_SIGNAL_RESPONSE):
        r = client.post(f"/agents/{agent_id}/signal", json={
            "signal_type": sig_type,
            "content": "Please reduce HVAC use during peak hours.",
            "environment": _ENV,
        })
    assert r.status_code == 200
    assert r.json()["response"] in {"accepted", "rejected", "deferred"}


def test_signal_llm_error_returns_500(client, agent_id):
    with patch("occupant_agent.agent.occupant.call_llm",
               side_effect=RuntimeError("Rate limit")):
        r = client.post(f"/agents/{agent_id}/signal", json={
            "signal_type": "A",
            "content": "Turn off appliances.",
            "environment": _ENV,
        })
    assert r.status_code == 500
    assert "Rate limit" in r.json()["detail"]


# ── /agents/{id}/state ────────────────────────────────────────────────────────

def test_get_state_fields(client, agent_id):
    r = client.get(f"/agents/{agent_id}/state")
    assert r.status_code == 200
    state = r.json()
    for field in ("agent_id", "stratum", "memory_count", "action_count"):
        assert field in state, f"Missing field {field!r} in state response"
    assert state["stratum"] == "O1"


def test_get_state_unknown_agent_404(client):
    r = client.get("/agents/nonexistent/state")
    assert r.status_code == 404


# ── /agents/ ──────────────────────────────────────────────────────────────────

def test_list_agents_contains_created(client, agent_id):
    r = client.get("/agents/")
    assert r.status_code == 200
    ids = [a["agent_id"] for a in r.json()]
    assert agent_id in ids


# ── /agents/{id} DELETE ───────────────────────────────────────────────────────

def test_delete_then_404(client):
    r = client.post("/agents/initialize", json={"stratum": "O2", "seed": 1})
    aid = r.json()["agent_id"]

    r = client.delete(f"/agents/{aid}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = client.get(f"/agents/{aid}/state")
    assert r.status_code == 404


def test_delete_unknown_agent_404(client):
    r = client.delete("/agents/nonexistent-uuid")
    assert r.status_code == 404
