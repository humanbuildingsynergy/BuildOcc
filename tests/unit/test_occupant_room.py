"""
Unit tests for activity-driven room movement in OccupantAgent.

Covers:
  - _suggest_room(): correct (room_id, is_deterministic) for each ATUS category
  - Auto-move path in step(): deterministic activities bypass the LLM
  - Soft-hint path: non-deterministic activities inject room context into the prompt
  - _build_room_hint(): "already there" vs "not there" variants
  - WFH special case: work + work_from_home → home_office
  - Room-not-in-env: graceful (None, False) return
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.environment.state import RoomState
from occupant_agent.testing.fixtures import make_env, make_persona


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent(stratum="O1", **persona_overrides) -> OccupantAgent:
    persona = make_persona(stratum=stratum, seed=0, **persona_overrides)
    return OccupantAgent(persona=persona)


def _rooms(*room_ids: str, occupied: str | None = None) -> list[RoomState]:
    """Build a RoomState list; `occupied` specifies which room_id is occupied."""
    return [RoomState(room_id=r, occupied=(r == occupied)) for r in room_ids]


# ── _suggest_room(): deterministic categories ─────────────────────────────────

class TestSuggestRoomDeterministic:
    def test_sleeping_maps_to_bedroom(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))
        room, is_det = agent._suggest_room("010101", env)  # Sleeping
        assert room == "bedroom"
        assert is_det is True

    def test_sleeping_prefers_master_bedroom_over_bedroom(self):
        agent = _make_agent("O2")
        env = make_env(rooms=_rooms("living_room", "master_bedroom", "kitchen", occupied="living_room"))
        room, is_det = agent._suggest_room("010101", env)
        assert room == "master_bedroom"
        assert is_det is True

    def test_sleeping_falls_back_to_bedroom_when_no_master(self):
        agent = _make_agent("O1")
        env = make_env(rooms=_rooms("living_room", "bedroom", occupied="living_room"))
        room, is_det = agent._suggest_room("010101", env)
        assert room == "bedroom"
        assert is_det is True

    def test_laundry_maps_to_laundry_room(self):
        agent = _make_agent("O3")
        env = make_env(rooms=_rooms("living_room", "kitchen", "laundry_room", occupied="living_room"))
        room, is_det = agent._suggest_room("020202", env)  # Laundry
        assert room == "laundry_room"
        assert is_det is True

    def test_laundry_returns_none_when_no_laundry_room(self):
        agent = _make_agent("O1")
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))
        room, is_det = agent._suggest_room("020202", env)
        assert room is None
        assert is_det is False


# ── _suggest_room(): soft categories ──────────────────────────────────────────

class TestSuggestRoomSoft:
    def test_food_prep_maps_to_kitchen(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))
        room, is_det = agent._suggest_room("020101", env)  # Food and drink preparation
        assert room == "kitchen"
        assert is_det is False

    def test_eating_prefers_dining_room_over_kitchen(self):
        agent = _make_agent("O2")
        env = make_env(rooms=_rooms("living_room", "kitchen", "dining_room", occupied="living_room"))
        room, is_det = agent._suggest_room("110101", env)  # Eating and drinking
        assert room == "dining_room"
        assert is_det is False

    def test_eating_falls_back_to_kitchen_when_no_dining_room(self):
        agent = _make_agent("O1")
        env = make_env(rooms=_rooms("living_room", "kitchen", "bedroom", occupied="living_room"))
        room, is_det = agent._suggest_room("110101", env)
        assert room == "kitchen"
        assert is_det is False

    def test_tv_maps_to_living_room(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="bedroom"))
        room, is_det = agent._suggest_room("120301", env)  # Watching TV
        assert room == "living_room"
        assert is_det is False

    def test_exercise_maps_to_living_room_when_home(self):
        agent = _make_agent(home_gym=True)
        env = make_env(rooms=_rooms("living_room", "bedroom", occupied="bedroom"))
        room, is_det = agent._suggest_room("130120", env)  # Cardiovascular equipment
        assert room == "living_room"
        assert is_det is False


# ── _suggest_room(): WFH special case ────────────────────────────────────────

class TestSuggestRoomWFH:
    def test_work_wfh_maps_to_home_office(self):
        agent = _make_agent(work_from_home=True)
        env = make_env(rooms=_rooms("living_room", "home_office", "bedroom", occupied="living_room"))
        room, is_det = agent._suggest_room("050101", env)  # Work, main job
        assert room == "home_office"
        assert is_det is False

    def test_work_no_wfh_returns_none(self):
        agent = _make_agent(work_from_home=False)
        env = make_env(rooms=_rooms("living_room", "home_office", "bedroom", occupied="living_room"))
        room, is_det = agent._suggest_room("050101", env)
        assert room is None
        assert is_det is False

    def test_work_wfh_but_no_home_office_returns_none(self):
        agent = _make_agent(work_from_home=True)
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))
        room, is_det = agent._suggest_room("050101", env)
        assert room is None
        assert is_det is False


# ── _suggest_room(): edge cases ───────────────────────────────────────────────

class TestSuggestRoomEdgeCases:
    def test_none_atus_code_returns_none(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", occupied="living_room"))
        room, is_det = agent._suggest_room(None, env)
        assert room is None
        assert is_det is False

    def test_empty_room_list_returns_none(self):
        agent = _make_agent()
        env = make_env(rooms=[])
        room, is_det = agent._suggest_room("010101", env)
        assert room is None
        assert is_det is False

    def test_other_category_returns_none(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))
        room, is_det = agent._suggest_room("090101", env)  # Religion / Other
        assert room is None
        assert is_det is False


# ── _build_room_hint() ────────────────────────────────────────────────────────

class TestBuildRoomHint:
    def test_not_there_hint_contains_target_room(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "kitchen", occupied="living_room"))
        hint = agent._build_room_hint("kitchen", "Food and drink preparation", env)
        assert "kitchen" in hint
        assert 'target_id="kitchen"' in hint

    def test_already_there_hint_is_confirmatory(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "kitchen", occupied="kitchen"))
        hint = agent._build_room_hint("kitchen", "Food and drink preparation", env)
        assert "as expected" in hint
        assert "move_room" not in hint

    def test_room_name_underscores_replaced(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "master_bedroom", occupied="living_room"))
        hint = agent._build_room_hint("master_bedroom", "Sleeping", env)
        assert "master bedroom" in hint


# ── Auto-move path in step() ──────────────────────────────────────────────────

class TestAutoMove:
    def test_sleeping_triggers_auto_move_to_bedroom(self):
        """Sleeping in wrong room → move_room returned without calling the LLM."""
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))

        with patch.object(agent, "_call") as mock_call:
            action = agent.step(env, atus_code="010101")

        assert action.action_type == "move_room"
        assert action.target_id == "bedroom"
        mock_call.assert_not_called()

    def test_sleeping_already_in_bedroom_calls_llm(self):
        """Agent already in bedroom → auto-move skipped; LLM called normally."""
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="bedroom"))

        fake_response = {
            "action_type": "do_nothing",
            "target_id": None,
            "value": None,
            "reasoning": "Already comfortable.",
            "_memory_note": "Resting in bedroom.",
            "_importance": 2,
        }
        with patch.object(agent, "_call", return_value=fake_response):
            action = agent.step(env, atus_code="010101")

        assert action.action_type == "do_nothing"

    def test_laundry_triggers_auto_move_to_laundry_room(self):
        agent = _make_agent("O3")
        env = make_env(rooms=_rooms("living_room", "kitchen", "laundry_room", occupied="living_room"))

        with patch.object(agent, "_call") as mock_call:
            action = agent.step(env, atus_code="020202")

        assert action.action_type == "move_room"
        assert action.target_id == "laundry_room"
        mock_call.assert_not_called()

    def test_auto_move_increments_action_count(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", occupied="living_room"))
        before = agent._action_count

        with patch.object(agent, "_call"):
            agent.step(env, atus_code="010101")

        assert agent._action_count == before + 1

    def test_auto_move_adds_memory_entry(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", occupied="living_room"))

        with patch.object(agent, "_call"):
            agent.step(env, atus_code="010101")

        assert len(agent.memory.entries) == 1
        assert "bedroom" in agent.memory.entries[0].content


# ── Soft-hint path in step() ──────────────────────────────────────────────────

class TestSoftHint:
    def _fake_llm_response(self):
        return {
            "action_type": "move_room",
            "target_id": "kitchen",
            "value": None,
            "reasoning": "Moving to kitchen to cook.",
            "_memory_note": "Moved to kitchen.",
            "_importance": 3,
        }

    def test_food_prep_injects_room_hint_into_prompt(self):
        """Food preparation → room hint mentioning 'kitchen' appears in user prompt."""
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", "kitchen", occupied="living_room"))

        captured_user_prompt = []

        def fake_call(system, user):
            captured_user_prompt.append(user)
            return self._fake_llm_response()

        with patch.object(agent, "_call", side_effect=fake_call):
            agent.step(env, atus_code="020101")

        assert captured_user_prompt, "LLM should have been called"
        assert "kitchen" in captured_user_prompt[0]
        assert "move_room" in captured_user_prompt[0]

    def test_tv_hint_contains_living_room(self):
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "bedroom", occupied="bedroom"))

        captured = []

        def fake_call(system, user):
            captured.append(user)
            return {
                "action_type": "move_room", "target_id": "living_room",
                "value": None, "reasoning": "Going to watch TV.",
                "_memory_note": "Moved to living room.", "_importance": 3,
            }

        with patch.object(agent, "_call", side_effect=fake_call):
            agent.step(env, atus_code="120301")

        assert "living room" in captured[0]

    def test_already_in_correct_room_hint_is_confirmatory(self):
        """Agent in kitchen for food_prep → confirmatory hint, no move_room directive."""
        agent = _make_agent()
        env = make_env(rooms=_rooms("living_room", "kitchen", occupied="kitchen"))

        captured = []

        def fake_call(system, user):
            captured.append(user)
            return {
                "action_type": "do_nothing", "target_id": None,
                "value": None, "reasoning": "Already in kitchen.",
                "_memory_note": "Cooking.", "_importance": 2,
            }

        with patch.object(agent, "_call", side_effect=fake_call):
            agent.step(env, atus_code="020101")

        assert "as expected" in captured[0]
