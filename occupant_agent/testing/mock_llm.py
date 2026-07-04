"""
MockLLMAgent — a drop-in OccupantAgent that never makes network calls.

Overrides _call() and _call_reflect() to return canned dicts, so that
step(), receive_signal(), and reflect() all work in unit tests without
an API key, network access, or latency.

Usage
─────
    from occupant_agent.testing import MockLLMAgent

    agent = MockLLMAgent.from_stratum("O1", seed=0)
    action = agent.step(env)
    assert action.action_type in {"do_nothing", "toggle_device",
                                   "adjust_thermostat", "move_room"}

    response = agent.receive_signal("B", "Save energy now.", env)
    assert response.response in {"accepted", "rejected", "deferred"}

Customising responses
─────────────────────
Pass a response_map to control what step() returns for specific
action types or hours:

    agent = MockLLMAgent.from_stratum("O1", seed=0,
        step_response={"action_type": "toggle_device",
                       "target_id": "hvac",
                       "value": False,
                       "reasoning": "Saving energy during peak."},
        signal_response={"response": "accepted",
                         "reasoning": "Makes sense."},
    )
"""

from __future__ import annotations

from typing import Any

from occupant_agent.agent.occupant import OccupantAgent

_DEFAULT_STEP_RESPONSE: dict[str, Any] = {
    "action_type": "do_nothing",
    "target_id": None,
    "value": None,
    "reasoning": "No action needed right now.",
    "_memory_note": "Uneventful timestep during testing.",
    "_importance": 2,
}

_DEFAULT_SIGNAL_RESPONSE: dict[str, Any] = {
    "response": "accepted",
    "reasoning": "The suggestion seems reasonable.",
    "_importance": 5,
}

_DEFAULT_REFLECT_RESPONSE: dict[str, Any] = {
    "insights": [
        "I tend to use more energy in the evenings.",
        "I respond well to cost-savings signals.",
        "My thermostat habits are fairly consistent.",
    ]
}


class MockLLMAgent(OccupantAgent):
    """
    OccupantAgent subclass that stubs out all LLM calls.

    Identical public API to OccupantAgent — use everywhere a real agent
    is used in tests. All methods that would call an LLM return
    configurable canned responses instead.
    """

    def __init__(
        self,
        *args,
        step_response: dict[str, Any] | None = None,
        signal_response: dict[str, Any] | None = None,
        reflect_response: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._step_response = {**_DEFAULT_STEP_RESPONSE, **(step_response or {})}
        self._signal_response = {**_DEFAULT_SIGNAL_RESPONSE, **(signal_response or {})}
        self._reflect_response = {**_DEFAULT_REFLECT_RESPONSE, **(reflect_response or {})}

    # ── Stub LLM calls ────────────────────────────────────────────────────────

    def _call(self, system: str, user: str, max_tokens: int = 512) -> dict[str, Any]:
        # Return step or signal response depending on which prompt was built.
        # Signal prompts contain "How do you respond?" — a reliable discriminator.
        if "How do you respond?" in user:
            return dict(self._signal_response)
        return dict(self._step_response)

    def _call_reflect(self, system: str, user: str) -> dict[str, Any]:
        return dict(self._reflect_response)

    @classmethod
    def from_stratum(  # type: ignore[override]
        cls,
        stratum: str,
        seed: int | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        state_fips: int = 48,
        scheduler=None,
        step_response: dict | None = None,
        signal_response: dict | None = None,
        reflect_response: dict | None = None,
    ) -> MockLLMAgent:
        """Like OccupantAgent.from_stratum() but accepts mock response overrides."""
        from occupant_agent.agent.persona import create_persona
        from occupant_agent.core.registry import get_stratum

        try:
            factory = get_stratum(stratum)
            persona = factory(seed=seed, state_fips=state_fips)
        except (KeyError, TypeError):
            persona = create_persona(stratum, seed=seed, state_fips=state_fips)  # type: ignore[arg-type]

        sched = None
        if isinstance(scheduler, str):
            from occupant_agent.core.registry import get_scheduler
            sched_cls = get_scheduler(scheduler)
            try:
                sched = sched_cls(stratum=stratum, seed=seed)
            except TypeError:
                sched = sched_cls()
        elif scheduler is not None:
            sched = scheduler

        return cls(
            persona=persona,
            scheduler=sched,
            llm_provider=llm_provider,
            llm_model=llm_model,
            step_response=step_response,
            signal_response=signal_response,
            reflect_response=reflect_response,
        )
