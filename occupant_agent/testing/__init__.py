"""
occupant_agent.testing — test utilities for OccupantAgent extensions.

This module is for use in test suites only. It is NOT imported by the
main occupant_agent package at runtime, so it adds zero overhead to
production code.

Quick reference
───────────────
MockLLMAgent            Drop-in OccupantAgent with no LLM calls
make_env()              Minimal EnvironmentState factory
make_peak_env()         EnvironmentState preset: DR scenario (high TOU, hot)
make_offpeak_env()      EnvironmentState preset: baseline (low TOU, mild)
make_persona()          Persona factory with per-field overrides
make_memory_stream()    Pre-populated MemoryStream for retrieval tests
make_device()           DeviceState shorthand
make_room()             RoomState shorthand
assert_persona_contract()   Verify a BasePersona subclass meets the contract
assert_scheduler_contract() Verify a BaseScheduler subclass meets the contract
"""

from occupant_agent.testing.conformance import (
    assert_persona_contract,
    assert_scheduler_contract,
)
from occupant_agent.testing.fixtures import (
    make_device,
    make_env,
    make_memory_stream,
    make_offpeak_env,
    make_peak_env,
    make_persona,
    make_room,
)
from occupant_agent.testing.mock_llm import MockLLMAgent

__all__ = [
    "MockLLMAgent",
    "make_env",
    "make_peak_env",
    "make_offpeak_env",
    "make_persona",
    "make_memory_stream",
    "make_device",
    "make_room",
    "assert_persona_contract",
    "assert_scheduler_contract",
]
