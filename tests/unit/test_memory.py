from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from occupant_agent.agent.memory import MemoryEntry, MemoryStream

BASE_TIME = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)


# ── MemoryEntry round-trip ────────────────────────────────────────────────────

def test_entry_round_trip():
    entry = MemoryEntry(
        memory_id="abc-123",
        content="Turned off the AC during a heat wave.",
        sim_time=BASE_TIME,
        importance=8.0,
        memory_type="observation",
        access_count=3,
        last_accessed=BASE_TIME + timedelta(hours=1),
    )
    restored = MemoryEntry.from_dict(entry.to_dict())

    assert restored.memory_id == entry.memory_id
    assert restored.content == entry.content
    assert restored.sim_time == entry.sim_time
    assert restored.importance == entry.importance
    assert restored.memory_type == entry.memory_type
    assert restored.access_count == entry.access_count
    assert restored.last_accessed == entry.last_accessed


# ── add() ─────────────────────────────────────────────────────────────────────

def test_add_increments_accumulator():
    stream = MemoryStream()
    stream.add("First memory.", importance=7, memory_type="observation", sim_time=BASE_TIME)
    assert stream.importance_accumulator == pytest.approx(7.0)

    stream.add("Second memory.", importance=3, memory_type="observation", sim_time=BASE_TIME)
    assert stream.importance_accumulator == pytest.approx(10.0)


def test_add_clamps_importance():
    stream = MemoryStream()
    e_high = stream.add("Too important.", importance=15, memory_type="observation", sim_time=BASE_TIME)
    assert e_high.importance == pytest.approx(10.0)

    e_low = stream.add("Negative importance.", importance=-2, memory_type="observation", sim_time=BASE_TIME)
    assert e_low.importance == pytest.approx(0.0)


# ── retrieve() ────────────────────────────────────────────────────────────────

def test_retrieve_returns_top_k():
    stream = MemoryStream()
    for i in range(10):
        stream.add(
            f"Memory {i}",
            importance=float(i),
            memory_type="observation",
            sim_time=BASE_TIME + timedelta(minutes=i * 15),
        )
    results = stream.retrieve(query_time=BASE_TIME + timedelta(hours=2), k=3)
    assert len(results) == 3


def test_retrieve_score_prefers_high_importance():
    stream = MemoryStream()

    # High-importance entry created long ago
    stream.add(
        "Critical event long ago.",
        importance=9,
        memory_type="observation",
        sim_time=BASE_TIME,
    )
    # Low-importance entry created much more recently
    recent_time = BASE_TIME + timedelta(hours=1)
    stream.add(
        "Trivial recent event.",
        importance=1,
        memory_type="observation",
        sim_time=recent_time,
    )

    # Query far in the future (~5 days later) so recency of both entries is near 0
    # importance term will dominate → high-importance entry should win
    query_time = BASE_TIME + timedelta(days=5)
    results = stream.retrieve(query_time=query_time, k=1)

    assert len(results) == 1
    assert results[0].content == "Critical event long ago."


def test_retrieve_updates_access_count():
    stream = MemoryStream()
    stream.add("Something happened.", importance=5, memory_type="observation", sim_time=BASE_TIME)

    query_time = BASE_TIME + timedelta(minutes=30)
    results = stream.retrieve(query_time=query_time, k=1)

    assert len(results) == 1
    assert results[0].access_count == 1
    assert results[0].last_accessed == query_time


# ── should_reflect() ──────────────────────────────────────────────────────────

def test_should_reflect_threshold():
    stream = MemoryStream(reflection_threshold=10.0)

    # Add entries summing to 9 — should NOT trigger
    stream.add("Memory A.", importance=5, memory_type="observation", sim_time=BASE_TIME)
    stream.add("Memory B.", importance=4, memory_type="observation", sim_time=BASE_TIME)
    assert stream.should_reflect() is False

    # Push accumulator over threshold
    stream.add("Memory C.", importance=1, memory_type="observation", sim_time=BASE_TIME)
    assert stream.should_reflect() is True


# ── reflect() ─────────────────────────────────────────────────────────────────

def test_reflect_resets_accumulator():
    stream = MemoryStream(reflection_threshold=10.0)
    stream.add("Memory.", importance=10, memory_type="observation", sim_time=BASE_TIME)
    assert stream.should_reflect() is True

    def mock_llm(system, user):
        return {"insights": ["Insight A.", "Insight B.", "Insight C."]}
    stream.reflect(call_llm_fn=mock_llm, sim_time=BASE_TIME + timedelta(hours=1))

    assert stream.importance_accumulator == pytest.approx(0.0)


def test_reflect_adds_entries():
    stream = MemoryStream()
    stream.add("Background memory.", importance=5, memory_type="observation", sim_time=BASE_TIME)

    reflect_time = BASE_TIME + timedelta(hours=1)
    def mock_llm(system, user):
        return {"insights": ["Insight A.", "Insight B.", "Insight C."]}
    new_entries = stream.reflect(call_llm_fn=mock_llm, sim_time=reflect_time)

    assert len(new_entries) == 3
    for entry in new_entries:
        assert entry.memory_type == "reflection"
        assert entry.importance == pytest.approx(9.0)
        assert entry.sim_time == reflect_time


def test_reflect_empty_insights_resets_accumulator():
    """Empty insights list must still reset accumulator — else reflect fires every step."""
    stream = MemoryStream(reflection_threshold=10.0)
    stream.add("Memory.", importance=10, memory_type="observation", sim_time=BASE_TIME)
    assert stream.should_reflect() is True

    def mock_llm_empty(system, user):
        return {"insights": []}
    stream.reflect(call_llm_fn=mock_llm_empty, sim_time=BASE_TIME + timedelta(hours=1))

    assert stream.importance_accumulator == pytest.approx(0.0)
    assert stream.should_reflect() is False


def test_reflect_string_insights_stored_as_empty():
    """String insights (LLM hallucination) must not produce character-level entries."""
    stream = MemoryStream()
    stream.add("Memory.", importance=5, memory_type="observation", sim_time=BASE_TIME)

    def mock_llm_string(system, user):
        return {"insights": "I should reduce peak-hour usage."}
    entries_before = len(stream.entries)
    stream.reflect(call_llm_fn=mock_llm_string, sim_time=BASE_TIME + timedelta(hours=1))

    # No reflection entries should have been added from iterating a string
    reflections = [e for e in stream.entries if e.memory_type == "reflection"]
    assert len(reflections) == 0
    assert len(stream.entries) == entries_before


# ── from_dicts() / to_dicts() ─────────────────────────────────────────────────

def test_from_dicts_restores_accumulator():
    stream = MemoryStream()
    stream.add("Memory A.", importance=6, memory_type="observation", sim_time=BASE_TIME)
    stream.add("Memory B.", importance=4, memory_type="observation", sim_time=BASE_TIME)

    records = stream.to_dicts()
    restored = MemoryStream.from_dicts(records, importance_accumulator=42.0)

    assert restored.importance_accumulator == pytest.approx(42.0)
    assert len(restored.entries) == 2


def test_from_dicts_restores_last_reflected_at():
    stream = MemoryStream()
    stream.add("Observation.", importance=5, memory_type="observation", sim_time=BASE_TIME)

    reflect_time = BASE_TIME + timedelta(hours=2)
    def mock_llm(system, user):
        return {"insights": ["Insight 1.", "Insight 2.", "Insight 3."]}
    stream.reflect(call_llm_fn=mock_llm, sim_time=reflect_time)

    records = stream.to_dicts()
    restored = MemoryStream.from_dicts(records)

    assert restored.last_reflected_at == reflect_time
