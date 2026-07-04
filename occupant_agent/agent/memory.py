"""
Memory architecture for the LLM occupant agent.

Implements the Park et al. (2023) generative agents memory model:
  - MemoryEntry: a single timestamped, importance-rated memory
  - MemoryStream: collection with retrieval scoring and reflection trigger

Retrieval score (Phase 1 — no embeddings):
  score = 0.5 * recency + 0.5 * (importance / 10)

Recency: exponential decay with 24-hour half-life.
  recency(t) = 2 ** (-(now - entry.sim_time) / timedelta(hours=24))

Importance (0–10): set by the LLM at the time of observation, as part of the
step() or receive_signal() response. High scores = memorable events
(turned off AC during heat wave = 8, watched TV as usual = 2).

Reflection: triggered when cumulative importance of entries since last
reflection exceeds `reflection_threshold` (default 100). The LLM synthesizes
3 insights from the last 30 memories; each insight becomes a new entry with
memory_type="reflection" and importance=9.

Phase 2 extension: add sentence embeddings (sentence-transformers or
OpenAI ada-002) and include cosine similarity term in retrieval score.

Reference: Park, J. S. et al. (2023). Generative agents: Interactive simulacra
of human behavior. UIST 2023. https://arxiv.org/abs/2304.03442
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from occupant_agent.core.base_memory import BaseMemoryStream

# ── MemoryEntry ───────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    A single memory in the agent's stream.

    Fields:
      memory_id:    UUID4 string, assigned at creation.
      content:      Natural-language description (1–2 sentences).
      sim_time:     Simulation time of the event (not wall clock).
      importance:   0–10 float, rated by the LLM when the memory is created.
      memory_type:  Semantic category (affects prompt formatting).
      access_count: How many times retrieve() has returned this entry.
      last_accessed: Simulation time of the most recent retrieval.
    """

    memory_id: str
    content: str
    sim_time: datetime
    importance: float             # 0–10
    memory_type: Literal["observation", "reflection", "signal_received"]
    access_count: int = 0
    last_accessed: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "sim_time": self.sim_time.isoformat(),
            "importance": self.importance,
            "memory_type": self.memory_type,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MemoryEntry:
        def _parse_dt(s: str) -> datetime:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

        return cls(
            memory_id=d["memory_id"],
            content=d["content"],
            sim_time=_parse_dt(d["sim_time"]),
            importance=float(d["importance"]),
            memory_type=d["memory_type"],
            access_count=int(d.get("access_count", 0)),
            last_accessed=(
                _parse_dt(d["last_accessed"])
                if d.get("last_accessed")
                else None
            ),
        )


# ── MemoryStream ──────────────────────────────────────────────────────────────

_RECENCY_HALF_LIFE = timedelta(hours=24)


class MemoryStream(BaseMemoryStream):
    """
    Ordered list of MemoryEntry objects with Park et al. (2023) retrieval.

    The stream grows indefinitely during a simulation run; retrieval keeps
    context manageable by surfacing only the most relevant entries.
    """

    def __init__(self, reflection_threshold: float = 100.0) -> None:
        self._entries: list[MemoryEntry] = []
        self._reflection_threshold = reflection_threshold
        self._importance_accumulator: float = 0.0
        self._last_reflected_at: datetime | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        importance: float,
        memory_type: Literal["observation", "reflection", "signal_received"],
        sim_time: datetime,
    ) -> MemoryEntry:
        """
        Add a new memory and return it. Increments the importance accumulator
        (which drives the reflection trigger).
        """
        importance = max(0.0, min(10.0, float(importance)))
        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=content,
            sim_time=sim_time,
            importance=importance,
            memory_type=memory_type,
        )
        self._entries.append(entry)
        self._importance_accumulator += importance
        return entry

    def retrieve(self, query_time: datetime, k: int = 5) -> list[MemoryEntry]:
        """
        Return the top-k entries ranked by retrieval score.

        score = 0.5 * recency + 0.5 * (importance / 10)

        Updates access_count and last_accessed for returned entries so the
        next retrieval can factor in access recency (Phase 2 extension point).
        """
        if not self._entries:
            return []

        scored = [
            (self._score(e, query_time), e)
            for e in self._entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[:k]]

        for e in top:
            e.access_count += 1
            e.last_accessed = query_time

        return top

    def count(self) -> int:
        return len(self._entries)

    def should_reflect(self) -> bool:
        """True when enough importance has accumulated to trigger a reflection."""
        return self._importance_accumulator >= self._reflection_threshold

    def reflect(
        self,
        call_llm_fn: Callable[[str, str], dict],
        sim_time: datetime,
    ) -> list[MemoryEntry]:
        """
        Synthesize 3 insights from the last 30 memories via LLM.

        call_llm_fn must accept (system: str, user: str) → dict.
        Returns the 3 new reflection MemoryEntry objects (already added to stream).
        Resets the importance accumulator.
        """
        recent = sorted(self._entries, key=lambda e: e.sim_time)[-30:]
        memory_text = "\n".join(
            f"{i+1}. [{e.sim_time.strftime('%H:%M')}] {e.content}"
            for i, e in enumerate(recent)
        )

        system = (
            "You are reflecting on your recent experiences at home. "
            "Write exactly 3 concise insights (1 sentence each) about your energy use "
            "patterns, habits, and how you respond to energy situations. "
            "These insights will guide your future decisions. "
            'Respond with JSON only: {"insights": ["...", "...", "..."]}'
        )
        user = f"Your recent memories:\n{memory_text}"

        result = call_llm_fn(system, user)
        insights = result.get("insights", [])
        # Guard against LLMs that return a string instead of a list — iterating a
        # string would store single characters as high-importance reflection entries.
        if not isinstance(insights, list):
            insights = []

        new_entries = []
        for insight in insights[:3]:
            entry = self.add(
                content=str(insight),
                importance=9.0,
                memory_type="reflection",
                sim_time=sim_time,
            )
            new_entries.append(entry)

        # Always reset the accumulator so an empty-insights response (e.g., model
        # refusal) does not cause reflect() to re-fire on every subsequent step.
        self._importance_accumulator = 0.0
        self._last_reflected_at = sim_time
        return new_entries

    @property
    def entries(self) -> list[MemoryEntry]:
        """All entries in chronological order (read-only view)."""
        return list(self._entries)

    @property
    def last_reflected_at(self) -> datetime | None:
        return self._last_reflected_at

    @property
    def importance_accumulator(self) -> float:
        """Current accumulator value — must be persisted and restored across save/load cycles."""
        return self._importance_accumulator

    def to_dicts(self) -> list[dict]:
        """Serialize all entries for SQLite persistence."""
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_dicts(
        cls,
        records: list[dict],
        importance_accumulator: float = 0.0,
        reflection_threshold: float = 100.0,
        **kwargs: Any,
    ) -> MemoryStream:
        """
        Restore a MemoryStream from serialized records.

        importance_accumulator must be passed from the persisted value
        (AgentRecord.memory_accumulator) so the reflection trigger fires
        correctly across stateless REST API load/save cycles.
        last_reflected_at is reconstructed from the most recent reflection entry.
        """
        stream = cls(reflection_threshold=reflection_threshold)
        for d in records:
            stream._entries.append(MemoryEntry.from_dict(d))
        stream._importance_accumulator = importance_accumulator
        reflections = [e for e in stream._entries if e.memory_type == "reflection"]
        if reflections:
            stream._last_reflected_at = max(reflections, key=lambda e: e.sim_time).sim_time
        return stream

    # ── Internal ─────────────────────────────────────────────────────────────

    def _score(self, entry: MemoryEntry, now: datetime) -> float:
        """
        Retrieval score: 0.5 * recency + 0.5 * normalized_importance.

        Recency uses exponential decay with a 24-hour half-life:
          recency = 2 ** (-(now - entry.sim_time) / 24h)
        Importance is normalized to [0, 1] by dividing by 10.
        """
        age = now - entry.sim_time
        # Guard against negative age (clock drift / test data)
        age_hours = max(0.0, age.total_seconds() / 3600)
        half_life_hours = _RECENCY_HALF_LIFE.total_seconds() / 3600
        recency = 2 ** (-age_hours / half_life_hours)
        importance_norm = entry.importance / 10.0
        return 0.5 * recency + 0.5 * importance_norm
