"""
BaseMemoryStream — abstract contract for occupant memory architectures.

Subclass this to experiment with alternative retrieval algorithms:
  - Attention-based retrieval
  - Graph-structured memory (event chains)
  - Episodic + semantic split (separate short-term and long-term stores)
  - Retrieval-augmented generation with a vector store

The default implementation (occupant_agent.agent.memory.MemoryStream) follows
Park et al. (2023): score = 0.5 × recency + 0.5 × (importance/10).

Example
───────
    from occupant_agent.core import BaseMemoryStream

    class AttentionMemory(BaseMemoryStream):
        \"\"\"Retrieval weighted by semantic similarity to current context.\"\"\"

        def retrieve(self, query_time, k=5):
            # Use sentence embeddings to rank by semantic relevance
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any


class BaseMemoryStream(ABC):
    """
    Abstract base class for occupant memory architectures.

    Subclass this — do NOT modify memory.py — to replace the retrieval
    function, storage backend, or reflection mechanism.

    All methods consumed by OccupantAgent are declared here. Return types
    use Any to avoid a circular import with agent.memory.MemoryEntry; the
    concrete MemoryStream returns properly typed MemoryEntry objects.
    """

    @abstractmethod
    def add(
        self,
        content: str,
        importance: float,
        memory_type: str,
        sim_time: datetime,
    ) -> Any:
        """
        Add a new memory and return it.

        Args:
            content:     Natural-language description (1–2 sentences).
            importance:  0–10 float; typically set by the LLM.
            memory_type: 'observation' | 'reflection' | 'signal_received'
            sim_time:    Simulation time of the event.
        """

    @abstractmethod
    def retrieve(self, query_time: datetime, k: int = 5) -> list[Any]:
        """
        Return the top-k most relevant entries for the given query time.

        The default implementation ranks by recency + importance.
        Override to add semantic similarity, graph traversal, etc.
        """

    @abstractmethod
    def should_reflect(self) -> bool:
        """Return True when accumulated importance exceeds the reflection threshold."""

    @abstractmethod
    def reflect(
        self,
        call_llm_fn: Callable[[str, str], dict],
        sim_time: datetime,
    ) -> list[Any]:
        """
        Synthesize high-level insights from recent memories via LLM.

        call_llm_fn(system, user) → dict must be provided by the caller
        (OccupantAgent passes its own LLM client).
        Returns the new reflection entry objects (already added to the stream).
        """

    @property
    @abstractmethod
    def entries(self) -> list[Any]:
        """All entries in chronological order."""

    @property
    @abstractmethod
    def importance_accumulator(self) -> float:
        """
        Current cumulative importance since the last reflection.
        Must be persisted and restored across stateless REST API calls.
        """

    @property
    @abstractmethod
    def last_reflected_at(self) -> datetime | None:
        """Simulation time of the most recent reflection, or None."""

    @abstractmethod
    def to_dicts(self) -> list[dict]:
        """Serialize all entries for SQLite persistence."""

    @classmethod
    @abstractmethod
    def from_dicts(
        cls,
        records: list[dict],
        importance_accumulator: float = 0.0,
        **kwargs: Any,
    ) -> BaseMemoryStream:
        """
        Restore a memory stream from serialized records.

        importance_accumulator must be passed from the persisted value so
        the reflection trigger fires correctly across stateless REST load/save.
        Extra kwargs (e.g., reflection_threshold) are passed through to the
        concrete implementation.
        """
