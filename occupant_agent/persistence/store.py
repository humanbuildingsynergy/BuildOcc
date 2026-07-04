"""
SQLite-backed session persistence for OccupantAgent.

Stores agent definitions, memory streams, actions, and signal interactions.
Each simulation run is identified by an agent_id (UUID4).

Design: stateless HTTP pattern.
  Each REST API request:
    1. Creates an AgentStore
    2. Calls load_agent(agent_id) to reconstruct the OccupantAgent from SQLite
    3. Runs the operation (step / signal / etc.)
    4. Calls sync_memories() + save_step() / save_signal() to persist results
    5. Discards the in-memory agent

This avoids global state in the server process and supports multi-process
deployment (e.g., uvicorn workers) without a shared memory layer.

Database path: env var OCCUPANT_AGENT_DB (default "./occupant_agent.db").

Tables:
  agents    — one row per agent (persona + LLM config)
  memories  — one row per MemoryEntry (joins on agent_id)
  actions   — one row per step() call (timestep + AgentAction JSON)
  signals   — one row per receive_signal() call
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from occupant_agent.agent.memory import MemoryStream
from occupant_agent.agent.occupant import OccupantAgent
from occupant_agent.agent.persona import Persona
from occupant_agent.environment.state import AgentAction, EnvironmentState, SignalResponse

# ── ORM base and models ───────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class AgentRecord(_Base):
    __tablename__ = "agents"

    agent_id   = Column(String, primary_key=True)
    stratum    = Column(String, nullable=False)
    seed       = Column(Integer, nullable=True)
    llm_provider = Column(String, nullable=False, default="anthropic")
    llm_model  = Column(String, nullable=True)
    persona_json = Column(Text, nullable=False)   # full Persona fields as JSON
    created_at = Column(DateTime, nullable=False)
    # Persisted so reflection fires correctly across stateless REST load/save cycles
    memory_accumulator = Column(Float, nullable=False, default=0.0)
    # Persisted so the 30-min thermostat cooldown guard works across REST API calls
    last_thermostat_step = Column(Integer, nullable=True)

    memories = relationship("MemoryRecord", back_populates="agent", cascade="all, delete-orphan")
    actions  = relationship("ActionRecord", back_populates="agent", cascade="all, delete-orphan")
    signals  = relationship("SignalRecord", back_populates="agent", cascade="all, delete-orphan")


class MemoryRecord(_Base):
    __tablename__ = "memories"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    agent_id     = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    memory_id    = Column(String, nullable=False, unique=True)
    content      = Column(Text, nullable=False)
    sim_time     = Column(DateTime, nullable=False)
    importance   = Column(Float, nullable=False)
    memory_type  = Column(String, nullable=False)
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime, nullable=True)

    agent = relationship("AgentRecord", back_populates="memories")


class ActionRecord(_Base):
    __tablename__ = "actions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    agent_id      = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    timestep      = Column(DateTime, nullable=False)
    atus_code     = Column(String, nullable=True)
    occupancy     = Column(String, nullable=True)   # always NULL; reserved for future per-room occupancy logging
    action_json   = Column(Text, nullable=False)
    extra_context = Column(Text, nullable=True)

    agent = relationship("AgentRecord", back_populates="actions")


class SignalRecord(_Base):
    __tablename__ = "signals"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    agent_id      = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    timestep      = Column(DateTime, nullable=False)
    signal_type   = Column(String, nullable=False)
    content       = Column(Text, nullable=False)
    response_json = Column(Text, nullable=False)
    atus_code     = Column(String, nullable=True)
    extra_context = Column(Text, nullable=True)

    agent = relationship("AgentRecord", back_populates="signals")


# ── AgentStore ────────────────────────────────────────────────────────────────

class AgentNotFoundError(KeyError):
    pass


class AgentStore:
    """
    Manages agent lifecycle (create, load, persist, delete) via SQLite.

    One store instance per process is fine; SQLAlchemy handles connection
    pooling. For multi-process uvicorn deployments, each worker creates its
    own store pointing at the same SQLite file (SQLite WAL mode handles
    concurrent reads; serialized writes via OS-level file locking).
    """

    def __init__(self, db_path: str | None = None) -> None:
        resolved = db_path or os.getenv("OCCUPANT_AGENT_DB", "./occupant_agent.db")
        self._engine = create_engine(
            f"sqlite:///{resolved}",
            connect_args={"check_same_thread": False},
        )
        _Base.metadata.create_all(self._engine)

    # ── Create ────────────────────────────────────────────────────────────────

    def create_agent(
        self,
        agent: OccupantAgent,
        seed: int | None = None,
    ) -> str:
        """
        Persist a newly created OccupantAgent and return its agent_id.

        Args:
            agent: Freshly created OccupantAgent (memory stream should be empty).
            seed:  RNG seed used to create the persona (for reproducibility).

        Returns:
            agent_id (UUID4 string).
        """
        agent_id = str(uuid.uuid4())
        persona_dict = _persona_to_dict(agent.persona)

        with Session(self._engine) as session:
            record = AgentRecord(
                agent_id=agent_id,
                stratum=agent.persona.stratum,
                seed=seed,
                llm_provider=agent.llm_provider,
                llm_model=agent.llm_model,
                persona_json=json.dumps(persona_dict),
                created_at=datetime.now(UTC),
                memory_accumulator=agent.memory.importance_accumulator,
                last_thermostat_step=None,
            )
            session.add(record)
            session.commit()

        return agent_id

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_agent(self, agent_id: str) -> OccupantAgent:
        """
        Reconstruct an OccupantAgent from the database.

        Raises:
            AgentNotFoundError: if agent_id is not found.
        """
        with Session(self._engine) as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                raise AgentNotFoundError(f"Agent not found: {agent_id!r}")

            persona = _persona_from_dict(json.loads(record.persona_json))

            memory_records = (
                session.execute(
                    select(MemoryRecord)
                    .where(MemoryRecord.agent_id == agent_id)
                    .order_by(MemoryRecord.id)
                )
                .scalars()
                .all()
            )
            memory_dicts = [_memory_record_to_dict(m) for m in memory_records]
            memory = MemoryStream.from_dicts(
                memory_dicts,
                importance_accumulator=record.memory_accumulator if record.memory_accumulator is not None else 0.0,
            )

            action_count = session.execute(
                select(func.count()).select_from(ActionRecord).where(
                    ActionRecord.agent_id == agent_id
                )
            ).scalar() or 0

        agent = OccupantAgent(
            persona=persona,
            memory=memory,
            llm_provider=record.llm_provider,
            llm_model=record.llm_model,
        )
        agent._action_count = action_count
        agent._last_thermostat_step = record.last_thermostat_step
        return agent

    # ── Persist ───────────────────────────────────────────────────────────────

    def sync_memories(
        self,
        agent_id: str,
        memory: MemoryStream,
        last_thermostat_step: int | None = None,
    ) -> None:
        """
        Upsert all memory entries and persist the importance accumulator.

        The accumulator and last_thermostat_step are written to AgentRecord so
        they survive the stateless REST API load/save cycle. Without persisting
        last_thermostat_step, the 30-min thermostat cooldown guard resets on
        every API request.
        """
        with Session(self._engine) as session:
            existing_ids: set[str] = set(
                session.execute(
                    select(MemoryRecord.memory_id).where(MemoryRecord.agent_id == agent_id)
                ).scalars()
            )

            for entry in memory.entries:
                if entry.memory_id in existing_ids:
                    rec = session.execute(
                        select(MemoryRecord).where(MemoryRecord.memory_id == entry.memory_id)
                    ).scalar_one_or_none()
                    if rec:
                        rec.access_count = entry.access_count
                        rec.last_accessed = entry.last_accessed
                else:
                    session.add(MemoryRecord(
                        agent_id=agent_id,
                        memory_id=entry.memory_id,
                        content=entry.content,
                        sim_time=entry.sim_time,
                        importance=entry.importance,
                        memory_type=entry.memory_type,
                        access_count=entry.access_count,
                        last_accessed=entry.last_accessed,
                    ))

            # Persist accumulator and thermostat cooldown state across requests
            agent_rec = session.get(AgentRecord, agent_id)
            if agent_rec is not None:
                agent_rec.memory_accumulator = memory.importance_accumulator
                agent_rec.last_thermostat_step = last_thermostat_step

            session.commit()

    def save_step(
        self,
        agent_id: str,
        action: AgentAction,
        env: EnvironmentState,
        atus_code: str | None = None,
        extra_context: str | None = None,
    ) -> None:
        """Persist an ActionRecord for a completed step() call."""
        with Session(self._engine) as session:
            session.add(ActionRecord(
                agent_id=agent_id,
                timestep=env.timestep,
                atus_code=atus_code,
                action_json=action.model_dump_json(),
                extra_context=extra_context,
            ))
            session.commit()

    def save_signal(
        self,
        agent_id: str,
        signal_type: str,
        content: str,
        response: SignalResponse,
        env: EnvironmentState,
        atus_code: str | None = None,
        extra_context: str | None = None,
    ) -> None:
        """Persist a SignalRecord for a completed receive_signal() call."""
        with Session(self._engine) as session:
            session.add(SignalRecord(
                agent_id=agent_id,
                timestep=env.timestep,
                signal_type=signal_type,
                content=content,
                response_json=response.model_dump_json(),
                atus_code=atus_code,
                extra_context=extra_context,
            ))
            session.commit()

    # ── List / Delete ─────────────────────────────────────────────────────────

    def list_agents(self) -> list[dict[str, Any]]:
        """Return summary metadata for all agents."""
        with Session(self._engine) as session:
            records = session.execute(select(AgentRecord)).scalars().all()
            result = []
            for r in records:
                mem_count = session.execute(
                    select(func.count()).select_from(MemoryRecord).where(
                        MemoryRecord.agent_id == r.agent_id
                    )
                ).scalar() or 0
                act_count = session.execute(
                    select(func.count()).select_from(ActionRecord).where(
                        ActionRecord.agent_id == r.agent_id
                    )
                ).scalar() or 0
                result.append({
                    "agent_id": r.agent_id,
                    "stratum": r.stratum,
                    "seed": r.seed,
                    "llm_provider": r.llm_provider,
                    "created_at": r.created_at.isoformat(),
                    "memory_count": mem_count,
                    "action_count": act_count,
                })
            return result

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all associated records. Returns True if deleted."""
        with Session(self._engine) as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True


# ── Serialization helpers ─────────────────────────────────────────────────────

def _persona_to_dict(p: Persona) -> dict[str, Any]:
    return {
        "stratum": p.stratum,
        "age": p.age,
        "sex": p.sex,
        "income_bracket": p.income_bracket,
        "state_fips": p.state_fips,
        "work_from_home": p.work_from_home,
        "home_gym": p.home_gym,
        "comfort_band_c": p.comfort_band_c,
        "appliances": sorted(p.appliances),
        "schedule_priors": p.schedule_priors,
        "core_memory_text": p.core_memory_text,
    }


def _persona_from_dict(d: dict[str, Any]) -> Persona:
    return Persona(
        stratum=d["stratum"],
        age=d["age"],
        sex=d["sex"],
        income_bracket=d["income_bracket"],
        state_fips=d["state_fips"],
        work_from_home=d["work_from_home"],
        home_gym=d["home_gym"],
        comfort_band_c=float(d.get("comfort_band_c", 2.0)),  # default for pre-v0.2 records
        appliances=set(d["appliances"]),
        schedule_priors={k: tuple(v) for k, v in d["schedule_priors"].items()},
        core_memory_text=d["core_memory_text"],
    )


def _memory_record_to_dict(m: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": m.memory_id,
        "content": m.content,
        "sim_time": m.sim_time.isoformat(),
        "importance": m.importance,
        "memory_type": m.memory_type,
        "access_count": m.access_count,
        "last_accessed": m.last_accessed.isoformat() if m.last_accessed else None,
    }
