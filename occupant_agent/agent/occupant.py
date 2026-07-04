"""
OccupantAgent — the core LLM-powered occupant simulation agent.

The agent receives an EnvironmentState at each 15-minute timestep and returns
an AgentAction. It also handles building control signals (Type A/B/C) via
receive_signal(), which returns a SignalResponse.

LLM reasoning is the key differentiator over rule-based baselines:
  - Multi-factor synthesis: balances comfort, cost, habit, and signal context
  - Signal comprehension: reasons differently for command vs. explanation vs. nudge
  - Persona coherence: all decisions flow through a demographically grounded persona
  - Memory reflection: emergent behavior change via accumulated experience insights

Internal flow (step):
  1. Resolve ATUS code → activity description + occupancy
  2. Retrieve top-5 memories by recency/importance
  3. Build system prompt (persona core memory + memories)
  4. Build user prompt (env state + activity + JSON output schema)
  5. Call LLM → parse dict with action fields + _memory_note + _importance
  6. Strip private fields, validate → AgentAction
  7. Add observation to memory stream
  8. If memory.should_reflect(): synthesize 3 insights via LLM
  9. Return AgentAction

The _ prefix convention: step() and receive_signal() ask the LLM to return
_memory_note and _importance alongside the public fields. These are stripped
before returning the Pydantic model to callers — the frozen AgentAction schema
is never changed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from occupant_agent.agent.memory import MemoryEntry, MemoryStream
from occupant_agent.agent.persona import Persona, create_persona
from occupant_agent.core.base_memory import BaseMemoryStream
from occupant_agent.core.base_persona import BasePersona
from occupant_agent.core.base_scheduler import BaseScheduler
from occupant_agent.environment.state import AgentAction, EnvironmentState, SignalResponse
from occupant_agent.llm.client import REFLECT_MODELS, call_llm

# ── Signal type labels (shown verbatim in the LLM prompt) ────────────────────

_SIGNAL_LABELS: dict[str, str] = {
    "A": "Direct instruction from your home's energy management system",
    "B": "Educational message explaining why a change would help",
    "C": "Comparison to similar households in your area",
}

# TOU threshold above which we label the rate "PEAK" in prompts
_PEAK_RATE_THRESHOLD = 0.15  # $/kWh

# Activity-to-room mapping for autonomous room movement.
# Deterministic: auto-move without an LLM call (high confidence, e.g. sleeping → bedroom).
# Soft: inject a directive hint and let the LLM decide (lower certainty).
# Each value is an ordered list of room_id candidates; first match in env.rooms wins.
_DETERMINISTIC_ROOMS: dict[str, list[str]] = {
    "sleeping": ["master_bedroom", "bedroom"],
    "laundry":  ["laundry_room"],
}
_SOFT_ROOMS: dict[str, list[str]] = {
    "food_prep": ["kitchen"],
    "eating":    ["dining_room", "kitchen"],
    "tv":        ["living_room"],
    "exercise":  ["living_room"],
}


class OccupantAgent:
    """
    LLM-powered occupant agent grounded in ATUS demographic priors.

    One agent instance = one simulated occupant. Multiple agents (e.g., O1 and
    O2 for a two-occupant household) are separate instances with separate memory
    streams.

    The agent is designed to be stateless between calls when used via the REST
    API: serialize with to_dict() after each step, restore with from_dict()
    before the next. The REST API handles this automatically.
    """

    def __init__(
        self,
        persona: BasePersona,
        memory: BaseMemoryStream | None = None,
        scheduler: BaseScheduler | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
    ) -> None:
        self.persona = persona
        self.memory = memory or MemoryStream()
        self.scheduler = scheduler
        self.llm_provider = llm_provider
        self.llm_model = llm_model  # None → provider default (haiku / gpt-4o-mini)
        self._action_count: int = 0
        self._last_action: AgentAction | None = None
        self._last_thermostat_step: int | None = None  # step index of last successful adjust_thermostat

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_stratum(
        cls,
        stratum: str,
        seed: int | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        state_fips: int = 48,
        scheduler: BaseScheduler | str | None = None,
    ) -> OccupantAgent:
        """
        Create an agent by sampling a persona from the given ATUS stratum.

        Args:
            stratum:      Registered stratum key — built-in: "O1" | "O2" | "O3" | "O4";
                          third-party strata registered via @register_stratum also work.
            seed:         RNG seed for reproducible persona sampling (None = random).
            llm_provider: "anthropic" | "openai" | "google" | "ollama"
            llm_model:    Override LLM model; None → provider default.
            state_fips:   US state FIPS (48 = Texas, Pecan Street cohort).
            scheduler:    Optional activity scheduler. Accepts:
                            - None: caller supplies atus_code to step() manually.
                            - BaseScheduler instance: used directly.
                            - str: registry key resolved via get_scheduler() and
                              instantiated with (stratum=stratum, seed=seed) if the
                              scheduler accepts those kwargs (e.g. "atus"), or with
                              no args otherwise (e.g. "fixed").
        """
        # Resolve persona from registry; fall back to built-in create_persona.
        try:
            from occupant_agent.core.registry import get_stratum
            factory = get_stratum(stratum)
            persona = factory(seed=seed, state_fips=state_fips)
        except (KeyError, TypeError):
            persona = create_persona(stratum, seed=seed, state_fips=state_fips)  # type: ignore[arg-type]

        # Resolve scheduler.
        sched: BaseScheduler | None = None
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
        )

    # ── Core API ──────────────────────────────────────────────────────────────

    def step(
        self,
        env: EnvironmentState,
        atus_code: str | None = None,
        wfh_today: bool | None = None,
        extra_context: str | None = None,
    ) -> AgentAction:
        """
        Advance one 15-minute timestep and return the agent's action.

        Args:
            env:           Current environment state (temp, devices, TOU rate, etc.).
            atus_code:     6-digit ATUS activity code for this timestep. If None,
                           the agent receives no activity context ("current activity: unknown").
            wfh_today:     Whether today is a work-from-home day for this agent. Call
                           persona.sample_wfh_today(rng) once per simulated day and pass
                           the result here. None = omit WFH context from the prompt.
            extra_context: Optional situational note injected into the user prompt
                           (e.g., "Today is a holiday", "Air quality AQI 180").
                           Useful when the calling application knows something that
                           EnvironmentState doesn't capture. Not stored in AgentAction.

        Returns:
            AgentAction (frozen schema v1.0).
        """
        if atus_code is None and self.scheduler is not None:
            atus_code = self.scheduler.sample(env.timestep)
        activity_desc, occupancy_str, devices_on = self._resolve_atus(atus_code)

        # ── Activity-driven room movement ─────────────────────────────────────
        suggested_room, is_deterministic = self._suggest_room(atus_code, env, wfh_today)
        current_room = next((r.room_id for r in env.rooms if r.occupied), None)

        if is_deterministic and suggested_room and suggested_room != current_room:
            room_name = suggested_room.replace("_", " ")
            action = AgentAction(
                action_type="move_room",
                target_id=suggested_room,
                value=None,
                reasoning=f"Transitioning to {activity_desc.lower()} — moving to {room_name}.",
            )
            self.memory.add(
                f"At {env.timestep.strftime('%H:%M')}, moved to {room_name} for {activity_desc.lower()}.",
                2.0,
                "observation",
                env.timestep,
            )
            self._action_count += 1
            self._last_action = action
            return action
        # ── End activity-driven room movement ─────────────────────────────────

        room_hint = ""
        if not is_deterministic and suggested_room:
            room_hint = self._build_room_hint(suggested_room, activity_desc, env)

        memories = self.memory.retrieve(env.timestep, k=5)

        # Always include the most recent entry so the agent knows what it just did,
        # even if high-importance older events push it out of the scored top-5.
        if self.memory.entries:
            most_recent = self.memory.entries[-1]
            retrieved_ids = {m.memory_id for m in memories}
            if most_recent.memory_id not in retrieved_ids:
                memories = [most_recent] + memories[:4]

        system = self._build_step_system(memories)
        user = self._build_step_user(env, activity_desc, occupancy_str, wfh_today, extra_context, devices_on, room_hint=room_hint)

        raw = self._call(system, user)
        action = self._parse_action(raw, env)

        # Hard floor: suppress back-to-back thermostat adjustments within 30 minutes.
        # _action_count increments AFTER this check, so diff==1 means one step (15 min)
        # has elapsed; diff<2 blocks only that one step, allowing again at diff==2 (30 min).
        # The soft prompt hint (in _build_step_user) is the primary lever; this is a
        # safety net for cases where the LLM ignores the hint.
        if (
            action.action_type == "adjust_thermostat"
            and self._last_thermostat_step is not None
            and (self._action_count - self._last_thermostat_step) < 2
        ):
            action = AgentAction(
                action_type="do_nothing",
                target_id=None,
                value=None,
                reasoning="Thermostat adjusted less than 30 minutes ago; holding setpoint.",
            )

        if action.action_type == "adjust_thermostat":
            self._last_thermostat_step = self._action_count

        memory_note = raw.get("_memory_note") or self._default_memory_note(action, env)
        importance = float(raw.get("_importance", 3))
        self.memory.add(memory_note, importance, "observation", env.timestep)

        if self.memory.should_reflect():
            try:
                self._run_reflect(env.timestep)
            except Exception:
                pass  # reflection failure does not discard the computed action

        self._action_count += 1
        self._last_action = action
        return action

    def receive_signal(
        self,
        signal_type: Literal["A", "B", "C"],
        content: str,
        env: EnvironmentState,
        atus_code: str | None = None,
        extra_context: str | None = None,
    ) -> SignalResponse:
        """
        Process a building control signal and return a response.

        Signal types (must stay distinct — see CLAUDE.md locked-in decisions):
          A — Direct command ("Turn off the dishwasher until after 9pm")
          B — Competence-building / boost (explains mechanism + rationale)
          C — Social norm / nudge (compares to similar households)

        Args:
            atus_code:     6-digit ATUS activity code for the current timestep. If None
                           and a scheduler is attached, the scheduler is sampled. Adds
                           activity context to the signal prompt for more grounded responses.
            extra_context: Optional situational note injected into the user prompt
                           (e.g., "Occupant is hosting guests"). Not stored in SignalResponse.

        Returns:
            SignalResponse with response ∈ {"accepted", "rejected", "deferred"}.
        """
        memories = self.memory.retrieve(env.timestep, k=5)
        signal_label = _SIGNAL_LABELS.get(signal_type, f"Signal type {signal_type}")

        # Resolve activity context: explicit code > scheduler > None (omitted from prompt)
        resolved_code = atus_code
        if resolved_code is None and self.scheduler is not None:
            resolved_code = self.scheduler.sample(env.timestep)
        if resolved_code is not None:
            activity_desc, occupancy_str, _ = self._resolve_atus(resolved_code)
        else:
            activity_desc, occupancy_str = None, None

        system = self._build_signal_system(memories)
        user = self._build_signal_user(signal_label, content, env, activity_desc, occupancy_str, extra_context)

        raw = self._call(system, user)
        response = self._parse_signal_response(raw, signal_type, content)

        importance = float(raw.get("_importance", 7))
        note = (
            f"Received {signal_type} signal: '{content[:80]}' → {response.response}"
        )
        self.memory.add(note, importance, "signal_received", env.timestep)

        return response

    @property
    def action_count(self) -> int:
        return self._action_count

    def get_state(self) -> dict[str, Any]:
        """
        Return a summary of the agent's current state (for REST API /state endpoint).
        """
        return {
            "stratum": self.persona.stratum,
            "age": self.persona.age,
            "sex": self.persona.sex,
            "work_from_home": self.persona.work_from_home,
            "home_gym": self.persona.home_gym,
            "llm_provider": self.llm_provider,
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
            "memory_count": len(self.memory.entries),
            "action_count": self._action_count,
            "last_reflected_at": (
                self.memory.last_reflected_at.isoformat()
                if self.memory.last_reflected_at
                else None
            ),
            "last_action": (
                self._last_action.model_dump() if self._last_action else None
            ),
        }

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_step_system(self, memories: list[MemoryEntry]) -> str:
        parts = [self.persona.core_memory_text]
        if memories:
            parts.append("\nRECENT MEMORIES (most relevant to now):")
            parts.append(self._format_memories(memories))
        if self.persona.prompt_suffix:
            parts.append(f"\n{self.persona.prompt_suffix}")
        return "\n".join(parts)

    def _device_activity_hint(self, devices_on: list[str], env: EnvironmentState) -> str:
        """Return a hint when the current activity implies devices that are currently OFF."""
        if not devices_on:
            return ""
        current_off = {d.device_id for d in env.devices if d.state is False}
        relevant = [did for did in devices_on if did in current_off]
        if not relevant:
            return ""
        # "oven" is forward-looking (Phase 2); it never appears in env.devices
        # but is listed here for completeness — _device_activity_hint() gracefully
        # skips absent device IDs because `relevant` filters against current_off.
        _names = {
            "tv": "TV",
            "washer": "washer",
            "dishwasher": "dishwasher",
            "oven": "oven",
            "lighting_living": "living room lights",
        }
        readable = ", ".join(_names.get(d, d) for d in relevant)
        return f"\nDevice context: Your current activity typically involves the {readable} — currently OFF."

    def _build_step_user(
        self,
        env: EnvironmentState,
        activity_desc: str,
        occupancy_str: str,
        wfh_today: bool | None = None,
        extra_context: str | None = None,
        devices_on: list[str] | None = None,
        room_hint: str = "",
    ) -> str:
        ts = env.timestep
        try:
            time_str = ts.strftime("%-I:%M %p")  # Linux
        except ValueError:
            time_str = ts.strftime("%I:%M %p").lstrip("0")  # macOS fallback

        day_str = ts.strftime("%A, %b %d")
        rate_label = "PEAK RATE" if env.tou_rate > _PEAK_RATE_THRESHOLD else "off-peak"

        setpoint_str = (
            f" | Thermostat set to {env.thermostat_setpoint_c:.1f}°C"
            if env.thermostat_setpoint_c is not None
            else ""
        )

        device_ids = [d.device_id for d in env.devices]
        room_ids   = [r.room_id   for r in env.rooms]

        wfh_line = ""
        if wfh_today is True:
            wfh_line = "\nNote: You are working from home today — you will not leave for the office."
        elif wfh_today is False and self.persona.wfh_probability > 0:
            wfh_line = "\nNote: Today is an in-office day — you will commute to work."

        context_line = f"\nNote: {extra_context}" if extra_context else ""

        # Cooling-mode hint: outdoor warmer than setpoint means HVAC is working against heat.
        # Raising setpoint reduces the temperature differential the HVAC must maintain.
        cooling_hint = ""
        if (
            env.thermostat_setpoint_c is not None
            and env.outdoor_temp_c > env.thermostat_setpoint_c
        ):
            cooling_hint = (
                "\nHVAC note: In cooling mode (outdoor warmer than setpoint), raising the "
                "thermostat setpoint reduces HVAC runtime and energy cost; lowering it "
                "increases cooling load."
            )

        device_hint = self._device_activity_hint(devices_on or [], env)

        # Comfort-band gate: show current deviation vs. persona threshold so the LLM
        # knows whether an adjustment is warranted. Present on every step.
        comfort_gate = ""
        if env.thermostat_setpoint_c is not None:
            band = self.persona.comfort_band_c
            deviation = abs(env.zone_temp_c - env.thermostat_setpoint_c)
            comfort_gate = (
                f"\nThermostat guidance: your comfort band is ±{band:.1f}°C from your "
                f"setpoint ({env.thermostat_setpoint_c:.1f}°C). Current deviation is "
                f"{deviation:.1f}°C. Only choose adjust_thermostat if this deviation "
                "exceeds your comfort band or you have a specific cost-saving reason."
            )

        # Inertia hint: remind the agent how recently it last adjusted, to discourage
        # micro-optimizing every 15 minutes. Shown only within the 2-hour window.
        recent_thermostat_note = ""
        if self._last_thermostat_step is not None:
            minutes_ago = (self._action_count - self._last_thermostat_step) * 15
            if minutes_ago < 120:
                recent_thermostat_note = (
                    f"\nNote: You adjusted the thermostat {minutes_ago} minutes ago. "
                    "Prefer do_nothing unless the temperature has moved significantly "
                    "outside your comfort band since then."
                )

        return f"""Time: {day_str} at {time_str}{wfh_line}
Current activity: {activity_desc} (you are {occupancy_str})
Indoor: {env.zone_temp_c:.1f}°C{setpoint_str} | Outdoor: {env.outdoor_temp_c:.1f}°C
Electricity rate: ${env.tou_rate:.3f}/kWh ({rate_label})

{self._format_env(env)}{context_line}{cooling_hint}{device_hint}{room_hint}{comfort_gate}{recent_thermostat_note}

What do you do right now? Respond with JSON only:
{{
  "action_type": "do_nothing" | "toggle_device" | "adjust_thermostat" | "move_room",
  "target_id": null | one of {device_ids} (for toggle_device) | one of {room_ids} (for move_room),
  "value": null | true | false | <float setpoint for adjust_thermostat>,
  "reasoning": "<1-2 sentences explaining your decision>",
  "_memory_note": "<1 sentence for your memory — what happened right now>",
  "_importance": <integer 1-10: 1-3=routine (nothing unusual), 4-6=notable (uncommon circumstance or conscious trade-off), 7-9=significant (major behavior change, peak event, signal accepted), 10=rare (extreme situation)>
}}"""

    def _build_signal_system(self, memories: list[MemoryEntry]) -> str:
        parts = [self.persona.core_memory_text]
        if memories:
            parts.append("\nPAST EXPERIENCES (relevant to this situation):")
            parts.append(self._format_memories(memories))
        if self.persona.prompt_suffix:
            parts.append(f"\n{self.persona.prompt_suffix}")
        return "\n".join(parts)

    def _build_signal_user(
        self,
        signal_label: str,
        content: str,
        env: EnvironmentState,
        activity_desc: str | None = None,
        occupancy_str: str | None = None,
        extra_context: str | None = None,
    ) -> str:
        ts = env.timestep
        try:
            time_str = ts.strftime("%-I:%M %p")
        except ValueError:
            time_str = ts.strftime("%I:%M %p").lstrip("0")

        rate_label = "PEAK RATE" if env.tou_rate > _PEAK_RATE_THRESHOLD else "off-peak"
        setpoint_str = (
            f" | Thermostat set to {env.thermostat_setpoint_c:.1f}°C"
            if env.thermostat_setpoint_c is not None
            else ""
        )
        activity_line = (
            f"\n- Current activity: {activity_desc} (you are {occupancy_str})"
            if activity_desc and occupancy_str
            else ""
        )
        context_line = f"\nNote: {extra_context}" if extra_context else ""

        return f"""You just received a message about your home energy use.

Message type: {signal_label}
Message: "{content}"

Current situation:
- Time: {ts.strftime('%A')} at {time_str}{activity_line}
- Indoor: {env.zone_temp_c:.1f}°C{setpoint_str} | Outdoor: {env.outdoor_temp_c:.1f}°C
- Electricity rate: ${env.tou_rate:.3f}/kWh ({rate_label})

{self._format_env(env)}{context_line}

How do you respond? Respond with JSON only:
{{
  "response": "accepted" | "rejected" | "deferred",
  "reasoning": "<1-2 sentences explaining why>",
  "_importance": <integer 1-10: 1-3=routine/ignored, 4-6=acknowledged but minor, 7-9=significant (changes your behavior), 10=rare urgent situation>
}}"""

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str, max_tokens: int = 512) -> dict[str, Any]:
        return call_llm(
            system=system,
            user=user,
            provider=self.llm_provider,
            model=self.llm_model,
            max_tokens=max_tokens,
        )

    def _call_reflect(self, system: str, user: str) -> dict[str, Any]:
        reflect_model = REFLECT_MODELS.get(self.llm_provider)
        return call_llm(
            system=system,
            user=user,
            provider=self.llm_provider,
            model=reflect_model,
            max_tokens=512,
        )

    # ── Response parsers ──────────────────────────────────────────────────────

    def _parse_action(self, raw: dict[str, Any], env: EnvironmentState) -> AgentAction:
        """
        Extract and validate AgentAction fields from the raw LLM dict.

        Enforces referential integrity (target_id must exist in env) and value
        constraints (thermostat clamped to 15–31°C) so downstream building
        controllers never receive semantically invalid commands.
        Unknown or out-of-range values fall back to do_nothing.
        """
        valid_types = {"do_nothing", "toggle_device", "adjust_thermostat", "move_room"}
        action_type = str(raw.get("action_type", "do_nothing")).strip().lower()
        if action_type not in valid_types:
            action_type = "do_nothing"

        target_id = raw.get("target_id")
        value = raw.get("value")

        device_ids = {d.device_id for d in env.devices}
        room_ids   = {r.room_id   for r in env.rooms}

        if action_type == "toggle_device":
            if target_id not in device_ids:
                action_type, target_id, value = "do_nothing", None, None
            else:
                # Coerce to bool; default True (turn on) if value absent.
                # Guard against string-encoded booleans ("false", "0") from LLMs
                # that don't enforce JSON schema — bool("false") == True in Python.
                if value is None:
                    value = True
                elif isinstance(value, str):
                    value = value.lower() not in ("false", "0", "no", "off")
                else:
                    value = bool(value)

        elif action_type == "adjust_thermostat":
            try:
                if isinstance(value, bool):
                    raise TypeError("boolean is not a valid thermostat setpoint")
                value = max(15.0, min(31.0, float(value)))
            except (TypeError, ValueError):
                action_type, target_id, value = "do_nothing", None, None

        elif action_type == "move_room":
            if target_id not in room_ids:
                action_type, target_id, value = "do_nothing", None, None

        return AgentAction(
            action_type=action_type,
            target_id=target_id,
            value=value,
            reasoning=raw.get("reasoning"),
        )

    def _parse_signal_response(
        self,
        raw: dict[str, Any],
        signal_type: str,
        content: str,
    ) -> SignalResponse:
        valid = {"accepted", "rejected", "deferred"}
        response = raw.get("response", "deferred")
        if response not in valid:
            response = "deferred"
        return SignalResponse(
            response=response,
            reasoning=raw.get("reasoning"),
        )

    # ── Reflection ────────────────────────────────────────────────────────────

    def _run_reflect(self, sim_time: datetime) -> None:
        """Trigger memory reflection using the smarter (sonnet/gpt-4o) model."""
        identity = self.persona.core_memory_text
        suffix = self.persona.prompt_suffix

        def reflect_call(system: str, user: str) -> dict:
            # Prepend persona identity so insights stay demographically grounded
            grounded = f"{identity}\n\n{system}"
            if suffix:
                grounded += f"\n{suffix}"
            return self._call_reflect(grounded, user)

        self.memory.reflect(reflect_call, sim_time)

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _format_memories(self, entries: list[MemoryEntry]) -> str:
        lines = []
        for i, e in enumerate(entries, 1):
            ts = e.sim_time.strftime("%a %H:%M")
            label = {"observation": "obs", "reflection": "insight", "signal_received": "signal"}.get(
                e.memory_type, e.memory_type
            )
            lines.append(f"  {i}. [{ts}][{label}] {e.content}")
        return "\n".join(lines)

    def _format_env(self, env: EnvironmentState) -> str:
        parts = []
        if env.devices:
            parts.append("Devices:")
            for d in env.devices:
                state_str = (
                    "ON" if d.state is True
                    else "OFF" if d.state is False
                    else f"{d.state}°C"
                )
                parts.append(f"  - {d.device_id}: {state_str} ({d.power_w:.0f}W rated)")
        if env.rooms:
            parts.append("Rooms:")
            for r in env.rooms:
                parts.append(f"  - {r.room_id}: {'occupied' if r.occupied else 'empty'}")
        return "\n".join(parts) if parts else "No device or room data."

    def _resolve_atus(self, atus_code: str | None) -> tuple[str, str, list[str]]:
        """
        Return (activity_description, occupancy_str, devices_on) for the given ATUS code.
        Falls back gracefully if the code is None or unmapped.
        """
        if atus_code is None:
            return "unspecified activity", "at home", []
        try:
            from occupant_agent.grounding.activity_code_map import lookup, resolve_occupancy

            mapping = lookup(atus_code)
            flags_obj = _PersonaFlagsAdapter(self.persona)
            occ = resolve_occupancy(atus_code, flags_obj)
            occ_str = {"home": "at home", "away": "away from home", "ambiguous": "at home"}.get(
                occ, "at home"
            )
            return mapping.description, occ_str, list(mapping.devices_on)
        except Exception:
            return f"activity {atus_code}", "at home", []

    def _suggest_room(
        self,
        atus_code: str | None,
        env: EnvironmentState,
        wfh_today: bool | None = None,
    ) -> tuple[str | None, bool]:
        """
        Return (room_id, is_deterministic) for the given ATUS activity, or (None, False).

        Deterministic rooms (sleeping, laundry) trigger an auto-move without an LLM call.
        Soft rooms (cooking, TV, eating, exercise, WFH work) produce a directive hint in
        the LLM prompt but leave the final decision to the agent.

        Room candidates are matched against env.rooms in priority order; the first match
        wins. Returns (None, False) when the activity has no clear home room (e.g. "other",
        away-bound activities, or when the suggested room is not in the room list).
        """
        if not atus_code or not env.rooms:
            return None, False

        from occupant_agent.grounding.scheduler import _get_category

        category = _get_category(atus_code)
        room_ids = {r.room_id for r in env.rooms}

        # WFH: use per-step flag when provided; fall back to persona-level sample
        effective_wfh = wfh_today if wfh_today is not None else self.persona.work_from_home
        if category == "work" and effective_wfh and "home_office" in room_ids:
            return "home_office", False

        for candidate in _DETERMINISTIC_ROOMS.get(category, []):
            if candidate in room_ids:
                return candidate, True

        for candidate in _SOFT_ROOMS.get(category, []):
            if candidate in room_ids:
                return candidate, False

        return None, False

    def _build_room_hint(
        self,
        suggested_room: str,
        activity_desc: str,
        env: EnvironmentState,
    ) -> str:
        """Return a directive room-context line for the LLM prompt."""
        room_name = suggested_room.replace("_", " ")
        already_there = any(r.room_id == suggested_room and r.occupied for r in env.rooms)
        if already_there:
            return f"\nActivity context: {activity_desc} — you are in the {room_name}, as expected."
        return (
            f"\nActivity context: {activity_desc} typically takes place in the {room_name}. "
            f'If you are transitioning to this activity, choose move_room with target_id="{suggested_room}".'
        )

    @staticmethod
    def _default_memory_note(action: AgentAction, env: EnvironmentState) -> str:
        ts = env.timestep.strftime("%H:%M")
        if action.action_type == "do_nothing":
            return f"At {ts}, did nothing in particular."
        if action.action_type == "toggle_device":
            state = "on" if action.value else "off"
            return f"At {ts}, turned {state} {action.target_id}."
        if action.action_type == "adjust_thermostat":
            return f"At {ts}, set thermostat to {action.value}°C."
        if action.action_type == "move_room":
            return f"At {ts}, moved to {action.target_id}."
        return f"At {ts}, took action: {action.action_type}."


# ── Adapter: Persona → PersonaFlags ──────────────────────────────────────────

class _PersonaFlagsAdapter:
    """Adapts Persona to the interface expected by activity_code_map.resolve_occupancy."""

    def __init__(self, persona: Persona) -> None:
        self.work_from_home = persona.work_from_home
        self.home_gym = persona.home_gym
