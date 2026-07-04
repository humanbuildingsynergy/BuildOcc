# Canonical location moved to occupant_agent/grounding/activity_code_map.py.
# This shim preserves backward compatibility for direct script callers.
from occupant_agent.grounding.activity_code_map import (  # noqa: F401
    ACTIVITY_MAP, TIER2_FALLBACK, TIER1_FALLBACK,
    ActivityMapping, PersonaFlags,
    lookup, resolve_occupancy,
)
