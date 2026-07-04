"""
OccupantAgent platform core — abstract base classes and plugin registry.

Extension point summary
───────────────────────
BasePersona      — subclass to define a new demographic profile or grounding source
BaseMemoryStream — subclass to swap the retrieval algorithm (e.g., attention-based)
BaseScheduler    — subclass to use a different activity grounding (Homer, fixed, custom)
register_*       — decorators to make extensions discoverable via OccupantAgent.from_stratum()

Quick example
─────────────
    from occupant_agent.core import BasePersona, BaseScheduler, register_stratum, register_scheduler
    from occupant_agent.agent.occupant import OccupantAgent

    @register_stratum("P5")
    class LowIncomeElderly(BasePersona):
        ...

    @register_scheduler("my_custom")
    class MyScheduler(BaseScheduler):
        ...

    agent = OccupantAgent.from_stratum("P5", seed=0, scheduler="my_custom")
"""

from occupant_agent.core.base_memory import BaseMemoryStream
from occupant_agent.core.base_persona import BasePersona
from occupant_agent.core.base_scheduler import BaseScheduler
from occupant_agent.core.registry import (
    get_scheduler,
    get_stratum,
    list_schedulers,
    list_strata,
    register_scheduler,
    register_stratum,
)

__all__ = [
    "BasePersona",
    "BaseMemoryStream",
    "BaseScheduler",
    "register_stratum",
    "register_scheduler",
    "get_stratum",
    "get_scheduler",
    "list_strata",
    "list_schedulers",
]
