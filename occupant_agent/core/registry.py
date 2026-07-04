"""
Plugin registry for OccupantAgent extensions.

Decorators register subclasses of BasePersona and BaseScheduler under
string keys. OccupantAgent.from_stratum() resolves these keys at runtime,
so third-party packages can extend the platform without forking the core.

Built-in registrations (auto-populated on import):
  Strata:     O1, O2, O3, O4  (from occupant_agent.agent.persona)
  Schedulers: atus, fixed      (from occupant_agent.grounding.*)

Two ways to register a third-party extension
─────────────────────────────────────────────
Option A — decorator (session-level; works after an import):

    from occupant_agent.core import register_stratum, BasePersona

    @register_stratum("P5")
    class MyPersona(BasePersona):
        ...

    # Users must `import my_package` before calling from_stratum("P5").

Option B — entry_points (install-level; works after `pip install`):

    # In your package's pyproject.toml:
    [project.entry-points."occupant_agent.strata"]
    P5 = "my_package.personas:MyPersona"

    [project.entry-points."occupant_agent.schedulers"]
    homer = "my_package.schedulers:HomerScheduler"

    # No import needed — OccupantAgent discovers the plugin automatically.
    # The class is called as MyPersona(seed=..., state_fips=...) by from_stratum().

Entry-point group names:
  "occupant_agent.strata"     — BasePersona subclasses (or callable factories)
  "occupant_agent.schedulers" — BaseScheduler subclasses
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")

_STRATA: dict[str, type] = {}
_SCHEDULERS: dict[str, type] = {}


# ── Registration decorators ───────────────────────────────────────────────────

def register_stratum(name: str):
    """
    Register a BasePersona subclass under the given stratum key.

    Usage:
        @register_stratum("P5")
        class MyPersona(BasePersona): ...

        agent = OccupantAgent.from_stratum("P5", seed=0)
    """
    def decorator(cls: type) -> type:
        _STRATA[name] = cls
        return cls
    return decorator


def register_scheduler(name: str):
    """
    Register a BaseScheduler subclass under the given key.

    Usage:
        @register_scheduler("homer")
        class HomerScheduler(BaseScheduler): ...

        agent = OccupantAgent.from_stratum("O1", scheduler="homer")
    """
    def decorator(cls: type) -> type:
        _SCHEDULERS[name] = cls
        return cls
    return decorator


# ── Lookup ────────────────────────────────────────────────────────────────────

def get_stratum(name: str) -> type:
    """
    Return the registered BasePersona subclass for the given key.

    Raises:
        KeyError: if the key has not been registered.
    """
    _ensure_builtins_loaded()
    if name not in _STRATA:
        available = sorted(_STRATA)
        raise KeyError(
            f"Unknown stratum {name!r}. Available: {available}. "
            "Register with @register_stratum or import the package that defines it."
        )
    return _STRATA[name]


def get_scheduler(name: str) -> type:
    """
    Return the registered BaseScheduler subclass for the given key.

    Raises:
        KeyError: if the key has not been registered.
    """
    _ensure_builtins_loaded()
    if name not in _SCHEDULERS:
        available = sorted(_SCHEDULERS)
        raise KeyError(
            f"Unknown scheduler {name!r}. Available: {available}. "
            "Register with @register_scheduler or import the package that defines it."
        )
    return _SCHEDULERS[name]


def list_strata() -> list[str]:
    """Return all registered stratum keys."""
    _ensure_builtins_loaded()
    return sorted(_STRATA)


def list_schedulers() -> list[str]:
    """Return all registered scheduler keys."""
    _ensure_builtins_loaded()
    return sorted(_SCHEDULERS)


# ── Built-in registration (lazy to avoid circular imports) ────────────────────

_builtins_loaded = False


def _ensure_builtins_loaded() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True

    # Import triggers the @register_* decorators in each module
    import occupant_agent.agent.persona  # noqa: F401
    import occupant_agent.grounding.fixed_schedule  # noqa: F401
    import occupant_agent.grounding.scheduler  # noqa: F401

    # Discover third-party plugins declared via pyproject.toml entry_points.
    # This runs AFTER built-ins so setdefault() protects O1-O4/atus/fixed
    # from being silently overwritten by a third-party package.
    _load_entry_points()


def _load_entry_points() -> None:
    """
    Scan installed packages for OccupantAgent plugin entry points.

    Uses importlib.metadata (stdlib ≥ 3.9) — no extra dependencies.
    Errors in individual plugins emit a warning and are skipped so that
    one broken extension cannot prevent the rest from loading.
    """
    import warnings

    try:
        from importlib.metadata import entry_points
    except ImportError:
        return  # Python < 3.9 — skip silently

    for group, registry in (
        ("occupant_agent.strata",     _STRATA),
        ("occupant_agent.schedulers", _SCHEDULERS),
    ):
        for ep in entry_points(group=group):
            if ep.name in registry:
                continue  # built-in or already registered — don't override
            try:
                obj = ep.load()
                registry[ep.name] = obj
            except Exception as exc:
                warnings.warn(
                    f"OccupantAgent: failed to load plugin "
                    f"{group!r}/{ep.name!r} from {ep.value!r}: {exc}",
                    stacklevel=4,
                )
