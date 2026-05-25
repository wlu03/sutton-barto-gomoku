"""Agent registry.

Decorate an `Agent` subclass (or a factory callable that returns one) and it
shows up in the web UI's dropdown. Example:

    from gomoku.agents.base import Agent
    from gomoku.web.registry import register_agent

    @register_agent("Random")
    class RandomAgent(Agent):
        def act(self, board, to_play, legal_mask):
            ...

Agents can expose introspection by setting a `last_debug` attribute (any
JSON-serializable dict) inside `act()`. The server forwards it to the UI and
the move history will surface it.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

from gomoku.agents.base import Agent

_REGISTRY: dict[str, dict[str, Any]] = {}


def register_agent(name: str, *, doc: str | None = None):
    def decorator(cls_or_factory):
        if name in _REGISTRY:
            raise ValueError(f"agent {name!r} already registered")
        first_line = (cls_or_factory.__doc__ or "").strip().split("\n", 1)[0]
        _REGISTRY[name] = {
            "factory": cls_or_factory,
            "doc": doc or first_line,
            "module": cls_or_factory.__module__,
        }
        return cls_or_factory
    return decorator


def get_factory(name: str) -> Callable[[], Agent]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown agent {name!r}")
    return _REGISTRY[name]["factory"]


def list_agents() -> list[dict[str, str]]:
    return [
        {"name": n, "doc": v["doc"], "module": v["module"]}
        for n, v in sorted(_REGISTRY.items())
    ]


def autoload(package_name: str = "gomoku.agents") -> None:
    """Import every submodule of `package_name` so @register_agent runs."""
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):
        return
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except ImportError:
            # Skip submodules whose optional deps aren't installed.
            pass
