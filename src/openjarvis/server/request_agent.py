"""Request-scoped agent container for authenticated API execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openjarvis.core.events import EventBus


@dataclass(frozen=True, slots=True)
class RequestAgentScope:
    """An agent and event bus that belong to exactly one HTTP request."""

    agent: Any
    bus: EventBus


__all__ = ["RequestAgentScope"]
