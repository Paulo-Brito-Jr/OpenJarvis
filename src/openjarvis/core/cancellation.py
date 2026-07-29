"""Cooperative cancellation shared by agents and tool execution.

``asyncio`` cannot forcibly stop synchronous work already running in a worker
thread.  A request therefore carries a thread-safe token that every boundary
can inspect before starting more work.  The token is also exposed through a
``ContextVar`` so existing synchronous agent/tool APIs do not need a new
parameter on every method.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event
from typing import Iterator


class AgentCancelledError(RuntimeError):
    """Raised when a request has been cancelled by its caller."""


class CancellationToken:
    """Thread-safe, one-way cooperative cancellation signal."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = Event()

    def cancel(self) -> None:
        """Signal cancellation.  Repeated calls are harmless."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been signalled."""
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        """Stop cooperative work after the caller disconnects."""
        if self.is_cancelled:
            raise AgentCancelledError("Agent request was cancelled by the caller.")


_CURRENT_CANCELLATION: ContextVar[CancellationToken | None] = ContextVar(
    "openjarvis_agent_cancellation",
    default=None,
)


@contextmanager
def cancellation_scope(token: CancellationToken) -> Iterator[None]:
    """Bind *token* to synchronous work in the current execution context."""
    context_token = _CURRENT_CANCELLATION.set(token)
    try:
        yield
    finally:
        _CURRENT_CANCELLATION.reset(context_token)


def current_cancellation_token() -> CancellationToken | None:
    """Return the token bound to the current agent/tool execution."""
    return _CURRENT_CANCELLATION.get()


def raise_if_cancelled() -> None:
    """Raise when the current execution context has been cancelled."""
    token = current_cancellation_token()
    if token is not None:
        token.raise_if_cancelled()


__all__ = [
    "AgentCancelledError",
    "CancellationToken",
    "cancellation_scope",
    "current_cancellation_token",
    "raise_if_cancelled",
]
