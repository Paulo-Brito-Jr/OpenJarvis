"""Private authorization receipt propagated by :class:`ToolExecutor`.

The receipt is carried out-of-band through ``ContextVar``. Tool parameters
cannot forge it, and filesystem tools require the exact tool and requested
path while using only the canonical resource carried by the receipt.

This is an executor boundary, not a sandbox for hostile Python extensions:
in-process tool implementations are trusted code. The receipt prevents model
arguments and direct public tool calls from bypassing ``ToolExecutor`` gates.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator

_RECEIPT_SEAL = object()


@dataclass(slots=True)
class _ExecutionReceipt:
    """Revocable, single-use proof that executor gates passed."""

    seal: object
    tool_name: str
    resource: str
    requested_resource: str | None
    agent_id: str
    active: bool = False
    consumed: bool = False
    revoked: bool = False
    lock: Lock = field(default_factory=Lock)


_CURRENT_EXECUTION: ContextVar[_ExecutionReceipt | None] = ContextVar(
    "openjarvis_authorized_tool_execution",
    default=None,
)


def _new_authorized_execution(
    tool_name: str,
    resource: str,
    requested_resource: str | None,
    agent_id: str,
) -> _ExecutionReceipt:
    """Create an unbound receipt after all executor gates have passed."""
    if not tool_name or not resource or not agent_id:
        raise ValueError("authorized execution receipt fields must be non-empty")
    return _ExecutionReceipt(
        seal=_RECEIPT_SEAL,
        tool_name=tool_name,
        resource=resource,
        requested_resource=requested_resource,
        agent_id=agent_id,
    )


@contextmanager
def _authorized_execution(
    receipt: _ExecutionReceipt,
) -> Iterator[None]:
    """Bind a receipt only inside the worker that executes the tool."""
    with receipt.lock:
        if receipt.seal is not _RECEIPT_SEAL or receipt.revoked:
            raise PermissionError("authorized execution receipt is invalid")
        receipt.active = True
    token = _CURRENT_EXECUTION.set(receipt)
    try:
        yield
    finally:
        _revoke_authorized_execution(receipt)
        _CURRENT_EXECUTION.reset(token)


def _revoke_authorized_execution(receipt: _ExecutionReceipt) -> None:
    """Invalidate a receipt, including copies inherited by async tasks."""
    with receipt.lock:
        receipt.active = False
        receipt.revoked = True


def _authorized_resource(
    tool_name: str,
    requested_resource: str,
) -> str | None:
    """Consume a matching receipt once and return its canonical resource."""
    receipt = _CURRENT_EXECUTION.get()
    if receipt is None:
        return None
    with receipt.lock:
        matches = bool(
            receipt.seal is _RECEIPT_SEAL
            and receipt.active
            and not receipt.consumed
            and not receipt.revoked
            and receipt.tool_name == tool_name
            and receipt.requested_resource == requested_resource
            and receipt.agent_id
        )
        if not matches:
            return None
        receipt.consumed = True
        return receipt.resource
