# ruff: noqa: E501
"""Proactive agent tools — check/record permissions, queue and execute actions.

These tools are used exclusively by ``ProactiveAgent`` to manage the
propose → approve → execute lifecycle for autonomous actions.

Permission key convention: ``"{action_type}:{context_key}"``

Approval response parsing
-------------------------
When the user replies to a pending-actions notification, their message is
expected to contain one or more tokens of the form:

    ``{action_id} yes``   or   ``{action_id} no``
    ``yes {action_id}``   or   ``no {action_id}``
    ``always yes {action_id}``  →  approve + remember
    ``always no {action_id}``   →  deny + remember
    ``yes all``  /  ``no all``  →  bulk approve/deny all pending

Call ``parse_approval_response(text, store)`` from any channel message handler
to process these replies without running the full agent.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.approval_store import (
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    TIER_TRIVIAL,
    ApprovalStore,
    PendingAction,
)

# ---------------------------------------------------------------------------
# Shared store (lazily initialised, one per process)
# ---------------------------------------------------------------------------

_store: Optional[ApprovalStore] = None
_DISABLED_REASON = (
    "Proactive actions are disabled until approval is authenticated, "
    "digest-bound, and TTL-validated server-side."
)


def _disabled_result(tool_name: str) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=False,
        content=_DISABLED_REASON,
        metadata={
            "security_disabled": True,
            "reason": "authenticated_digest_approval_required",
        },
    )


def get_store() -> ApprovalStore:
    global _store
    if _store is None:
        _store = ApprovalStore()
    return _store


# ---------------------------------------------------------------------------
# check_permission
# ---------------------------------------------------------------------------


@ToolRegistry.register("check_permission")
class CheckPermissionTool(BaseTool):
    """Look up whether the user has a remembered decision for a permission key."""

    tool_id = "check_permission"

    def __init__(self, store: Optional[ApprovalStore] = None) -> None:
        self._store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="check_permission",
            description=(
                "Check whether the user has a remembered permission decision for "
                "an action pattern. Returns 'always_approve', 'always_deny', or 'unknown'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "permission_key": {
                        "type": "string",
                        "description": (
                            "Permission pattern key, e.g. "
                            "'email_delete:domain:noreply.github.com'"
                        ),
                    },
                },
                "required": ["permission_key"],
            },
            category="proactive",
            required_capabilities=["memory:read"],
        )

    def execute(self, **params: Any) -> ToolResult:
        key = params.get("permission_key", "")
        store = self._store or get_store()
        rule = store.get_permission(key)
        decision = rule.decision if rule else "unknown"
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=decision,
            metadata={"permission_key": key, "decision": decision},
        )

    def authorization_resource(self, params: Dict[str, Any]) -> str:
        key = str(params.get("permission_key", "")).strip()
        return f"permission:{key or 'unknown'}"


# ---------------------------------------------------------------------------
# queue_action
# ---------------------------------------------------------------------------


@ToolRegistry.register("queue_action")
class QueueActionTool(BaseTool):
    """Queue a proposed action for user approval or immediate execution."""

    tool_id = "queue_action"

    def __init__(self, store: Optional[ApprovalStore] = None) -> None:
        self._store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="queue_action",
            description=(
                "Queue a proposed action. Tier controls whether user approval is required:\n"
                f"  '{TIER_TRIVIAL}' — execute immediately, no approval needed\n"
                f"  '{TIER_LOW}'     — ask once per pattern, then remember\n"
                f"  '{TIER_MEDIUM}'  — ask each time unless user said 'always'\n"
                f"  '{TIER_HIGH}'    — always ask, never auto-remember\n"
                "Returns the action_id so you can reference it in notifications."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": "Short slug, e.g. 'email_delete', 'sms_draft_reply'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what will be done.",
                    },
                    "payload": {
                        "type": "object",
                        "description": "JSON payload the executor will use to carry out the action.",
                    },
                    "permission_key": {
                        "type": "string",
                        "description": "Pattern key for permission memory lookup.",
                    },
                    "tier": {
                        "type": "string",
                        "enum": [TIER_TRIVIAL, TIER_LOW, TIER_MEDIUM, TIER_HIGH],
                        "description": "Approval tier.",
                    },
                },
                "required": [
                    "action_type",
                    "description",
                    "payload",
                    "permission_key",
                    "tier",
                ],
            },
            category="proactive",
            requires_confirmation=True,
            required_capabilities=["memory:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        del params
        return _disabled_result(self.spec.name)

    def authorization_resource(self, params: Dict[str, Any]) -> str:
        key = str(params.get("permission_key", "")).strip()
        return f"approval-proposal:{key or 'unknown'}"


# ---------------------------------------------------------------------------
# get_pending_actions
# ---------------------------------------------------------------------------


@ToolRegistry.register("get_pending_actions")
class GetPendingActionsTool(BaseTool):
    """Return all pending (not yet decided) actions as a JSON list."""

    tool_id = "get_pending_actions"

    def __init__(self, store: Optional[ApprovalStore] = None) -> None:
        self._store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_pending_actions",
            description="Return all pending actions awaiting user approval as a JSON list.",
            parameters={"type": "object", "properties": {}},
            category="proactive",
            required_capabilities=["memory:read"],
        )

    def execute(self, **params: Any) -> ToolResult:
        store = self._store or get_store()
        actions = store.list_pending()
        data = [
            {
                "id": a.id,
                "action_type": a.action_type,
                "description": a.description,
                "tier": a.tier,
                "permission_key": a.permission_key,
                "created_at": a.created_at,
            }
            for a in actions
        ]
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=json.dumps(data, indent=2),
            metadata={"count": len(data)},
        )

    def authorization_resource(self, params: Dict[str, Any]) -> str:
        del params
        return "approval-queue"


# ---------------------------------------------------------------------------
# record_decision
# ---------------------------------------------------------------------------


@ToolRegistry.register("record_decision")
class RecordDecisionTool(BaseTool):
    """Record a user approval or denial for a queued action."""

    tool_id = "record_decision"

    def __init__(self, store: Optional[ApprovalStore] = None) -> None:
        self._store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="record_decision",
            description=(
                "Record the user's approval or denial for a pending action. "
                "Set remember=true to save the decision to permission memory so "
                "the same pattern is handled automatically in future."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "The action_id returned by queue_action.",
                    },
                    "approved": {
                        "type": "boolean",
                        "description": "True to approve, false to deny.",
                    },
                    "remember": {
                        "type": "boolean",
                        "description": "Save decision to permission memory for this pattern.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional note to store alongside the permission rule.",
                    },
                },
                "required": ["action_id", "approved"],
            },
            category="proactive",
            requires_confirmation=True,
            required_capabilities=["approval:decide"],
        )

    def execute(self, **params: Any) -> ToolResult:
        del params
        return _disabled_result(self.spec.name)

    def authorization_resource(self, params: Dict[str, Any]) -> str:
        action_id = str(params.get("action_id", "")).strip()
        return f"approval:{action_id or 'unknown'}"


# ---------------------------------------------------------------------------
# execute_pending_actions
# ---------------------------------------------------------------------------


@ToolRegistry.register("execute_pending_actions")
class ExecutePendingActionsTool(BaseTool):
    """Execute all approved (or trivial) actions and return a summary."""

    tool_id = "execute_pending_actions"

    def __init__(
        self,
        store: Optional[ApprovalStore] = None,
        executor_fn: Optional[Any] = None,
    ) -> None:
        self._store = store
        # executor_fn(action: PendingAction) -> (success: bool, message: str)
        self._executor_fn = executor_fn

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="execute_pending_actions",
            description=(
                "Execute all approved actions in the queue. "
                "Returns a JSON summary of what succeeded and what failed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific action IDs to execute. "
                        "If omitted, executes all approved actions.",
                    },
                },
            },
            category="proactive",
            requires_confirmation=True,
            required_capabilities=[
                "approval:decide",
                "code:execute",
                "email:write",
                "calendar:write",
                "message:send",
            ],
        )

    def execute(self, **params: Any) -> ToolResult:
        del params
        return _disabled_result(self.spec.name)

    def authorization_resource(self, params: Dict[str, Any]) -> str:
        raw_ids = params.get("action_ids")
        if isinstance(raw_ids, list):
            ids = sorted(
                str(value).strip()
                for value in raw_ids
                if isinstance(value, str) and value.strip()
            )
            if ids:
                return "approval-batch:" + ",".join(ids)
        return "approval-batch:all"

    def _run_action(self, action: PendingAction) -> Tuple[bool, str]:
        del action
        return False, _DISABLED_REASON


# ---------------------------------------------------------------------------
# Built-in action executors (thin wrappers around connector/channel APIs)
# ---------------------------------------------------------------------------


def _exec_email_delete(payload: Dict[str, Any]) -> Tuple[bool, str]:
    del payload
    return False, _DISABLED_REASON


def _exec_email_archive(payload: Dict[str, Any]) -> Tuple[bool, str]:
    del payload
    return False, _DISABLED_REASON


def _exec_sms_send(payload: Dict[str, Any]) -> Tuple[bool, str]:
    del payload
    return False, _DISABLED_REASON


def _exec_calendar_decline(payload: Dict[str, Any]) -> Tuple[bool, str]:
    del payload
    return False, _DISABLED_REASON


def _exec_calendar_accept(payload: Dict[str, Any]) -> Tuple[bool, str]:
    del payload
    return False, _DISABLED_REASON


# ---------------------------------------------------------------------------
# Approval response parser (for channel message handlers)
# ---------------------------------------------------------------------------

# Matches: "abc123 yes", "yes abc123", "always yes abc123", "yes all", etc.
_APPROVAL_RE = re.compile(
    r"\b(?P<always>always\s+)?(?P<decision>yes|no|approve|deny)\s+(?P<target>[a-f0-9]{12}|all)\b"
    r"|"
    r"\b(?P<target2>[a-f0-9]{12}|all)\s+(?P<always2>always\s+)?(?P<decision2>yes|no|approve|deny)\b",
    re.IGNORECASE,
)


def parse_approval_response(
    text: str,
    store: Optional[ApprovalStore] = None,
) -> List[Dict[str, Any]]:
    """Parse a free-text message for approval tokens and update the store.

    Returns a list of dicts describing each decision that was processed,
    for use in an acknowledgement message back to the user.

    Call this from any channel message handler before routing the message
    to the main agent, e.g. inside the iMessage daemon or Telegram bot.
    """
    del text, store
    # Free text from a channel is neither an authenticated human decision nor
    # bound to the exact action digest.  It must never mutate approval state.
    return []


__all__ = [
    "CheckPermissionTool",
    "ExecutePendingActionsTool",
    "GetPendingActionsTool",
    "QueueActionTool",
    "RecordDecisionTool",
    "get_store",
    "parse_approval_response",
]
