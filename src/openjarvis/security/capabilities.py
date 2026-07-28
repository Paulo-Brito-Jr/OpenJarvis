"""RBAC capability system — fine-grained permission model for tool dispatch."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_UNSUPPORTED_GLOB_META = frozenset("?[]")


def _valid_glob_pattern(pattern: str) -> bool:
    """Return whether *pattern* uses the shared Python/Rust glob subset.

    Capability patterns are deliberately much smaller than shell globs:
    ``*`` is the only metacharacter.  Treating ``?`` or character classes
    differently in Python and Rust can turn a fallback into an authorization
    bypass, so malformed policy patterns invalidate the policy fail-closed.
    """
    return bool(pattern) and not any(char in pattern for char in _UNSUPPORTED_GLOB_META)


def _glob_match(pattern: str, text: str) -> bool:
    """Match the shared, case-sensitive, ``*``-only policy glob syntax."""
    if not _valid_glob_pattern(pattern):
        return False
    if "*" not in pattern:
        return pattern == text

    parts = pattern.split("*")
    position = 0

    first = parts[0]
    if first:
        if not text.startswith(first):
            return False
        position = len(first)

    for part in parts[1:-1]:
        if not part:
            continue
        found = text.find(part, position)
        if found < 0:
            return False
        position = found + len(part)

    last = parts[-1]
    if last:
        suffix_start = len(text) - len(last)
        if suffix_start < position or not text.endswith(last):
            return False
    return True


class Capability(str, Enum):
    """Fine-grained capability labels."""

    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    NETWORK_FETCH = "network:fetch"
    CODE_EXECUTE = "code:execute"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    CHANNEL_SEND = "channel:send"
    EMAIL_WRITE = "email:write"
    CALENDAR_WRITE = "calendar:write"
    MESSAGE_SEND = "message:send"
    APPROVAL_DECIDE = "approval:decide"
    TOOL_INVOKE = "tool:invoke"
    SCHEDULE_CREATE = "schedule:create"
    SYSTEM_ADMIN = "system:admin"


@dataclass(slots=True)
class CapabilityGrant:
    """A single capability grant for an agent."""

    capability: str  # Capability value or glob pattern
    pattern: str = "*"  # resource glob pattern


@dataclass(slots=True)
class AgentPolicy:
    """Policy for a specific agent."""

    agent_id: str
    grants: List[CapabilityGrant] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)  # explicit denials


class CapabilityPolicy:
    """RBAC capability policy for tool dispatch.

    Checks whether an agent has the required capability to invoke a tool.
    Policy can be loaded from a JSON file or configured programmatically.

    The default is deny-by-default.  The optional Rust implementation is an
    optimisation only; the Python policy remains authoritative and is used as
    a safe fallback when the native extension is unavailable.
    """

    def __init__(
        self,
        *,
        policy_path: Optional[str] = None,
        default_deny: bool = True,
        enabled: bool = True,
        enforce_tool_confirmation: bool = True,
    ) -> None:
        self._policies: Dict[str, AgentPolicy] = {}
        self._default_deny = default_deny
        self.enabled = enabled
        self.enforce_tool_confirmation = enforce_tool_confirmation
        self._rust_impl = None

        if enabled:
            try:
                from openjarvis._rust_bridge import get_rust_module

                _rust = get_rust_module()
                self._rust_impl = _rust.CapabilityPolicy(default_deny=default_deny)
            except Exception as exc:
                logger.info(
                    "Rust capability policy unavailable; "
                    "using safe Python fallback: %s",
                    exc,
                )

        if policy_path:
            self._load_file(Path(policy_path).expanduser())

    def grant(self, agent_id: str, capability: str, pattern: str = "*") -> None:
        """Grant a capability to an agent."""
        policy = self._policies.setdefault(
            agent_id,
            AgentPolicy(agent_id=agent_id),
        )
        policy.grants.append(CapabilityGrant(capability=capability, pattern=pattern))
        if self._rust_impl is not None:
            try:
                self._rust_impl.grant(agent_id, capability, pattern)
            except Exception:
                logger.warning(
                    "Rust capability grant failed; switching to Python fallback",
                    exc_info=True,
                )
                self._rust_impl = None

    def deny(self, agent_id: str, capability: str) -> None:
        """Explicitly deny a capability to an agent."""
        policy = self._policies.setdefault(
            agent_id,
            AgentPolicy(agent_id=agent_id),
        )
        policy.deny.append(capability)
        if self._rust_impl is not None:
            try:
                self._rust_impl.deny(agent_id, capability)
            except Exception:
                logger.warning(
                    "Rust capability denial failed; switching to Python fallback",
                    exc_info=True,
                )
                self._rust_impl = None

    def check(self, agent_id: str, capability: str, resource: str = "") -> bool:
        """Check whether *agent_id* has *capability* for *resource*.

        Returns True if allowed, False if denied.
        """
        if not self.enabled:
            # Disabling enforcement must never become an allow-all bypass.
            # Callers that intentionally want execution must provide an
            # enabled policy with explicit grants.
            return False
        if self._rust_impl is not None:
            try:
                return bool(self._rust_impl.check(agent_id, capability, resource))
            except Exception:
                logger.warning(
                    "Rust capability check failed; using Python fallback",
                    exc_info=True,
                )
                self._rust_impl = None
        return self._check_python(agent_id, capability, resource)

    def _check_python(self, agent_id: str, capability: str, resource: str = "") -> bool:
        """Evaluate a capability using the in-process policy state."""
        policy = self._policies.get(agent_id)
        if policy is None:
            # No explicit policy — use default
            return not self._default_deny

        # One malformed pattern makes the agent policy untrustworthy.  Deny
        # the whole check instead of ignoring a malformed deny or evaluating
        # it differently from the Rust backend.
        if any(not _valid_glob_pattern(denied) for denied in policy.deny):
            return False
        if any(
            not _valid_glob_pattern(grant.capability)
            or not _valid_glob_pattern(grant.pattern)
            for grant in policy.grants
        ):
            return False

        # Explicit denials take precedence
        for denied in policy.deny:
            if _glob_match(denied, capability):
                return False

        # Check grants
        for grant in policy.grants:
            if _glob_match(grant.capability, capability):
                if grant.pattern == "*":
                    return True
                if resource and _glob_match(grant.pattern, resource):
                    return True

        # No matching grant found
        return not self._default_deny

    def list_grants(self, agent_id: str) -> List[CapabilityGrant]:
        """List all grants for an agent."""
        policy = self._policies.get(agent_id)
        return list(policy.grants) if policy else []

    def list_agents(self) -> List[str]:
        """List all agents with explicit policies."""
        return list(self._policies.keys())

    def _load_file(self, path: Path) -> None:
        """Load policy from a JSON file."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for agent_data in data.get("agents", []):
                agent_id = agent_data["agent_id"]
                for grant_data in agent_data.get("grants", []):
                    self.grant(
                        agent_id,
                        grant_data["capability"],
                        grant_data.get("pattern", "*"),
                    )
                for denied in agent_data.get("deny", []):
                    self.deny(agent_id, denied)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse capability policy: %s", exc)

    def save(self, path: Path) -> None:
        """Save policy to a JSON file."""
        agents = []
        for agent_id, policy in self._policies.items():
            agents.append(
                {
                    "agent_id": agent_id,
                    "grants": [
                        {"capability": g.capability, "pattern": g.pattern}
                        for g in policy.grants
                    ],
                    "deny": policy.deny,
                }
            )
        path.write_text(json.dumps({"agents": agents}, indent=2))


# Default capability requirements for built-in tools
DEFAULT_TOOL_CAPABILITIES: Dict[str, List[str]] = {
    "file_read": [Capability.FILE_READ],
    "web_search": [Capability.NETWORK_FETCH],
    "code_interpreter": [Capability.CODE_EXECUTE],
    "code_interpreter_docker": [Capability.CODE_EXECUTE],
    "repl": [Capability.CODE_EXECUTE],
    "memory_store": [Capability.MEMORY_WRITE],
    "memory_retrieve": [Capability.MEMORY_READ],
    "memory_search": [Capability.MEMORY_READ],
    "memory_index": [Capability.FILE_READ, Capability.MEMORY_WRITE],
    "schedule_task": [Capability.SCHEDULE_CREATE],
    "channel_send": [Capability.CHANNEL_SEND],
    "check_permission": [Capability.MEMORY_READ],
    "queue_action": [Capability.MEMORY_WRITE],
    "get_pending_actions": [Capability.MEMORY_READ],
    "record_decision": [Capability.APPROVAL_DECIDE],
    "execute_pending_actions": [Capability.CODE_EXECUTE],
    "skill_manage": [Capability.FILE_READ],
}


__all__ = [
    "AgentPolicy",
    "Capability",
    "CapabilityGrant",
    "CapabilityPolicy",
    "DEFAULT_TOOL_CAPABILITIES",
]
