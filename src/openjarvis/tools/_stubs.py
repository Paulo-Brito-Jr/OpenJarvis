"""ABC for tool implementations and the ToolExecutor dispatch engine.

Follows the same registry pattern as ``engine/_stubs.py`` and ``memory/_stubs.py``.
Each tool is registered via ``@ToolRegistry.register("name")`` and implements
``BaseTool`` with a ``spec`` property and ``execute()`` method.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.tools._execution_context import (
    _authorized_execution,
    _new_authorized_execution,
    _revoke_authorized_execution,
)

logger = logging.getLogger(__name__)

_MAX_TOOL_ARGUMENT_BYTES = 262_144
_MAX_TOOL_ARGUMENT_DEPTH = 20
_MAX_TOOL_ARGUMENT_NODES = 10_000

# ---------------------------------------------------------------------------
# ToolSpec — metadata describing a tool's interface
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolSpec:
    """Declarative description of a tool's interface and characteristics."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    cost_estimate: float = 0.0
    latency_estimate: float = 0.0
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0
    required_capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseTool ABC
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """Base class for all tool implementations.

    Subclasses must be registered via
    ``@ToolRegistry.register("name")`` to become discoverable.
    """

    tool_id: str
    is_local: bool = True
    execution_disabled_reason: str = ""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool specification."""

    @abstractmethod
    def execute(self, **params: Any) -> ToolResult:
        """Execute the tool with the given parameters."""

    def authorization_resource(self, params: Dict[str, Any]) -> str:
        """Return the concrete resource governed by capability policy.

        Tools with richer schemas may override this method.  The default
        deliberately derives a resource from user-controlled parameters
        instead of authorizing only against the tool name.
        """
        resource_keys = (
            "path",
            "file_path",
            "destination",
            "dest",
            "url",
            "uri",
            "endpoint",
            "workspace",
            "repo",
            "cwd",
            "device_id",
            "device",
            "entity_id",
            "entity",
            "channel",
            "to",
            "action_type",
            "action",
            "command",
            "query",
        )
        for key in resource_keys:
            value = params.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                resource = str(value).strip()
                if key in {"path", "file_path", "cwd"}:
                    # Authorize the same canonical target the filesystem will
                    # reach.  Otherwise `/safe/../etc/passwd` or an existing
                    # symlink could satisfy a `/safe/*` grant.
                    resource = str(Path(resource).expanduser().resolve(strict=False))
                return resource
        return f"tool:{self.spec.name}"

    def authorization_capabilities(self, params: Dict[str, Any]) -> List[str]:
        """Return capabilities required for this exact invocation."""
        del params
        # Keep the central mapping effective for legacy tools whose ToolSpec
        # predates declarative capabilities.  Explicit spec requirements are
        # additive, never a way to erase the secure built-in defaults.
        from openjarvis.security.capabilities import DEFAULT_TOOL_CAPABILITIES

        capabilities = list(DEFAULT_TOOL_CAPABILITIES.get(self.spec.name, []))
        capabilities.extend(self.spec.required_capabilities)
        return list(dict.fromkeys(capabilities))

    def requires_confirmation_for(self, params: Dict[str, Any]) -> bool:
        """Return whether this exact invocation needs live confirmation."""
        del params
        return bool(self.spec.requires_confirmation)

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        from openjarvis.tools.description_loader import (
            get_tool_description_override,
        )

        s = self.spec
        desc = get_tool_description_override(s.name) or s.description
        return {
            "type": "function",
            "function": {
                "name": s.name,
                "description": desc,
                "parameters": s.parameters,
            },
        }


# ---------------------------------------------------------------------------
# ToolExecutor — dispatch engine for tool calls
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Dispatch tool calls to registered tools with event bus integration.

    Parameters
    ----------
    tools:
        List of tool instances to make available.
    bus:
        Optional event bus for publishing ``TOOL_CALL_START``/``TOOL_CALL_END``.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        bus: Optional[EventBus] = None,
        *,
        interactive: bool = False,
        confirm_callback: Optional[Callable[[str], bool]] = None,
        default_timeout: float = 30.0,
        capability_policy: Optional[Any] = None,
        agent_id: str = "",
        boundary_guard: Optional[Any] = None,
    ) -> None:
        self._tools: Dict[str, BaseTool] = {t.spec.name: t for t in tools}
        self._bus = bus
        self._interactive = interactive
        self._confirm_callback = confirm_callback
        self._default_timeout = default_timeout
        self._boundary_guard = boundary_guard
        self.bind_security(capability_policy, agent_id)

    def bind_boundary_guard(self, boundary_guard: Optional[Any]) -> None:
        """Bind the mandatory scanner used at external tool boundaries."""
        self._boundary_guard = boundary_guard

    def guard_outbound_content(self, content: str, destination: str) -> str:
        """Scan arbitrary provider-bound text or deny when DLP is unwired."""
        if self._boundary_guard is None:
            raise PermissionError(
                "Boundary guard unavailable; outbound content denied."
            )
        try:
            guarded = self._boundary_guard.scan_outbound(content, destination)
        except Exception as exc:
            raise PermissionError(f"Security block: {exc}") from exc
        if not isinstance(guarded, str):
            raise PermissionError(
                "Boundary guard returned invalid content; outbound content denied."
            )
        return guarded

    def bind_security(
        self,
        capability_policy: Optional[Any],
        agent_id: str,
    ) -> None:
        """Bind the policy and identity used for all subsequent executions."""
        self._capability_policy = capability_policy
        self._agent_id = agent_id if isinstance(agent_id, str) else ""
        self._security_enabled = bool(
            capability_policy is not None
            and getattr(capability_policy, "enabled", True)
        )
        # A mutating tool may never turn a missing human gate into an implicit
        # approval.  Policy can add restrictions, but cannot disable the
        # confirmation declared by the tool.
        self._enforce_tool_confirmation = True

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Parse arguments, dispatch to tool, measure latency, emit events."""
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_name=tool_call.name,
                content=f"Unknown tool: {tool_call.name}",
                success=False,
            )
        disabled_reason = str(getattr(tool, "execution_disabled_reason", "")).strip()
        if disabled_reason:
            return ToolResult(
                tool_name=tool_call.name,
                content=f"Security block: {disabled_reason}",
                success=False,
                metadata={"security_disabled": True},
            )

        # Parse arguments
        try:
            if len(tool_call.arguments) > _MAX_TOOL_ARGUMENT_BYTES:
                raise ValueError("arguments exceed the 256 KiB limit")
            if len(tool_call.arguments.encode("utf-8")) > _MAX_TOOL_ARGUMENT_BYTES:
                raise ValueError("arguments exceed the 256 KiB limit")
            params = json.loads(tool_call.arguments) if tool_call.arguments else {}
            self._validate_argument_shape(params)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
            return ToolResult(
                tool_name=tool_call.name,
                content=f"Invalid arguments JSON: {exc}",
                success=False,
            )

        # Boundary guard: every external tool call must be scanned.  Missing
        # wiring is a denial, never a reason to silently skip DLP.
        if not getattr(tool, "is_local", True):
            if self._boundary_guard is None:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=(
                        "Security block: boundary guard unavailable; "
                        "external tool execution denied."
                    ),
                    success=False,
                )
            try:
                tool_call = self._boundary_guard.check_outbound(tool_call)
                # Re-parse arguments after potential redaction
                params = json.loads(tool_call.arguments) if tool_call.arguments else {}
            except Exception as exc:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=f"Security block: {exc}",
                    success=False,
                )

        # Taint checking (sink policy)
        taint_payload = params.get("_taint") if isinstance(params, dict) else None
        input_taint = None
        if taint_payload is not None:
            try:
                from openjarvis.security.taint import TaintSet, check_taint

                input_taint = (
                    taint_payload
                    if isinstance(taint_payload, TaintSet)
                    else TaintSet.from_json(taint_payload)
                )
                violation = check_taint(tool_call.name, input_taint)
                if violation:
                    if self._bus:
                        self._bus.publish(
                            EventType.TAINT_VIOLATION,
                            {
                                "tool": tool_call.name,
                                "violation": violation,
                            },
                        )
                    return ToolResult(
                        tool_name=tool_call.name,
                        content=f"Taint violation: {violation}",
                        success=False,
                    )
            except (ImportError, TypeError, ValueError) as exc:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=f"Invalid taint metadata; execution denied: {exc}",
                    success=False,
                )
            # Remove internal taint key before passing to tool
            if isinstance(params, dict):
                params.pop("_taint", None)

        requested_resource = params.get("path")
        if not isinstance(requested_resource, str):
            requested_resource = None
        try:
            resource = tool.authorization_resource(params)
            required_capabilities = tool.authorization_capabilities(params)
            requires_confirmation = tool.requires_confirmation_for(params)
        except Exception as exc:
            logger.warning(
                "Tool authorization metadata failed for tool=%s",
                tool_call.name,
                exc_info=True,
            )
            return ToolResult(
                tool_name=tool_call.name,
                content=f"Authorization metadata invalid; execution denied: {exc}",
                success=False,
            )

        denied = self.authorize(
            resource,
            required_capabilities,
            tool_name=tool_call.name,
        )
        if denied is not None:
            return denied

        # Confirmation check for sensitive tools
        if requires_confirmation and self._enforce_tool_confirmation:
            confirmation_denied = self.confirm_action(tool_call.name, params)
            if confirmation_denied is not None:
                return confirmation_denied

        # Emit start event. ``agent`` carries the managed-agent UUID so the
        # AgentExecutor's trace subscriber (which filters by agent_id) can
        # actually match this event — without it, every tool call is silently
        # dropped from traces.
        if self._bus:
            self._bus.publish(
                EventType.TOOL_CALL_START,
                {
                    "tool": tool_call.name,
                    "arguments": params,
                    "agent": self._agent_id,
                },
            )

        # Execute with timeout
        timeout = tool.spec.timeout_seconds or self._default_timeout
        t0 = time.time()
        pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        receipt = None
        try:
            receipt = _new_authorized_execution(
                tool_call.name,
                resource,
                requested_resource,
                self._agent_id,
            )
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            def execute_authorized() -> ToolResult:
                with _authorized_execution(receipt):
                    return tool.execute(**params)

            future = pool.submit(execute_authorized)
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if receipt is not None:
                _revoke_authorized_execution(receipt)
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            if self._bus:
                self._bus.publish(
                    EventType.TOOL_TIMEOUT,
                    {
                        "tool": tool_call.name,
                        "timeout": timeout,
                        "outcome": "unknown",
                        "reconcile_required": True,
                    },
                )
            result = ToolResult(
                tool_name=tool_call.name,
                content=(
                    f"Tool '{tool_call.name}' exceeded its {timeout:.0f}s timeout. "
                    "Outcome is unknown; reconcile the real target state before "
                    "retrying."
                ),
                success=False,
                metadata={
                    "outcome": "unknown",
                    "reconcile_required": True,
                    "resource": resource,
                },
            )
        except Exception as exc:
            if receipt is not None:
                _revoke_authorized_execution(receipt)
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            result = ToolResult(
                tool_name=tool_call.name,
                content=f"Tool execution error: {exc}",
                success=False,
            )
        else:
            pool.shutdown(wait=True)
        latency = time.time() - t0
        result.latency_seconds = latency
        result.metadata["arguments"] = params

        # Auto-detect taints in results
        if result.success:
            try:
                from openjarvis.security.taint import (
                    TaintSet,
                    propagate_taint,
                )

                base_taint = input_taint or TaintSet()
                existing_payload = result.metadata.get("_taint")
                if existing_payload is not None:
                    existing_taint = TaintSet.from_json(existing_payload)
                    base_taint = base_taint.union(existing_taint)
                detected = propagate_taint(
                    base_taint,
                    result.content,
                )
                if detected and detected.labels:
                    result.metadata["_taint"] = detected.to_json()
            except Exception:
                # The action may already have completed, but output with
                # malformed/unknown provenance cannot safely continue through
                # the agent. Preserve reconciliation metadata while withholding
                # the content.
                logger.warning(
                    "Output taint propagation failed for tool=%s",
                    tool_call.name,
                    exc_info=True,
                )
                result.success = False
                result.content = (
                    "Tool completed but output security metadata was invalid; "
                    "result withheld."
                )
                result.metadata.pop("_taint", None)
                result.metadata.update(
                    {
                        "outcome": "completed",
                        "result_withheld": True,
                        "resource": resource,
                    }
                )

        # Emit end event
        if self._bus:
            result_text = str(result.content)[:10240] if result.content else ""
            # Pass through ToolResult.metadata so downstream consumers
            # (TraceCollector → TraceStep.metadata → SkillOptimizer) can
            # see skill-tagged invocations.  Filter to JSON-serializable
            # values only — internal objects like TaintSet (added by the
            # taint auto-detect above) must not leak to event subscribers
            # since the trace store will JSON-serialize them later.
            event_metadata = self._json_safe_metadata(result.metadata)
            self._bus.publish(
                EventType.TOOL_CALL_END,
                {
                    "tool": tool_call.name,
                    "success": result.success,
                    "latency": latency,
                    "result": result_text,
                    "metadata": event_metadata,
                    "agent": self._agent_id,
                },
            )

        return result

    @staticmethod
    def _validate_argument_shape(params: Any) -> None:
        """Bound decoded argument shape before authorization or execution."""
        if not isinstance(params, dict):
            raise TypeError("tool arguments must decode to an object")
        stack = [(params, 1)]
        nodes = 0
        while stack:
            value, depth = stack.pop()
            nodes += 1
            if nodes > _MAX_TOOL_ARGUMENT_NODES:
                raise ValueError("arguments contain too many values")
            if depth > _MAX_TOOL_ARGUMENT_DEPTH:
                raise ValueError("arguments are nested too deeply")
            if isinstance(value, dict):
                stack.extend((item, depth + 1) for item in value.values())
            elif isinstance(value, list):
                stack.extend((item, depth + 1) for item in value)

    def confirm_action(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[ToolResult]:
        """Run the central live-confirmation gate for a sensitive action.

        Provider adapters sometimes perform an action without a registered
        ``BaseTool``.  They must call this after :meth:`authorize`; missing
        interactivity, callback, or a real TTY is a denial.
        """
        try:
            has_live_tty = bool(sys.stdin.isatty())
        except Exception:
            has_live_tty = False
        if not self._interactive or self._confirm_callback is None or not has_live_tty:
            return ToolResult(
                tool_name=tool_name,
                content=(
                    f"Tool '{tool_name}' requires"
                    " confirmation from a live TTY at execution time."
                ),
                success=False,
            )
        prompt = f"Allow execution of tool '{tool_name}' with args {params or {}}?"
        try:
            confirmed = self._confirm_callback(prompt)
        except Exception:
            logger.warning(
                "Tool confirmation callback failed for tool=%s",
                tool_name,
                exc_info=True,
            )
            return ToolResult(
                tool_name=tool_name,
                content=(f"Tool '{tool_name}' confirmation failed; execution denied."),
                success=False,
            )
        if confirmed is not True:
            return ToolResult(
                tool_name=tool_name,
                content=f"Tool '{tool_name}' execution denied by user.",
                success=False,
            )
        return None

    def authorize(
        self,
        resource: str,
        required_capabilities: Optional[List[str]] = None,
        *,
        tool_name: Optional[str] = None,
    ) -> Optional[ToolResult]:
        """Authorize a registered or virtual tool action.

        Returns ``None`` when allowed, otherwise a failed ``ToolResult``.
        Virtual actions (provider-side tools and internal REPLs) use this
        method so they cannot bypass the same policy boundary as normal tools.
        """
        if self._capability_policy is None:
            return self._capability_denied(
                tool_name or resource,
                "tool:invoke",
                "Capability policy unavailable; tool execution denied.",
            )
        if not self._agent_id.strip():
            return self._capability_denied(
                tool_name or resource,
                "tool:invoke",
                "Agent identity unavailable; tool execution denied.",
            )
        if not self._security_enabled:
            return self._capability_denied(
                tool_name or resource,
                "tool:invoke",
                "Capability policy is disabled; tool execution denied.",
            )

        capabilities = ["tool:invoke"]
        capabilities.extend(required_capabilities or [])
        capabilities = list(dict.fromkeys(capabilities))

        for raw_capability in capabilities:
            capability = str(getattr(raw_capability, "value", raw_capability))
            try:
                allowed = self._capability_policy.check(
                    self._agent_id,
                    capability,
                    resource,
                )
            except Exception:
                logger.warning(
                    "Capability policy check failed for agent=%s resource=%s",
                    self._agent_id,
                    resource,
                    exc_info=True,
                )
                return self._capability_denied(
                    tool_name or resource,
                    capability,
                    "Capability policy check failed; tool execution denied.",
                )
            if allowed is not True:
                return self._capability_denied(
                    tool_name or resource,
                    capability,
                    (
                        f"Capability '{capability}' denied for"
                        f" agent '{self._agent_id}'"
                        f" on resource '{resource}'."
                    ),
                )
        return None

    def _capability_denied(
        self,
        tool_name: str,
        capability: str,
        content: str,
    ) -> ToolResult:
        """Publish a denial event and return a failed tool result."""
        if self._bus:
            self._bus.publish(
                EventType.CAPABILITY_DENIED,
                {
                    "agent_id": self._agent_id,
                    "capability": capability,
                    "tool": tool_name,
                },
            )
        return ToolResult(
            tool_name=tool_name,
            content=content,
            success=False,
        )

    @staticmethod
    def _json_safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a copy of *metadata* containing only JSON-serializable values.

        ``ToolExecutor`` annotates ``ToolResult.metadata`` with internal
        objects (currently ``_taint: TaintSet``).  Those are useful for
        in-process security checks but cannot be serialized when the
        ``TraceCollector`` writes ``TraceStep.metadata`` to JSON in the
        SQLite trace store.  This helper drops any keys whose value is
        not JSON-safe — silently, since the missing data is not
        load-bearing for downstream consumers.
        """
        if not metadata:
            return {}

        import json

        safe: Dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                # Skip non-serializable values (e.g. TaintSet)
                continue
            safe[key] = value
        return safe

    def available_tools(self) -> List[ToolSpec]:
        """Return specs for all available tools."""
        return [t.spec for t in self._tools.values()]

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Return tools in OpenAI function-calling format."""
        return [t.to_openai_function() for t in self._tools.values()]


def build_tool_descriptions(
    tools: List[BaseTool],
    *,
    include_category: bool = True,
    include_cost: bool = False,
) -> str:
    """Build rich text descriptions from a list of tools.

    This is the single source of truth for all text-based agents that need
    to describe available tools in their system prompts.

    Parameters
    ----------
    tools:
        List of tool instances.
    include_category:
        Whether to include the ``Category:`` line.
    include_cost:
        Whether to include ``Cost estimate:`` and ``Latency estimate:`` lines.

    Returns
    -------
    str
        Formatted multi-tool description, or ``"No tools available."`` if
        *tools* is empty.
    """
    if not tools:
        return "No tools available."

    from openjarvis.tools.description_loader import (
        get_tool_description_override,
    )

    sections: list[str] = []
    for t in tools:
        s = t.spec
        desc = get_tool_description_override(s.name) or s.description
        lines = [f"### {s.name}", desc]

        if include_category and s.category:
            lines.append(f"Category: {s.category}")

        if include_cost:
            if s.cost_estimate:
                lines.append(f"Cost estimate: ${s.cost_estimate:.4f}")
            if s.latency_estimate:
                lines.append(f"Latency estimate: {s.latency_estimate:.1f}s")

        # Parameter descriptions
        props = s.parameters.get("properties", {})
        required = set(s.parameters.get("required", []))
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                req_mark = ", required" if pname in required else ""
                desc = pinfo.get("description", "")
                if desc:
                    lines.append(f"  - {pname} ({ptype}{req_mark}): {desc}")
                else:
                    lines.append(f"  - {pname} ({ptype}{req_mark})")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


__all__ = ["BaseTool", "ToolExecutor", "ToolSpec", "build_tool_descriptions"]
