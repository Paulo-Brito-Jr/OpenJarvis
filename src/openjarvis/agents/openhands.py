"""Security-disabled OpenHands SDK adapter.

The external SDK remains unavailable until it exposes a verified isolated
sandbox and OpenJarvis capability bridge. Construction never reads or retains
provider credentials.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.engine._stubs import InferenceEngine


@AgentRegistry.register("openhands")
class OpenHandsAgent(BaseAgent):
    """Agent that wraps the real openhands-sdk package.

    This is a thin adapter that delegates to the ``openhands-sdk``
    library for AI-driven software development tasks.  Requires
    ``openhands-sdk`` to be installed.
    """

    agent_id = "openhands"
    accepts_tools = False
    requires_security_context = True
    _default_temperature = 0.7
    _default_max_tokens = 1024

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            engine,
            model,
            bus=bus,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._workspace = workspace or os.getcwd()
        del api_key
        self._api_key = ""
        self._capability_policy = None
        self._security_agent_id = ""

    def bind_security(
        self,
        capability_policy: Optional[Any],
        agent_id: Optional[str] = None,
        boundary_guard: Optional[Any] = None,
    ) -> None:
        self._capability_policy = capability_policy
        self._security_agent_id = agent_id if isinstance(agent_id, str) else ""
        self._boundary_guard = boundary_guard

    def bind_boundary_guard(self, boundary_guard: Optional[Any]) -> None:
        self._boundary_guard = boundary_guard

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)
        self._emit_turn_end(turns=0, error=True)
        return AgentResult(
            content=(
                "OpenHands adapter disabled: the external SDK does not expose "
                "a verifiable OpenJarvis capability and isolated-sandbox "
                "boundary."
            ),
            turns=0,
            metadata={
                "error": True,
                "security_disabled": True,
                "reason": "unverified_external_sandbox",
            },
        )


__all__ = ["OpenHandsAgent"]
