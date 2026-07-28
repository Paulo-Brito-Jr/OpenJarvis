"""Executes user queries through the engine or through an agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openjarvis.core.types import Message, Role
from openjarvis.tools._stubs import BaseTool

if TYPE_CHECKING:
    from openjarvis.system.protocols import OrchestratorDeps

logger = logging.getLogger(__name__)


class _ScopedCapabilityPolicy:
    """Intersect live grants with an immutable invocation capability snapshot."""

    def __init__(self, policy: Any, capabilities: List[str]) -> None:
        self._policy = policy
        self._capabilities = frozenset(capabilities)
        self.enabled = bool(
            policy is not None and getattr(policy, "enabled", True)
        )

    def check(self, agent_id: str, capability: str, resource: str = "") -> bool:
        if (
            not self.enabled
            or capability not in self._capabilities
            or self._policy is None
        ):
            return False
        try:
            return self._policy.check(agent_id, capability, resource) is True
        except Exception:
            logger.exception("Scoped capability check failed")
            return False


def _normalize_operator_id(operator_id: Any) -> Optional[str]:
    """Preserve the distinction between internal and external invocations."""
    if operator_id is None:
        return None
    if isinstance(operator_id, str):
        return operator_id.strip()
    return ""


class QueryOrchestrator:
    def __init__(self, system: OrchestratorDeps) -> None:
        self._system = system

    def ask(
        self,
        query: str,
        *,
        context: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent: Optional[str] = None,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        operator_id: Optional[str] = None,
        prior_messages: Optional[List[Message]] = None,
        capability_scope: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a query through the system and return a result dict."""
        s = self._system
        if temperature is None:
            temperature = s.config.intelligence.temperature
        if max_tokens is None:
            max_tokens = s.config.intelligence.max_tokens

        messages = [Message(role=Role.USER, content=query)]
        effective_operator_id = _normalize_operator_id(operator_id)

        execution_policy = s.capability_policy
        if capability_scope is not None:
            execution_policy = _ScopedCapabilityPolicy(
                s.capability_policy,
                capability_scope,
            )

        memory_context_allowed = True
        if effective_operator_id is not None:
            memory_context_allowed = False
            policy = execution_policy
            if effective_operator_id and policy is not None:
                try:
                    memory_context_allowed = (
                        policy.check(
                            effective_operator_id,
                            "memory:read",
                            "memory:context",
                        )
                        is True
                    )
                except Exception:
                    logger.exception(
                        "Memory context capability check failed; "
                        "continuing without memory"
                    )

        if (
            context
            and memory_context_allowed
            and s.memory_backend
            and s.config.agent.context_from_memory
        ):
            try:
                from openjarvis.tools.storage.context import (
                    ContextConfig,
                    inject_context,
                )

                ctx_cfg = ContextConfig(
                    top_k=s.config.memory.context_top_k,
                    min_score=s.config.memory.context_min_score,
                    max_context_tokens=s.config.memory.context_max_tokens,
                )
                messages = inject_context(
                    query,
                    messages,
                    s.memory_backend,
                    config=ctx_cfg,
                )
            except Exception as exc:
                logger.warning("Failed to inject memory context: %s", exc)

        use_agent = agent or s.agent_name
        if not agent and use_agent != "none":
            detected = self._detect_agent_intent(query)
            if detected:
                use_agent = detected
        if use_agent and use_agent != "none":
            return self._run_agent(
                query,
                messages,
                use_agent,
                tools,
                temperature,
                max_tokens,
                system_prompt=system_prompt,
                operator_id=effective_operator_id,
                prior_messages=prior_messages,
                capability_policy=execution_policy,
            )

        result = s.engine.generate(
            messages,
            model=s.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": result.get("content", ""),
            "usage": result.get("usage", {}),
            "model": s.model,
            "engine": s.engine_key,
        }

    def _detect_agent_intent(self, query: str) -> Optional[str]:
        """Detect if a query should be routed to a specific agent."""
        import re

        from openjarvis.core.registry import AgentRegistry

        if re.search(
            r"\b(good\s+morning|morning\s+digest|daily\s+briefing|morning\s+briefing)\b",
            query,
            re.IGNORECASE,
        ):
            if AgentRegistry.contains("morning_digest"):
                return "morning_digest"

        return None

    def _run_agent(
        self,
        query,
        messages,
        agent_name,
        tool_names,
        temperature,
        max_tokens,
        *,
        system_prompt=None,
        operator_id=None,
        prior_messages=None,
        capability_policy=None,
    ) -> Dict[str, Any]:
        """Run through an agent."""
        from openjarvis.agents._stubs import AgentContext
        from openjarvis.core.events import EventType
        from openjarvis.core.registry import AgentRegistry

        s = self._system
        # ``operator_id is None`` means an in-process trusted invocation that
        # may use the configured agent principal for backwards compatibility.
        # Once a channel/API caller supplies the field, even an invalid or
        # empty value must never fall back to the agent's service identity.
        effective_operator_id = _normalize_operator_id(operator_id)

        try:
            agent_cls = AgentRegistry.get(agent_name)
        except KeyError:
            return {"content": f"Unknown agent: {agent_name}", "error": True}

        agent_tools = s.tools
        # ``None`` means use the system's configured tool set; an explicit
        # empty list means no tools. Treating both as false would silently
        # widen a caller's requested authority.
        if tool_names is not None:
            agent_tools = self._build_tools(tool_names)

        ctx = AgentContext()

        if prior_messages:
            for msg in prior_messages:
                ctx.conversation.add(msg)

        if messages and len(messages) > 1:
            for msg in messages[:-1]:
                ctx.conversation.add(msg)

        agent_kwargs: Dict[str, Any] = {
            "bus": s.bus,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if getattr(agent_cls, "accepts_tools", False):
            agent_kwargs["tools"] = agent_tools
            agent_kwargs["max_turns"] = s.config.agent.max_turns
            examples = getattr(s, "_skill_few_shot_examples", None)
            if examples:
                agent_kwargs["skill_few_shot_examples"] = examples
        if system_prompt is not None:
            agent_kwargs["system_prompt"] = system_prompt
        if operator_id is not None:
            agent_kwargs["operator_id"] = effective_operator_id
            agent_kwargs["session_store"] = s.session_store
            agent_kwargs["memory_backend"] = s.memory_backend

        if agent_name == "morning_digest" and hasattr(s.config, "digest"):
            dc = s.config.digest
            section_sources = {}
            for sec in dc.sections:
                sc = getattr(dc, sec, None)
                if sc and hasattr(sc, "sources"):
                    section_sources[sec] = sc.sources
            agent_kwargs.update(
                {
                    "persona": dc.persona,
                    "sections": dc.sections,
                    "section_sources": section_sources,
                    "timezone": dc.timezone,
                    "voice_id": dc.voice_id,
                    "voice_speed": dc.voice_speed,
                    "tts_backend": dc.tts_backend,
                    "honorific": dc.honorific,
                }
            )
            from openjarvis.tools.digest_collect import DigestCollectTool
            from openjarvis.tools.text_to_speech import TextToSpeechTool

            digest_tools = [DigestCollectTool(), TextToSpeechTool()]
            if tool_names is not None:
                declared = set(tool_names)
                digest_tools = [
                    tool
                    for tool in digest_tools
                    if tool.spec.name in declared
                ]
            existing = agent_kwargs.get("tools", [])
            existing_names = {tool.spec.name for tool in existing}
            agent_kwargs["tools"] = list(existing) + [
                tool
                for tool in digest_tools
                if tool.spec.name not in existing_names
            ]

        # Filter compatibility kwargs using the concrete constructor
        # signature.  The previous TypeError fallback rebuilt agents without
        # tools or policy, silently turning a wiring error into fail-open
        # behaviour.
        import inspect

        constructor = inspect.signature(agent_cls.__init__)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in constructor.parameters.values()
        )
        if not accepts_kwargs:
            agent_kwargs = {
                key: value
                for key, value in agent_kwargs.items()
                if key in constructor.parameters
            }
        try:
            ag = agent_cls(s.engine, s.model, **agent_kwargs)
        except TypeError as exc:
            logger.error(
                "Agent '%s' constructor rejected secure wiring: %s",
                agent_name,
                exc,
            )
            return {
                "content": f"Agent '{agent_name}' failed secure initialization.",
                "error": True,
            }

        needs_security = bool(
            getattr(agent_cls, "accepts_tools", False)
            or getattr(agent_cls, "requires_security_context", False)
        )
        if needs_security:
            # The authenticated human/channel principal is the authority for
            # this invocation.  Binding the agent's display name here would
            # let an unprivileged sender inherit service-level grants.
            security_principal = (
                agent_name
                if effective_operator_id is None
                else effective_operator_id
            )
            try:
                from openjarvis.agents.executor import _bind_agent_security

                _bind_agent_security(
                    ag,
                    capability_policy
                    if capability_policy is not None
                    else s.capability_policy,
                    security_principal,
                    getattr(s, "boundary_guard", None),
                    agent_type=agent_name,
                )
            except Exception as exc:
                logger.error(
                    "Agent '%s' failed security binding for principal %r: %s",
                    agent_name,
                    security_principal,
                    exc,
                )
                return {
                    "content": (
                        f"Agent '{agent_name}' cannot bind a valid security "
                        "principal."
                    ),
                    "error": True,
                }

        telemetry_events: List[Dict[str, Any]] = []

        def _on_inference_end(event: Any) -> None:
            telemetry_events.append(event.data if hasattr(event, "data") else event)

        s.bus.subscribe(EventType.INFERENCE_END, _on_inference_end)

        # Check trace_store (set at build time) instead of config.traces.enabled
        # because the shared config singleton can be mutated by other SystemBuilder
        # instances (e.g. the judge backend).
        try:
            if s.trace_store is not None:
                from openjarvis.traces.collector import TraceCollector

                collector = TraceCollector(
                    ag,
                    store=s.trace_store,
                    bus=s.bus,
                )
                result = collector.run(query, context=ctx)
                s.trace_collector = collector
            else:
                result = ag.run(query, context=ctx)
        finally:
            s.bus.unsubscribe(EventType.INFERENCE_END, _on_inference_end)

        _telemetry: Dict[str, Any] = {}
        if telemetry_events:
            total_energy = sum(e.get("energy_joules", 0.0) for e in telemetry_events)
            total_latency = sum(e.get("latency", 0.0) for e in telemetry_events)
            power_vals = [
                e.get("power_watts", 0.0)
                for e in telemetry_events
                if e.get("power_watts", 0.0) > 0
            ]
            util_vals = [
                e.get("gpu_utilization_pct", 0.0)
                for e in telemetry_events
                if e.get("gpu_utilization_pct", 0.0) > 0
            ]
            throughput_vals = [
                e.get("throughput_tok_per_sec", 0.0)
                for e in telemetry_events
                if e.get("throughput_tok_per_sec", 0.0) > 0
            ]
            _telemetry = {
                "ttft": telemetry_events[0].get("ttft", 0.0),
                "energy_joules": total_energy,
                "power_watts": (
                    sum(power_vals) / len(power_vals) if power_vals else 0.0
                ),
                "gpu_utilization_pct": (
                    sum(util_vals) / len(util_vals) if util_vals else 0.0
                ),
                "throughput_tok_per_sec": (
                    sum(throughput_vals) / len(throughput_vals)
                    if throughput_vals
                    else 0.0
                ),
                "gpu_memory_used_gb": max(
                    (e.get("gpu_memory_used_gb", 0.0) for e in telemetry_events),
                    default=0.0,
                ),
                "gpu_temperature_c": max(
                    (e.get("gpu_temperature_c", 0.0) for e in telemetry_events),
                    default=0.0,
                ),
                "inference_calls": len(telemetry_events),
                "total_inference_latency": total_latency,
            }

        return {
            "content": result.content,
            "usage": getattr(result, "usage", {}),
            "tool_results": [
                {
                    "tool_name": tr.tool_name,
                    "content": tr.content,
                    "success": tr.success,
                    "arguments": tr.metadata.get("arguments", {}),
                }
                for tr in getattr(result, "tool_results", [])
            ],
            "turns": getattr(result, "turns", 1),
            "metadata": getattr(result, "metadata", {}),
            "model": s.model,
            "engine": s.engine_key,
            "_telemetry": _telemetry,
        }

    def _build_tools(self, tool_names: List[str]) -> List[BaseTool]:
        """Build tool instances from tool names."""
        from openjarvis.core.registry import ToolRegistry

        s = self._system
        tools: List[BaseTool] = []
        for name in tool_names:
            try:
                if name == "retrieval" and s.memory_backend:
                    from openjarvis.tools.retrieval import RetrievalTool

                    tools.append(RetrievalTool(s.memory_backend))
                elif name == "llm":
                    from openjarvis.tools.llm_tool import LLMTool

                    tools.append(LLMTool(s.engine, model=s.model))
                elif ToolRegistry.contains(name):
                    tool = ToolRegistry.create(name)
                    if getattr(tool, "execution_disabled_reason", ""):
                        logger.warning(
                            "Tool %r is security-disabled and was omitted",
                            name,
                        )
                        continue
                    tools.append(tool)
            except Exception as exc:
                logger.warning("Failed to build tool %r: %s", name, exc)
        return tools
