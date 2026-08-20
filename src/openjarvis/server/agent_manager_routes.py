"""FastAPI routes for the Agent Manager."""

from __future__ import annotations

import asyncio
import logging
import re as _re
import threading
from typing import Any, Dict, List, Optional, Tuple

from starlette.concurrency import run_in_threadpool

from openjarvis.agents.manager import AgentManager
from openjarvis.core.cancellation import (
    AgentCancelledError,
    CancellationToken,
    cancellation_scope,
)
from openjarvis.server.response_security import (
    project_channel_binding,
    project_managed_agent,
)
from openjarvis.server.stream_bridge import CancelableStreamingResponse

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("fastapi and pydantic are required for server routes")

logger = logging.getLogger("openjarvis.server.agent_manager")

_DEEP_RESEARCH_STREAM_TIMEOUT_SECONDS = 600.0
_DEEP_RESEARCH_TIMEOUT_CONTENT = "Error: Deep research timed out."


class CreateAgentRequest(BaseModel):
    name: str
    agent_type: str = "monitor_operative"
    config: Optional[Dict[str, Any]] = None
    template_id: Optional[str] = None


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    agent_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class CreateTaskRequest(BaseModel):
    description: str


class UpdateTaskRequest(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    findings: Optional[List[Any]] = None


class BindChannelRequest(BaseModel):
    channel_type: str
    config: Optional[Dict[str, Any]] = None
    routing_mode: str = "dedicated"


class SendMessageRequest(BaseModel):
    content: str
    mode: str = "queued"
    stream: bool = False  # SSE streaming mode


class FeedbackRequest(BaseModel):
    score: float
    reason: Optional[str] = None


_BROWSER_SUB_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_screenshot",
    "browser_extract",
    "browser_axtree",
}


def _resolve_memory_backend(config: Any) -> Any:
    """Instantiate the configured memory backend, or None if unavailable.

    Mirrors serve.py's setup so an agent tick that runs through the server
    can actually use its memory_* tools. Returns None (so the tools degrade
    gracefully) when memory context is disabled or the backend can't load.
    """
    if config is None or not getattr(config.agent, "context_from_memory", False):
        return None
    try:
        import openjarvis.tools.storage  # noqa: F401
        from openjarvis.core.registry import MemoryRegistry

        key = config.memory.default_backend
        if MemoryRegistry.contains(key):
            return MemoryRegistry.create(key, db_path=config.memory.db_path)
    except Exception:
        logger.debug("Lightweight system: memory backend init failed", exc_info=True)
    return None


class _LightweightSystem:
    """Minimal system facade for the executor — avoids rebuilding the
    full JarvisSystem (which picks a random model from Ollama)."""

    def __init__(self, engine: Any, model: str, config: Any = None):
        self.engine = engine
        self.model = model
        self.config = config
        # Wire the configured memory backend so an agent's memory_store /
        # memory_retrieve tools work when the tick runs through the server.
        # The executor injects system.memory_backend into those tools; this
        # facade previously left it None, so they reported "No memory backend
        # configured" even though the backend was configured and active.
        self.memory_backend = _resolve_memory_backend(config)


def _make_lightweight_system(
    engine: Any,
    model: str,
    config: Any = None,
) -> _LightweightSystem:
    """Build a minimal system with a fresh inference engine.

    The server's ``app.state.engine`` is heavily wrapped
    (MultiEngine -> InstrumentedEngine -> GuardrailsEngine) and can
    return empty content from background threads. Create a fresh
    engine directly (no health checks or model discovery that
    could interfere with in-flight requests).
    """
    try:
        from openjarvis.engine._discovery import get_engine

        cfg = config
        if cfg is None:
            from openjarvis.core.config import load_config

            cfg = load_config()

        pref = cfg.intelligence.preferred_engine
        key = pref or cfg.engine.default
        resolved = get_engine(cfg, key)

        if resolved is not None:
            plain_engine = resolved[1]
        else:
            from openjarvis.engine.ollama import OllamaEngine

            host = cfg.engine.ollama.host if cfg else ""
            plain_engine = OllamaEngine(host=host) if host else OllamaEngine()

        # Wrap with InstrumentedEngine so agent ticks are recorded
        # in telemetry (FLOPs, energy, cost savings).
        try:
            from openjarvis.core.events import get_event_bus
            from openjarvis.telemetry.instrumented_engine import (
                InstrumentedEngine,
            )

            plain_engine = InstrumentedEngine(
                plain_engine,
                get_event_bus(),
            )
        except Exception:
            pass  # telemetry is optional
        return _LightweightSystem(plain_engine, model, cfg)
    except Exception:
        pass
    return _LightweightSystem(engine, model, config)


def _parse_param_count(model_name: str) -> float:
    """Extract parameter count in billions from model name.

    Examples: 'qwen3.5:9b' -> 9.0, 'qwen3.5:0.8b' -> 0.8
    """
    m = _re.search(r":(\d+(?:\.\d+)?)b", model_name.lower())
    return float(m.group(1)) if m else 0.0


_CLOUD_PREFIXES = ("gpt-", "claude-", "gemini-", "o1-", "o3-", "o4-")


def _pick_recommended_model(
    model_ids: list[str],
) -> dict[str, str]:
    """Pick the second-largest local model from a list."""
    local = [m for m in model_ids if not any(m.startswith(p) for p in _CLOUD_PREFIXES)]
    if not local:
        return {
            "model": model_ids[0] if model_ids else "",
            "reason": "Only model available",
        }
    sized = sorted(local, key=_parse_param_count, reverse=True)
    if len(sized) == 1:
        return {"model": sized[0], "reason": "Only local model available"}
    pick = sized[1]  # second-largest
    params = _parse_param_count(pick)
    return {
        "model": pick,
        "reason": f"Second-largest local model ({params}B parameters)",
    }


def _ensure_registries_populated() -> None:
    """Ensure ToolRegistry and ChannelRegistry are populated.

    If the registries are empty (e.g. cleared by test fixtures) but the
    modules are already cached in sys.modules, reload the individual
    submodules to re-execute their @register decorators.
    """
    import importlib
    import sys

    from openjarvis.core.registry import ChannelRegistry, ToolRegistry

    # First, try a normal import (works if modules haven't been imported yet)
    try:
        import openjarvis.channels  # noqa: F401
    except Exception:
        pass

    try:
        import openjarvis.tools  # noqa: F401
    except Exception:
        pass

    # Also try to import browser tools (not included in openjarvis.tools.__init__)
    for _browser_mod in ("openjarvis.tools.browser", "openjarvis.tools.browser_axtree"):
        try:
            importlib.import_module(_browser_mod)
        except Exception:
            pass

    # If registries are still empty, reload individual submodules from sys.modules
    if not ChannelRegistry.keys():
        for mod_name in list(sys.modules):
            if mod_name.startswith("openjarvis.channels.") and not mod_name.endswith(
                "_stubs"
            ):
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass

    if not ToolRegistry.keys():
        for mod_name in list(sys.modules):
            if (
                mod_name.startswith("openjarvis.tools.")
                and not mod_name.endswith("_stubs")
                and not mod_name.endswith("agent_tools")
            ):
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass

    # After reloading tools, also try browser tools if still not registered
    if not any(ToolRegistry.contains(n) for n in _BROWSER_SUB_TOOLS):
        for _browser_mod in (
            "openjarvis.tools.browser",
            "openjarvis.tools.browser_axtree",
        ):
            mod = sys.modules.get(_browser_mod)
            if mod is not None:
                try:
                    importlib.reload(mod)
                except Exception:
                    pass


def build_tools_list() -> List[Dict[str, Any]]:
    """Build unified tools list from ToolRegistry + ChannelRegistry."""
    import os

    from openjarvis.core.credentials import TOOL_CREDENTIALS
    from openjarvis.core.registry import ChannelRegistry, ToolRegistry

    _ensure_registries_populated()

    items: List[Dict[str, Any]] = []

    for name, tool_cls in ToolRegistry.items():
        if name in _BROWSER_SUB_TOOLS:
            continue
        # `spec` is an instance @property on BaseTool subclasses, so
        # we have to instantiate the tool to read it. The earlier
        # implementation used getattr(tool_cls, 'spec') which returns
        # the property descriptor and crashed on spec.description,
        # silently dropping every real tool from the picker.
        try:
            spec = tool_cls().spec
        except Exception as exc:
            logger.debug("Could not instantiate tool %s: %s", name, exc)
            spec = None
        cred_keys = TOOL_CREDENTIALS.get(name, [])
        items.append(
            {
                "name": name,
                "description": spec.description if spec else "",
                "category": spec.category if spec else "",
                "source": "tool",
                "requires_credentials": len(cred_keys) > 0,
                "credential_keys": cred_keys,
                "configured": (
                    all(bool(os.environ.get(k)) for k in cred_keys)
                    if cred_keys
                    else True
                ),
            }
        )

    try:
        if any(ToolRegistry.contains(n) for n in _BROWSER_SUB_TOOLS):
            items.append(
                {
                    "name": "browser",
                    "description": (
                        "Web browser automation"
                        " (navigate, click, type, screenshot, extract)"
                    ),
                    "category": "browser",
                    "source": "tool",
                    "requires_credentials": False,
                    "credential_keys": [],
                    "configured": True,
                }
            )
    except Exception:
        pass

    try:
        for name, _cls in ChannelRegistry.items():
            cred_keys = TOOL_CREDENTIALS.get(name, [])
            items.append(
                {
                    "name": name,
                    "description": (
                        f"{name.replace('_', ' ').title()} messaging channel"
                    ),
                    "category": "communication",
                    "source": "channel",
                    "requires_credentials": len(cred_keys) > 0,
                    "credential_keys": cred_keys,
                    "configured": (
                        all(bool(os.environ.get(k)) for k in cred_keys)
                        if cred_keys
                        else True
                    ),
                }
            )
    except Exception:
        pass

    return items


def _resolve_tool_specs(
    tool_config: Any,
) -> List[Dict[str, Any]]:
    """Convert a template's ``tools`` config into OpenAI-format function specs.

    The template TOML stores tools as a list of string names (e.g.
    ``["file_read", "shell_exec"]``). Engines expect OpenAI-shaped dicts:
    ``{"type": "function", "function": {"name, description, parameters"}}``.

    Special handling:
      * Dict entries pass through as-is (allows advanced configs to
        supply fully-formed specs).
      * ``browser`` is a synthetic display-only meta-tool that expands
        to the 6 real browser sub-tools (browser_navigate, click, …).
      * Channel names (``slack``, ``gmail``, …) come from the
        ``ChannelRegistry`` and are not directly callable by the LLM —
        they're destinations for ``channel_send``. Silently skip them.
      * Unknown tool names are dropped with a warning.
    """
    if not tool_config:
        return []

    from openjarvis.core.registry import ChannelRegistry, ToolRegistry

    _ensure_registries_populated()

    def _spec_dict_for(name: str) -> Optional[Dict[str, Any]]:
        try:
            spec = ToolRegistry.get(name)().spec
        except Exception as exc:
            logger.warning(
                "Could not build spec for tool '%s' (%s) — dropping",
                name,
                exc,
            )
            return None
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    resolved: List[Dict[str, Any]] = []
    seen: set = set()

    for entry in tool_config:
        if isinstance(entry, dict):
            resolved.append(entry)
            continue
        if not isinstance(entry, str):
            continue

        # Expand the synthetic "browser" meta-tool into its sub-tools.
        if entry == "browser":
            for sub in _BROWSER_SUB_TOOLS:
                if sub in seen or not ToolRegistry.contains(sub):
                    continue
                spec_dict = _spec_dict_for(sub)
                if spec_dict:
                    resolved.append(spec_dict)
                    seen.add(sub)
            continue

        # Channels (slack, gmail, …) live in ChannelRegistry and aren't
        # callable by the LLM. Skip silently — the agent talks to them
        # through the `channel_send` tool with a `channel` argument.
        if ChannelRegistry.contains(entry):
            continue

        if not ToolRegistry.contains(entry):
            logger.warning(
                "Tool '%s' referenced in agent config but not in ToolRegistry",
                entry,
            )
            continue

        if entry in seen:
            continue
        spec_dict = _spec_dict_for(entry)
        if spec_dict:
            resolved.append(spec_dict)
            seen.add(entry)

    return resolved


# Per-agent sampler params forwarded to the engine when present in config.
# The OpenAI-compat engine passes **kwargs straight through to the upstream
# payload, so these reach local servers (vLLM / mlx_lm / etc.) that support
# them. Only forwarded when explicitly set, so default agents send nothing
# extra and engines that don't support a key never receive it. (#386)
_SAMPLER_PARAM_KEYS = (
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
)


def _build_managed_system_prompt(system_prompt: str, app_config: Any) -> str:
    """Build the streaming managed-agent system prompt via SystemPromptBuilder.

    Routes the agent's own ``system_prompt`` through the same builder the
    CLI/ask path uses, so SOUL.md / MEMORY.md / USER.md persona files are
    injected for streaming chat too (#431). Returns the assembled prompt
    (caller decides whether to append a SYSTEM message); an agent with
    neither persona nor template yields an empty string, preserving the
    prior no-SYSTEM-message behavior.
    """
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template=system_prompt or "",
        memory_files_config=getattr(app_config, "memory_files", None),
        system_prompt_config=getattr(app_config, "system_prompt", None),
    )
    return builder.build()


def _sampler_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract per-agent sampler params from a managed agent's config (#386)."""
    out: Dict[str, Any] = {}
    for key in _SAMPLER_PARAM_KEYS:
        val = config.get(key)
        if val is not None:
            out[key] = val
    return out


def _replay_history_messages(
    history: List[Dict[str, Any]],
    exclude_id: str,
) -> List[Any]:
    """Rebuild prior-turn LLM messages from stored managed-agent history.

    When a stored assistant turn recorded ``tool_calls``, replay them as an
    assistant tool-use message followed by the corresponding tool-result
    messages — so the model sees its own prior tool pattern instead of
    regressing to fabricated tool output on later turns. Without this, only
    the assistant's text is replayed and the tool-use signal is lost (#382).
    """
    from openjarvis.core.types import Message, Role, ToolCall

    messages: List[Any] = []
    for m in reversed(history):
        if m.get("id") == exclude_id:
            continue
        direction = m.get("direction")
        if direction == "user_to_agent":
            messages.append(Message(role=Role.USER, content=m.get("content") or ""))
        elif direction == "agent_to_user":
            stored = m.get("tool_calls")
            if stored:
                calls = []
                results = []
                for i, tc in enumerate(stored):
                    call_id = f"hist-{m.get('id', '')}-{i}"
                    calls.append(
                        ToolCall(
                            id=call_id,
                            name=tc.get("tool", ""),
                            arguments=tc.get("arguments") or "",
                        )
                    )
                    results.append(
                        Message(
                            role=Role.TOOL,
                            content=str(tc.get("result", "")),
                            tool_call_id=call_id,
                            name=tc.get("tool", ""),
                        )
                    )
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=m.get("content") or None,
                        tool_calls=calls,
                    )
                )
                messages.extend(results)
            else:
                messages.append(
                    Message(role=Role.ASSISTANT, content=m.get("content") or "")
                )
    return messages


def _instantiate_managed_tool(
    tool_cls: Any,
    name: str,
    *,
    engine: Any,
    model: str,
    app_state: Any,
) -> Any:
    """Instantiate a tool with the same dependency injection as the canonical
    ``cli/ask.py`` path, so ``memory_*`` / ``channel_*`` / ``llm`` tools work
    instead of silently failing with "No backend configured" (#395).
    """
    try:
        from openjarvis.cli.ask import (
            _CHANNEL_TOOLS,
            _MEMORY_TOOLS,
            _get_memory_backend,
        )
    except Exception:  # pragma: no cover - cli import should always succeed
        _MEMORY_TOOLS = frozenset()
        _CHANNEL_TOOLS = frozenset()
        _get_memory_backend = None

    if name in _MEMORY_TOOLS:
        backend = getattr(app_state, "memory_backend", None) if app_state else None
        if backend is None and app_state is not None and _get_memory_backend:
            cfg = getattr(app_state, "config", None)
            if cfg is not None:
                backend = _get_memory_backend(cfg)
        if backend is None:
            logger.warning(
                "Memory tool %r instantiated without a backend — calls will "
                "return no results.",
                name,
            )
        return tool_cls(backend=backend)
    if name in _CHANNEL_TOOLS:
        channel = getattr(app_state, "channel_bridge", None) if app_state else None
        if channel is None:
            logger.warning(
                "Channel tool %r instantiated without a channel — calls will "
                "fail with 'No channel backend configured'.",
                name,
            )
        return tool_cls(channel=channel)
    if name == "llm":
        return tool_cls(engine=engine, model=model)
    return tool_cls()


def _build_deep_research_tools(
    engine: Any,
    model: str,
    knowledge_db_path: str = "",
) -> list:
    """Build the 4 DeepResearch tools from a KnowledgeStore.

    Returns an empty list if the knowledge DB does not exist.
    """
    from pathlib import Path

    if not knowledge_db_path:
        from openjarvis.core.config import DEFAULT_CONFIG_DIR

        knowledge_db_path = str(DEFAULT_CONFIG_DIR / "knowledge.db")

    if not Path(knowledge_db_path).exists():
        return []

    from openjarvis.connectors.retriever import TwoStageRetriever
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.tools.knowledge_search import KnowledgeSearchTool
    from openjarvis.tools.knowledge_sql import KnowledgeSQLTool
    from openjarvis.tools.scan_chunks import ScanChunksTool
    from openjarvis.tools.think import ThinkTool

    store = KnowledgeStore(knowledge_db_path)
    retriever = TwoStageRetriever(store)
    return [
        KnowledgeSearchTool(retriever=retriever),
        KnowledgeSQLTool(store=store),
        ScanChunksTool(store=store, engine=engine, model=model),
        ThinkTool(),
    ]


def _merge_tool_call_fragments(
    accumulated: Dict[int, Dict[str, Any]],
    fragments: List[Dict[str, Any]],
) -> None:
    """Merge incremental tool_call delta fragments into accumulated state.

    OpenAI-compatible APIs send tool_calls as incremental fragments keyed
    by ``index``. Each fragment may contain partial ``function.name`` and/or
    ``function.arguments`` strings that must be concatenated.
    """
    for frag in fragments:
        idx = frag.get("index", 0)
        if idx not in accumulated:
            accumulated[idx] = {
                "id": frag.get("id", ""),
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        entry = accumulated[idx]
        if frag.get("id"):
            entry["id"] = frag["id"]
        fn = frag.get("function", {})
        if fn.get("name"):
            entry["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]


def _get_mcp_tools(app_state: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (openai_tools_list, mcp_adapters_by_name).

    Lazily discovers MCP tools from config and caches them on ``app_state``
    so that subsequent requests reuse the same connections.
    """
    cached = getattr(app_state, "_mcp_tools_cache", None)
    if cached is not None:
        return cached

    from openjarvis.core.config import load_config

    openai_tools: List[Dict[str, Any]] = []
    adapters_by_name: Dict[str, Any] = {}

    try:
        app_config = load_config()
    except Exception as exc:
        logger.warning("Failed to load config for MCP discovery: %s", exc)
        return openai_tools, adapters_by_name

    if not app_config.tools.mcp.enabled or not app_config.tools.mcp.servers:
        return openai_tools, adapters_by_name

    try:
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        discovered, mcp_clients = load_mcp_tools_from_config(app_config.tools.mcp)
    except Exception as exc:
        # Cache the failure as an empty, non-executable surface so repeated
        # requests cannot observe a partially loaded/ambiguous tool set.
        logger.error(
            "MCP discovery rejected: %s",
            type(exc).__name__,
        )
        app_state._mcp_tools_cache = (openai_tools, adapters_by_name)
        return openai_tools, adapters_by_name

    for adapter in discovered:
        spec = adapter.spec
        if spec.name in adapters_by_name:
            raise RuntimeError(f"Duplicate MCP tool name: {spec.name}")
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
        )
        adapters_by_name[spec.name] = adapter

    app_state._mcp_clients = mcp_clients
    app_state._mcp_tools_cache = (openai_tools, adapters_by_name)
    return openai_tools, adapters_by_name


def _sse_chunk(chunk_id: str, model: str, content: str) -> str:
    """Build a single SSE content chunk."""
    import json as _json

    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {_json.dumps(data)}\n\n"


def _tool_progress_label(tool_name: str, args: str) -> str:
    """Human-readable label for a tool call in progress."""
    labels = {
        "knowledge_search": "Searching your knowledge base",
        "knowledge_sql": "Querying data with SQL",
        "scan_chunks": "Scanning documents for semantic matches",
        "think": "Planning next step",
    }
    label = labels.get(tool_name, f"Using {tool_name}")
    if args and tool_name != "think":
        # Try to extract the query/question from args JSON
        try:
            import json as _json

            parsed = _json.loads(args)
            q = parsed.get("query") or parsed.get("question") or ""
            if q:
                label += f' — "{q[:50]}"'
        except Exception:
            pass
    return label


def _build_server_tool_executor(
    *,
    tools: List[Any],
    bus: Any,
    app_state: Any,
    agent_id: str,
) -> Any:
    """Build a non-interactive, identity-bound executor for server requests."""
    from openjarvis.tools._stubs import ToolExecutor

    return ToolExecutor(
        tools=tools,
        bus=bus,
        capability_policy=getattr(app_state, "capability_policy", None),
        agent_id=agent_id,
        boundary_guard=getattr(app_state, "boundary_guard", None),
        interactive=False,
    )


async def _stream_managed_agent(
    *,
    manager: AgentManager,
    agent_record: Dict[str, Any],
    user_content: str,
    message_id: str,
    engine: Any,
    bus: Any,
    app_state: Any = None,
    operator_id: str,
) -> StreamingResponse:
    """Run a managed agent with real LLM token streaming via SSE.

    Uses ``engine.stream_full()`` to yield tokens as they arrive from the
    LLM. Supports multi-turn tool-calling: when the model emits tool_calls,
    they are executed and the results fed back for the next turn.
    """
    import json
    import uuid

    from openjarvis.core.types import Message, Role

    agent_id = agent_record["id"]
    security_principal = operator_id.strip()
    if not security_principal:
        raise RuntimeError("Managed-agent streaming requires an operator identity")
    config = agent_record.get("config", {})
    # Resolve the model: prefer the agent's own config, then the server's
    # resolved model (app.state.model — what the engine was booted with),
    # and only then the legacy engine._model attr. OllamaEngine takes the
    # model per-call and exposes no _model attr, so without the app_state
    # fallback this resolved to "" and Ollama 400'd on an empty model.
    model = (
        config.get("model")
        or getattr(app_state, "model", None)
        or getattr(engine, "_model", "")
    )
    system_prompt = config.get("system_prompt")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 1024)
    max_turns = config.get("max_turns", 10)

    # Build conversation messages from history + current input
    llm_messages: List[Message] = []

    # Wire the SystemPromptBuilder to inject SOUL.md / MEMORY.md / USER.md
    # persona files (parity with the CLI/ask path) — see #431.
    app_config = getattr(app_state, "config", None)
    if app_config is None:
        from openjarvis.core.config import load_config

        app_config = load_config()
    capability_policy = getattr(app_state, "capability_policy", None)

    final_system_prompt = _build_managed_system_prompt(system_prompt or "", app_config)

    if final_system_prompt and final_system_prompt.strip():
        llm_messages.append(
            Message(role=Role.SYSTEM, content=final_system_prompt.strip())
        )

    # Resolve agent type and class for DeepResearch tool wiring
    agent_type = agent_record.get("agent_type", "")
    if agent_type == "deep_research":
        dr_tools = _build_deep_research_tools(
            engine=engine,
            model=model,
        )
        # Store on app_state so streaming loop can access them
        if app_state is not None and dr_tools:
            app_state._dr_tools = dr_tools

    # Load prior conversation context (DESC order, reverse for chronological).
    # Replaying recorded tool_calls (assistant tool-use + tool results) keeps
    # multi-turn tool behaviour from regressing to fabricated output (#382).
    history = manager.list_messages(agent_id, limit=50)
    llm_messages.extend(_replay_history_messages(history, message_id))

    # Append the current user message
    llm_messages.append(Message(role=Role.USER, content=user_content))

    # Mark the user message as delivered
    manager.mark_message_delivered(message_id)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    cancellation_token = CancellationToken()

    # For deep_research agents: run the full agent loop, not raw streaming
    if agent_type == "deep_research" and app_state is not None:
        dr_tools = getattr(app_state, "_dr_tools", None)
        if dr_tools:
            deep_research_lock = threading.Lock()
            deep_research_state: Dict[str, Any] = {
                "closed": False,
                "persisted": False,
                "content": None,
                "tool_calls": [],
                "in_flight": 0,
                "worker_settled": False,
                "timed_out": False,
                "terminal_event": None,
            }

            def _persist_deep_research(*, mark_closed: bool = False) -> None:
                """Persist the best completed state once the response closes.

                The worker may finish a tool or final answer concurrently with
                an ASGI send failure. Keeping the state outside the async
                generator lets the response-level close hook save whichever
                completed records exist without starting a second agent turn.
                """
                with deep_research_lock:
                    if mark_closed:
                        deep_research_state["closed"] = True
                    if (
                        not deep_research_state["closed"]
                        or deep_research_state["persisted"]
                        or deep_research_state["in_flight"] > 0
                        or not deep_research_state["worker_settled"]
                    ):
                        return

                    content = deep_research_state["content"]
                    tool_calls = list(deep_research_state["tool_calls"])
                    if content is None:
                        if tool_calls:
                            # A timed-out/cancelled agent may still have
                            # completed an externally visible tool effect. Keep
                            # that replay record without storing timeout text as
                            # though it were the agent's final answer.
                            content = ""
                        elif deep_research_state["timed_out"]:
                            content = _DEEP_RESEARCH_TIMEOUT_CONTENT
                        else:
                            return

                    try:
                        manager.store_agent_response(
                            agent_id,
                            content or "",
                            tool_calls=tool_calls or None,
                        )
                    except Exception as persist_exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to persist deep-research response: %s",
                            persist_exc,
                        )
                    else:
                        deep_research_state["persisted"] = True

            def _record_deep_research_terminal(
                event: Dict[str, Any],
            ) -> None:
                with deep_research_lock:
                    deep_research_state["content"] = event["content"]
                    deep_research_state["terminal_event"] = dict(event)

            def _deep_research_terminal_snapshot() -> Dict[str, Any] | None:
                with deep_research_lock:
                    terminal = deep_research_state["terminal_event"]
                    return dict(terminal) if terminal is not None else None

            def _claim_deep_research_timeout() -> Dict[str, Any] | None:
                with deep_research_lock:
                    terminal = deep_research_state["terminal_event"]
                    if terminal is not None:
                        return dict(terminal)
                    deep_research_state["timed_out"] = True
                    return None

            def _begin_deep_research_tool() -> None:
                with deep_research_lock:
                    deep_research_state["in_flight"] += 1

            def _finish_deep_research_tool() -> None:
                with deep_research_lock:
                    deep_research_state["in_flight"] -= 1
                _persist_deep_research()

            def _settle_deep_research_worker() -> None:
                with deep_research_lock:
                    deep_research_state["worker_settled"] = True
                _persist_deep_research()

            async def generate_deep_research():
                """Run DeepResearchAgent in thread, stream progress + result."""
                import time as _dr_time

                from openjarvis.agents.deep_research import DeepResearchAgent

                loop = asyncio.get_running_loop()
                progress_q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

                def _enqueue_progress(event: Dict[str, Any]) -> None:
                    if cancellation_token.is_cancelled or loop.is_closed():
                        return
                    try:
                        loop.call_soon_threadsafe(progress_q.put_nowait, event)
                    except RuntimeError:
                        return

                # Log query start
                _dr_start = _dr_time.time()
                try:
                    manager.add_learning_log(
                        agent_id,
                        "query_start",
                        f"Query: {user_content[:100]}",
                        {"full_query": user_content},
                    )
                except Exception as _log_exc:
                    logger.warning(
                        "Failed to log query_start: %s",
                        _log_exc,
                    )

                # Patch the agent's tool executor to emit progress
                dr_agent = DeepResearchAgent(
                    engine=engine,
                    model=model,
                    tools=dr_tools,
                    max_turns=int(config.get("max_turns", 8)),
                    temperature=float(config.get("temperature", 0.3)),
                    interactive=False,
                )
                dr_agent.bind_security(
                    capability_policy,
                    security_principal,
                    getattr(app_state, "boundary_guard", None),
                )

                # Wrap the executor to capture tool calls
                original_execute = dr_agent._executor.execute

                def _tracked_execute(tc):
                    cancellation_token.raise_if_cancelled()
                    tool_name = tc.name
                    full_args = tc.arguments or ""
                    args_str = full_args[:80]
                    # Log tool call start
                    try:
                        manager.add_learning_log(
                            agent_id,
                            "tool_call",
                            f"Calling {tool_name}: {args_str}",
                            {"tool": tool_name, "arguments": full_args},
                        )
                    except Exception as _tc_exc:
                        logger.warning("Log tool_call failed: %s", _tc_exc)

                    _enqueue_progress(
                        {
                            "type": "tool_start",
                            "tool": tool_name,
                            "args": args_str,
                            "full_args": full_args,
                        }
                    )
                    _tool_start = _dr_time.monotonic()
                    _begin_deep_research_tool()
                    try:
                        cancellation_token.raise_if_cancelled()
                        result = original_execute(tc)
                        _tool_latency_ms = (_dr_time.monotonic() - _tool_start) * 1000

                        # Log tool result
                        try:
                            _ok = "succeeded" if result.success else "failed"
                            _clen = len(result.content) if result.content else 0
                            manager.add_learning_log(
                                agent_id,
                                "tool_result",
                                f"{tool_name} {_ok} ({_clen} chars)",
                                {
                                    "tool": tool_name,
                                    "success": result.success,
                                    "output_length": _clen,
                                },
                            )
                        except Exception as _tr_exc:
                            logger.warning("Log tool_result failed: %s", _tr_exc)

                        completed_tool_call = {
                            "tool": tool_name,
                            "arguments": full_args,
                            "result": result.content or "",
                            "success": bool(result.success),
                            "latency": float(_tool_latency_ms),
                        }
                        with deep_research_lock:
                            deep_research_state["tool_calls"].append(
                                completed_tool_call
                            )

                        _enqueue_progress({"type": "tool_end", **completed_tool_call})
                        return result
                    finally:
                        _finish_deep_research_tool()

                dr_agent._executor.execute = _tracked_execute

                def _log_deep_research_cancelled() -> None:
                    elapsed = _dr_time.time() - _dr_start
                    try:
                        manager.add_learning_log(
                            agent_id,
                            "query_cancelled",
                            f"Cancelled after {elapsed:.1f}s",
                            {
                                "elapsed_seconds": round(elapsed, 2),
                                "cancelled": True,
                            },
                        )
                    except Exception as cancel_log_exc:
                        logger.warning(
                            "Cancellation log failed: %s",
                            cancel_log_exc,
                        )

                def _run_agent():
                    agent_metadata = {}
                    try:
                        try:
                            with cancellation_scope(cancellation_token):
                                cancellation_token.raise_if_cancelled()
                                result = dr_agent.run(user_content)
                            content = result.content or "No results found."
                            agent_metadata = result.metadata or {}
                        except AgentCancelledError:
                            raise
                        except Exception as exc:
                            if cancellation_token.is_cancelled:
                                cancellation_token.raise_if_cancelled()
                            content = f"Error: {exc}"

                        elapsed = _dr_time.time() - _dr_start
                        terminal_event = {
                            "type": (
                                "error" if content.startswith("Error:") else "done"
                            ),
                            "content": content,
                            "metadata": agent_metadata,
                            "elapsed": elapsed,
                        }
                        # Publish terminal state before scheduling its queue
                        # callback. The deadline consumer can now observe a
                        # completed result even if call_soon_threadsafe has not
                        # run yet.
                        _record_deep_research_terminal(terminal_event)
                        cancellation_token.raise_if_cancelled()

                        # Log BEFORE queue put (put triggers SSE end)
                        try:
                            is_err = content.startswith("Error:")
                            manager.add_learning_log(
                                agent_id,
                                "query_error" if is_err else "query_complete",
                                f"{'Error' if is_err else 'Response'}: "
                                f"{len(content)} chars in {elapsed:.1f}s",
                                {
                                    "response_length": len(content),
                                    "elapsed_seconds": round(elapsed, 2),
                                },
                            )
                        except Exception as _qc_exc:
                            logger.warning(
                                "Log failed: %s",
                                _qc_exc,
                            )

                        _enqueue_progress(terminal_event)
                    except AgentCancelledError:
                        _log_deep_research_cancelled()
                    finally:
                        _settle_deep_research_worker()

                def _run_agent_safely() -> None:
                    try:
                        _run_agent()
                    except Exception as worker_exc:  # noqa: BLE001
                        # Raw daemon threads have no Future whose exception can
                        # be retrieved. Consume any unexpected wrapper failure
                        # here so shutdown never emits an unhandled traceback.
                        logger.exception(
                            "Deep-research worker failed during teardown: %s",
                            worker_exc,
                        )

                # An executor-backed ``asyncio.to_thread`` worker can keep the
                # interpreter's default executor alive after its Task is
                # cancelled. This request-scoped daemon is cooperatively
                # cancelled by the token and cannot block server shutdown.
                worker_thread = threading.Thread(
                    target=_run_agent_safely,
                    name=f"deep-research-{agent_id}",
                    daemon=True,
                )
                worker_thread.start()

                deadline = loop.time() + _DEEP_RESEARCH_STREAM_TIMEOUT_SECONDS

                # Stream progress events and final content
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        event = None
                    else:
                        try:
                            event = await asyncio.wait_for(
                                progress_q.get(),
                                timeout=remaining,
                            )
                        except asyncio.TimeoutError:
                            event = None

                    if event is None:
                        # The worker publishes terminal state synchronously
                        # before scheduling its queue callback. This snapshot
                        # therefore wins even when call_soon_threadsafe is
                        # still pending at the deadline boundary.
                        event = _deep_research_terminal_snapshot()

                    if event is None:
                        try:
                            event = progress_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                    if event is None:
                        # Close the final snapshot→timeout race under the same
                        # lock used by the worker's terminal publication.
                        event = _claim_deep_research_timeout()

                    if event is None:
                        cancellation_token.cancel()
                        timeout_content = _DEEP_RESEARCH_TIMEOUT_CONTENT
                        elapsed = _dr_time.time() - _dr_start
                        try:
                            manager.add_learning_log(
                                agent_id,
                                "query_error",
                                f"Deep research timed out after {elapsed:.1f}s",
                                {
                                    "elapsed_seconds": round(elapsed, 2),
                                    "timeout": True,
                                },
                            )
                        except Exception as timeout_log_exc:
                            logger.warning(
                                "Deep-research timeout log failed: %s",
                                timeout_log_exc,
                            )
                        event = {
                            "type": "error",
                            "content": timeout_content,
                            "metadata": {},
                            "elapsed": elapsed,
                        }

                    if event["type"] == "tool_start":
                        tool = event["tool"]
                        args = event.get("args", "")
                        full_args = event.get("full_args", "")
                        # Structured event so the UI can render a tool_call
                        # message card (same shape as the non-DR path).
                        _start_payload = json.dumps(
                            {"tool": tool, "arguments": full_args}
                        )
                        yield f"event: tool_call_start\ndata: {_start_payload}\n\n"
                        # Keep the human-readable progress label for the
                        # thinking-bubble fallback.
                        label = _tool_progress_label(tool, args)
                        progress_data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None,
                                    "tool_progress": label,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(progress_data)}\n\n"

                    elif event["type"] == "tool_end":
                        tool = event["tool"]
                        _end_payload = json.dumps(
                            {
                                "tool": tool,
                                "success": bool(event.get("success", False)),
                                "latency": float(event.get("latency", 0.0)),
                                "result": event.get("result", ""),
                            }
                        )
                        yield f"event: tool_call_end\ndata: {_end_payload}\n\n"

                    elif event["type"] in ("done", "error"):
                        content = event["content"]
                        meta = event.get("metadata", {})
                        elapsed_s = event.get("elapsed", 0)

                        # Stream content word-by-word
                        words = content.split(" ")
                        for i, word in enumerate(words):
                            token = word if i == 0 else " " + word
                            yield _sse_chunk(chunk_id, model, token)

                        # Build usage + telemetry
                        prompt_tok = meta.get("prompt_tokens", 0)
                        comp_tok = meta.get("completion_tokens", 0)
                        total_tok = meta.get("total_tokens", 0)
                        word_count = len(words)
                        speed = round(word_count / elapsed_s) if elapsed_s > 0 else 0

                        # Final chunk with usage + telemetry
                        finish_data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tok,
                                "completion_tokens": comp_tok,
                                "total_tokens": total_tok or (prompt_tok + comp_tok),
                            },
                            "telemetry": {
                                "engine": "ollama",
                                "model_id": model,
                                "total_ms": round(elapsed_s * 1000),
                                "tokens_per_sec": speed,
                                "tool_calls": len(meta.get("sources", [])),
                            },
                        }
                        yield f"data: {json.dumps(finish_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

            def _close_deep_research() -> None:
                _persist_deep_research(mark_closed=True)

            return CancelableStreamingResponse(
                generate_deep_research(),
                cancel=cancellation_token.cancel,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
                on_close=_close_deep_research,
            )

    # Build extra kwargs for stream_full (e.g. tools from config).
    # Template stores tool names as strings; convert to OpenAI function specs
    # so the engine can actually bind them to the model.
    stream_kwargs: Dict[str, Any] = {}
    resolved_tools = _resolve_tool_specs(config.get("tools"))
    if resolved_tools:
        stream_kwargs["tools"] = resolved_tools

    # Forward any per-agent sampler params (repetition_penalty, top_p, …) so
    # locally-hosted models can be tuned per agent (#386).
    stream_kwargs.update(_sampler_kwargs(config))

    # Discover MCP tools and merge into stream_kwargs
    mcp_adapters: Dict[str, Any] = {}
    if app_state is not None:
        try:
            mcp_openai_tools, mcp_adapters = _get_mcp_tools(app_state)
            if mcp_openai_tools:
                existing_tools = stream_kwargs.get("tools", [])
                stream_kwargs["tools"] = existing_tools + mcp_openai_tools
                logger.info(
                    "Added %d MCP tools to streaming request",
                    len(mcp_openai_tools),
                )
        except Exception as exc:
            logger.warning(
                "Failed to get MCP tools for streaming: %s", exc, exc_info=True
            )

    # Shared state between the generator and its idempotent close callback.
    # Starlette skips BackgroundTask after an ASGI 2.4 send failure, so the
    # cancelable response invokes the same persistence function while closing.
    persist_lock = threading.Lock()
    persist_state: Dict[str, Any] = {
        "content": "",
        "tool_calls": [],
        "closed": False,
        "in_flight": 0,
        "stored": False,
        "logged": False,
        "persisting": False,
    }

    def _set_persist_content(content: str) -> None:
        with persist_lock:
            persist_state["content"] = content

    def _record_persisted_tool_call(tool_call: Dict[str, Any]) -> None:
        with persist_lock:
            persist_state["tool_calls"].append(tool_call)

    def _begin_in_flight_tool() -> None:
        with persist_lock:
            persist_state["in_flight"] += 1

    def _persist_final(*, mark_closed: bool = False) -> None:
        with persist_lock:
            if mark_closed:
                persist_state["closed"] = True
            if (
                not persist_state["closed"]
                or persist_state["in_flight"] > 0
                or persist_state["persisting"]
            ):
                return

            content = persist_state["content"] or ""
            tool_calls = list(persist_state["tool_calls"])
            if not content and not tool_calls:
                return
            if persist_state["stored"] and persist_state["logged"]:
                return

            should_store = not persist_state["stored"]
            should_log = not persist_state["logged"]
            persist_state["persisting"] = True

        if should_store:
            store_succeeded = False
            for attempt in range(2):
                try:
                    manager.store_agent_response(
                        agent_id,
                        content,
                        tool_calls=tool_calls or None,
                    )
                except Exception as store_exc:
                    logger.error(
                        "Failed to store agent response (attempt %d/2): %s",
                        attempt + 1,
                        store_exc,
                        exc_info=True,
                    )
                else:
                    store_succeeded = True
                    break

            if not store_succeeded:
                with persist_lock:
                    persist_state["persisting"] = False
                return
            with persist_lock:
                persist_state["stored"] = True

        if should_log:
            try:
                manager.add_learning_log(
                    agent_id,
                    "query_complete",
                    f"Response: {len(content)} chars, {len(tool_calls)} tool calls",
                    {
                        "response_length": len(content),
                        "tool_calls": len(tool_calls),
                    },
                )
            except Exception as log_exc:
                logger.warning("Log query_complete failed: %s", log_exc)
            else:
                with persist_lock:
                    persist_state["logged"] = True

        with persist_lock:
            persist_state["persisting"] = False

    def _finish_in_flight_tool() -> None:
        with persist_lock:
            persist_state["in_flight"] -= 1
        _persist_final()

    async def generate():
        """Async generator yielding SSE-formatted chunks with real token streaming."""

        cancellation_token.raise_if_cancelled()
        collected_content = ""
        collected_tool_calls: List[Dict[str, Any]] = []
        messages_for_llm = list(llm_messages)
        turns = 0

        import time as _lgtime

        _query_start_ts = _lgtime.time()
        try:
            manager.add_learning_log(
                agent_id,
                "query_start",
                f"Query: {user_content[:100]}",
                {"full_query": user_content},
            )
        except Exception as _qs_exc:
            logger.warning("Log query_start failed: %s", _qs_exc)

        while turns < max_turns:
            cancellation_token.raise_if_cancelled()
            turns += 1
            turn_content = ""
            tool_call_fragments: Dict[int, Dict[str, Any]] = {}
            current_finish_reason = None

            try:
                cancellation_token.raise_if_cancelled()
                provider_stream = engine.stream_full(
                    messages_for_llm,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **stream_kwargs,
                )
                try:
                    async for chunk in provider_stream:
                        cancellation_token.raise_if_cancelled()
                        # Stream content tokens immediately to the client
                        if chunk.content:
                            turn_content += chunk.content
                            # Mirror partial content so a disconnect during
                            # generation still saves what we've produced.
                            _set_persist_content(collected_content + turn_content)
                            chunk_data = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk.content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"

                        # Accumulate tool_call fragments
                        if chunk.tool_calls:
                            _merge_tool_call_fragments(
                                tool_call_fragments,
                                chunk.tool_calls,
                            )

                        if chunk.finish_reason:
                            current_finish_reason = chunk.finish_reason
                finally:
                    close_provider = getattr(provider_stream, "aclose", None)
                    if close_provider is not None:
                        try:
                            await close_provider()
                        except Exception as close_exc:
                            logger.debug(
                                "Managed provider stream close failed: %s",
                                close_exc,
                                exc_info=True,
                            )

                cancellation_token.raise_if_cancelled()
            except AgentCancelledError:
                raise
            except Exception as exc:
                logger.error("Managed agent stream error: %s", exc, exc_info=True)
                error_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"Error: {exc}"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Handle tool calls: execute tools and loop for next turn
            if tool_call_fragments and current_finish_reason == "tool_calls":
                # Build the assistant message with tool_calls
                sorted_tcs = [
                    tool_call_fragments[i] for i in sorted(tool_call_fragments.keys())
                ]

                # Add assistant message with tool_calls to conversation
                from openjarvis.core.types import ToolCall as MsgToolCall

                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=turn_content or None,
                    tool_calls=[
                        MsgToolCall(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        )
                        for tc in sorted_tcs
                    ],
                )
                messages_for_llm.append(assistant_msg)

                # Execute each tool call and append results. Emit
                # tool_call_start/tool_call_end around each call so the UI
                # can render them live (same event names as the main chat
                # in stream_bridge.py).
                import time as _time

                def _execute_and_capture_tool(
                    executor: Any,
                    tool_call: Any,
                    tool_name: str,
                    tool_args: str,
                    tool_start_ms: float,
                ) -> Dict[str, Any]:
                    """Run one dispatched tool and preserve its terminal record.

                    ``asyncio`` cannot stop the underlying ``to_thread`` call
                    after a disconnect. The worker therefore owns result
                    capture and decrements ``in_flight`` only after the replay
                    record is in shared state.
                    """
                    try:
                        with cancellation_scope(cancellation_token):
                            cancellation_token.raise_if_cancelled()
                            result = executor.execute(tool_call)

                        tool_record = {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result.content,
                            "success": bool(result.success),
                            "latency": ((_time.monotonic() * 1000) - tool_start_ms),
                        }
                        _record_persisted_tool_call(tool_record)
                        # Preserve a result that completed concurrently with
                        # cancellation, then prevent any later agent turn.
                        cancellation_token.raise_if_cancelled()
                        return tool_record
                    except AgentCancelledError:
                        raise
                    except Exception as tool_exc:
                        if cancellation_token.is_cancelled:
                            cancellation_token.raise_if_cancelled()
                        logger.error(
                            "Tool execution error for %s: %s",
                            tool_name,
                            tool_exc,
                            exc_info=True,
                        )
                        tool_record = {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": (f"Error executing {tool_name}: {tool_exc}"),
                            "success": False,
                            "latency": ((_time.monotonic() * 1000) - tool_start_ms),
                        }
                        _record_persisted_tool_call(tool_record)
                        return tool_record
                    finally:
                        _finish_in_flight_tool()

                for tc in sorted_tcs:
                    cancellation_token.raise_if_cancelled()
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tool_result_content = f"Tool '{tool_name}' not available"
                    tool_succeeded = False

                    _start_payload = json.dumps(
                        {"tool": tool_name, "arguments": tool_args}
                    )
                    yield f"event: tool_call_start\ndata: {_start_payload}\n\n"
                    try:
                        manager.add_learning_log(
                            agent_id,
                            "tool_call",
                            f"Calling {tool_name}: {tool_args[:80]}",
                            {"tool": tool_name, "arguments": tool_args or ""},
                        )
                    except Exception as _tc_exc:
                        logger.warning("Log tool_call failed: %s", _tc_exc)
                    tool_start_ms = _time.monotonic() * 1000
                    tool_record: Dict[str, Any] | None = None

                    try:
                        from openjarvis.tools._stubs import (
                            ToolCall as StubToolCall,
                        )

                        tool_instance = None
                        # Try MCP adapter first (external tools).
                        mcp_adapter = mcp_adapters.get(tool_name)
                        if mcp_adapter is not None:
                            tool_instance = mcp_adapter
                        else:
                            # Try to use ToolExecutor if tools are configured
                            from openjarvis.core.registry import ToolRegistry

                            tool_cls = ToolRegistry.get(tool_name)
                            if tool_cls is not None:
                                # Inject backend / channel / engine the same
                                # way cli/ask.py does, else memory_* / channel_*
                                # / llm tools fail with "No backend configured"
                                # on every call (#395).
                                tool_instance = _instantiate_managed_tool(
                                    tool_cls,
                                    tool_name,
                                    engine=engine,
                                    model=model,
                                    app_state=app_state,
                                )
                            else:
                                logger.warning(
                                    "Tool '%s' not found in registry or MCP adapters",
                                    tool_name,
                                )

                        if tool_instance is not None:
                            # Server requests have no synchronous human
                            # approval channel.  Every local and MCP tool goes
                            # through the same fail-closed executor.
                            executor = _build_server_tool_executor(
                                tools=[tool_instance],
                                bus=bus,
                                app_state=app_state,
                                agent_id=security_principal,
                            )
                            stub_tool_call = StubToolCall(
                                id=tc["id"],
                                name=tool_name,
                                arguments=tool_args,
                            )
                            _begin_in_flight_tool()
                            try:
                                tool_worker = asyncio.create_task(
                                    asyncio.to_thread(
                                        _execute_and_capture_tool,
                                        executor,
                                        stub_tool_call,
                                        tool_name,
                                        tool_args,
                                        tool_start_ms,
                                    )
                                )
                            except Exception:
                                _finish_in_flight_tool()
                                raise

                            def _consume_tool_worker(
                                worker: asyncio.Task,
                            ) -> None:
                                if not worker.cancelled():
                                    worker.exception()

                            tool_worker.add_done_callback(_consume_tool_worker)
                            tool_record = await asyncio.shield(tool_worker)
                    except AgentCancelledError:
                        raise
                    except asyncio.CancelledError:
                        raise
                    except Exception as tool_exc:
                        logger.error(
                            "Tool execution error for %s: %s",
                            tool_name,
                            tool_exc,
                            exc_info=True,
                        )
                        tool_result_content = f"Error executing {tool_name}: {tool_exc}"

                    if tool_record is None:
                        tool_record = {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": tool_result_content,
                            "success": tool_succeeded,
                            "latency": ((_time.monotonic() * 1000) - tool_start_ms),
                        }
                        _record_persisted_tool_call(tool_record)

                    tool_result_content = str(tool_record["result"])
                    tool_succeeded = bool(tool_record["success"])
                    tool_latency_ms = float(tool_record["latency"])
                    collected_tool_calls.append(tool_record)
                    try:
                        _ok = "succeeded" if tool_succeeded else "failed"
                        _clen = len(tool_result_content) if tool_result_content else 0
                        manager.add_learning_log(
                            agent_id,
                            "tool_result",
                            f"{tool_name} {_ok} ({_clen} chars)",
                            {
                                "tool": tool_name,
                                "success": tool_succeeded,
                                "output_length": _clen,
                            },
                        )
                    except Exception as _tr_exc:
                        logger.warning("Log tool_result failed: %s", _tr_exc)
                    _end_payload = json.dumps(
                        {
                            "tool": tool_name,
                            "success": tool_succeeded,
                            "latency": tool_latency_ms,
                            "result": tool_result_content,
                        }
                    )
                    yield f"event: tool_call_end\ndata: {_end_payload}\n\n"

                    # Add tool result message to conversation
                    messages_for_llm.append(
                        Message(
                            role=Role.TOOL,
                            content=tool_result_content,
                            tool_call_id=tc["id"],
                            name=tool_name,
                        )
                    )

                # Continue to next turn (loop back to stream_full)
                collected_content += turn_content
                # Mirror to shared state so BackgroundTask can persist
                # even if the client disconnects mid-stream.
                _set_persist_content(collected_content)
                continue

            # No tool calls — this is the final response
            collected_content += turn_content
            _set_persist_content(collected_content)
            break

        # Final chunk with finish_reason
        final_data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"

    async def generate_scoped():
        """Bind the request token across provider and tool worker boundaries."""

        stream = generate()
        try:
            with cancellation_scope(cancellation_token):
                async for frame in stream:
                    cancellation_token.raise_if_cancelled()
                    yield frame
        finally:
            cancellation_token.cancel()
            await stream.aclose()

    async def _close_managed_response() -> None:
        # Mark closure before the first cancellation checkpoint. ASGI 2.3
        # disconnect handling runs inside a cancelled task group, so deferring
        # this state transition to the thread pool can lose the worker's late
        # tool result even though the I/O itself must stay off the event loop.
        with persist_lock:
            persist_state["closed"] = True
        await run_in_threadpool(_persist_final)

    from starlette.background import BackgroundTask

    return CancelableStreamingResponse(
        generate_scoped(),
        cancel=cancellation_token.cancel,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        background=BackgroundTask(_persist_final),
        on_close=_close_managed_response,
    )


def create_agent_manager_router(
    manager: AgentManager,
) -> Tuple[APIRouter, APIRouter, APIRouter, APIRouter, APIRouter]:
    """Create FastAPI routers with agent management endpoints.

    Returns a 5-tuple:
    ``(agents_router, templates_router, global_router, tools_router, sendblue_router)``.
    """
    agents_router = APIRouter(prefix="/v1/managed-agents", tags=["managed-agents"])
    templates_router = APIRouter(prefix="/v1/templates", tags=["templates"])

    # ── Agent lifecycle ──────────────────────────────────────

    @agents_router.get("")
    async def list_agents():
        return {
            "agents": [project_managed_agent(agent) for agent in manager.list_agents()]
        }

    @agents_router.post("")
    async def create_agent(req: CreateAgentRequest, request: Request):
        if req.template_id:
            agent = manager.create_from_template(
                req.template_id, req.name, overrides=req.config
            )
        else:
            agent = manager.create_agent(
                name=req.name, agent_type=req.agent_type, config=req.config
            )

        # Register with scheduler if cron/interval
        scheduler = getattr(request.app.state, "agent_scheduler", None)
        sched_type = (req.config or {}).get("schedule_type", "manual")
        if scheduler and sched_type in ("cron", "interval"):
            scheduler.register_agent(agent["id"])

        return project_managed_agent(agent)

    @agents_router.get("/{agent_id}")
    async def get_agent(agent_id: str):
        agent = manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return project_managed_agent(agent)

    @agents_router.patch("/{agent_id}")
    async def update_agent(agent_id: str, req: UpdateAgentRequest):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        kwargs: Dict[str, Any] = {}
        if req.name is not None:
            kwargs["name"] = req.name
        if req.agent_type is not None:
            kwargs["agent_type"] = req.agent_type
        if req.config is not None:
            kwargs["config"] = req.config
        return project_managed_agent(manager.update_agent(agent_id, **kwargs))

    @agents_router.delete("/{agent_id}")
    async def delete_agent(agent_id: str):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        manager.delete_agent(agent_id)
        return {"status": "archived"}

    @agents_router.post("/{agent_id}/pause")
    async def pause_agent(agent_id: str):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        manager.pause_agent(agent_id)
        return {"status": "paused"}

    @agents_router.post("/{agent_id}/resume")
    async def resume_agent(agent_id: str):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        manager.resume_agent(agent_id)
        return {"status": "idle"}

    @agents_router.post("/{agent_id}/run")
    async def run_agent(agent_id: str, request: Request):
        agent = manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent["status"] == "archived":
            raise HTTPException(status_code=400, detail="Agent is archived")
        # AgentExecutor currently binds tool/memory authority to ``agent_id``.
        # An external request principal would therefore inherit service-agent
        # grants.  Refuse this asynchronous path until the immutable operator
        # identity is carried into the tick.
        raise HTTPException(
            status_code=403,
            detail=(
                "External managed-agent runs are disabled until execution "
                "preserves the authenticated operator identity"
            ),
        )

    # ── Recover ──────────────────────────────────────────────

    @agents_router.post("/{agent_id}/recover")
    def recover_agent(agent_id: str):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        checkpoint = manager.recover_agent(agent_id)
        checkpoint_summary = None
        if isinstance(checkpoint, dict):
            checkpoint_summary = {
                key: checkpoint[key]
                for key in ("tick_id", "created_at")
                if key in checkpoint
            }
        return {"recovered": True, "checkpoint": checkpoint_summary}

    # ── Tasks ────────────────────────────────────────────────

    @agents_router.get("/{agent_id}/tasks")
    async def list_tasks(agent_id: str, status: Optional[str] = None):
        return {"tasks": manager.list_tasks(agent_id, status=status)}

    @agents_router.post("/{agent_id}/tasks")
    async def create_task(agent_id: str, req: CreateTaskRequest):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        return manager.create_task(agent_id, description=req.description)

    @agents_router.get("/{agent_id}/tasks/{task_id}")
    async def get_task(agent_id: str, task_id: str):
        task = manager._get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @agents_router.patch("/{agent_id}/tasks/{task_id}")
    async def update_task(agent_id: str, task_id: str, req: UpdateTaskRequest):
        kwargs: Dict[str, Any] = {}
        if req.description is not None:
            kwargs["description"] = req.description
        if req.status is not None:
            kwargs["status"] = req.status
        if req.progress is not None:
            kwargs["progress"] = req.progress
        if req.findings is not None:
            kwargs["findings"] = req.findings
        return manager.update_task(task_id, **kwargs)

    @agents_router.delete("/{agent_id}/tasks/{task_id}")
    async def delete_task(agent_id: str, task_id: str):
        manager.delete_task(task_id)
        return {"status": "deleted"}

    # ── Channel bindings ─────────────────────────────────────

    @agents_router.get("/{agent_id}/channels")
    async def list_channels(agent_id: str):
        return {
            "bindings": [
                project_channel_binding(binding)
                for binding in manager.list_channel_bindings(agent_id)
            ]
        }

    @agents_router.post("/{agent_id}/channels")
    async def bind_channel(
        agent_id: str,
        req: BindChannelRequest,
        request: Request,
    ):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        if req.channel_type in {"imessage", "sendblue", "slack"}:
            raise HTTPException(
                status_code=503,
                detail=(
                    "This managed channel is disabled until request-scoped "
                    "operator identity and vault-backed credentials are "
                    "preserved end to end"
                ),
            )
        channel_config = req.config or {}
        allowed_senders = channel_config.get("allowed_senders", [])
        if (
            not isinstance(allowed_senders, list)
            or not allowed_senders
            or not all(
                isinstance(sender, str) and sender.strip() for sender in allowed_senders
            )
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Channel bindings require a non-empty allowed_senders "
                    "array of exact sender IDs"
                ),
            )
        capability_policy = getattr(request.app.state, "capability_policy", None)
        binding = manager.bind_channel(
            agent_id,
            channel_type=req.channel_type,
            config=req.config,
            routing_mode=req.routing_mode,
        )

        # Start iMessage daemon if binding iMessage
        if req.channel_type == "imessage":
            identifier = (req.config or {}).get("identifier", "")
            if identifier:
                try:
                    from openjarvis.channels.imessage_daemon import (
                        is_running,
                        run_daemon,
                    )

                    if not is_running():
                        import threading

                        engine = getattr(request.app.state, "engine", None)
                        if engine:
                            from openjarvis.server.agent_manager_routes import (
                                _build_deep_research_tools,
                            )

                            tools = _build_deep_research_tools(engine=engine, model="")
                            if tools:
                                from openjarvis.agents.deep_research import (
                                    DeepResearchAgent,
                                )

                                agent_inst = DeepResearchAgent(
                                    engine=engine,
                                    model=getattr(engine, "_model", ""),
                                    tools=tools,
                                    interactive=False,
                                )
                                agent_inst.bind_security(
                                    capability_policy,
                                    agent_id,
                                    getattr(
                                        request.app.state,
                                        "boundary_guard",
                                        None,
                                    ),
                                )

                                def handler(text: str) -> str:
                                    result = agent_inst.run(text)
                                    return result.content or "No results."

                                t = threading.Thread(
                                    target=run_daemon,
                                    kwargs={
                                        "chat_identifier": identifier,
                                        "handler": handler,
                                    },
                                    daemon=True,
                                )
                                t.start()
                except Exception as exc:
                    logger.warning("Failed to start iMessage daemon: %s", exc)

        # Initialize SendBlue channel if binding sendblue
        if req.channel_type == "sendblue":
            config = req.config or {}
            api_key_id = config.get("api_key_id", "")
            api_secret_key = config.get("api_secret_key", "")
            from_number = config.get("from_number", "")
            webhook_secret = config.get("webhook_secret", "")
            if api_key_id and api_secret_key and webhook_secret:
                try:
                    from openjarvis.channels.sendblue import (
                        SendBlueChannel,
                    )

                    sb_channel = SendBlueChannel(
                        api_key_id=api_key_id,
                        api_secret_key=api_secret_key,
                        from_number=from_number,
                        webhook_secret=webhook_secret,
                    )
                    sb_channel.connect()
                    # Store on app state so webhook route can use it
                    request.app.state.sendblue_channel = sb_channel

                    # Create or update the channel bridge
                    bridge = getattr(request.app.state, "channel_bridge", None)
                    if bridge and hasattr(bridge, "_channels"):
                        bridge._channels["sendblue"] = sb_channel
                        bridge.set_sender_allowlist(
                            "sendblue",
                            allowed_senders,
                        )
                    else:
                        # Create a new ChannelBridge with DeepResearch
                        from openjarvis.server.channel_bridge import (
                            ChannelBridge,
                        )
                        from openjarvis.server.session_store import (
                            SessionStore,
                        )

                        session_store = SessionStore()
                        engine = getattr(request.app.state, "engine", None)
                        dr_agent = None
                        if engine:
                            from openjarvis.server.agent_manager_routes import (
                                _build_deep_research_tools as _bdr,
                            )

                            tools = _bdr(engine=engine, model="")
                            if tools:
                                from openjarvis.agents.deep_research import (
                                    DeepResearchAgent,
                                )

                                model_name = getattr(engine, "_model", "") or getattr(
                                    request.app.state,
                                    "model",
                                    "",
                                )
                                dr_agent = DeepResearchAgent(
                                    engine=engine,
                                    model=model_name,
                                    tools=tools,
                                    interactive=False,
                                )
                                dr_agent.bind_security(
                                    capability_policy,
                                    agent_id,
                                    getattr(
                                        request.app.state,
                                        "boundary_guard",
                                        None,
                                    ),
                                )
                        bus = getattr(request.app.state, "bus", None)
                        if bus is None:
                            from openjarvis.core.events import EventBus

                            bus = EventBus()
                        bridge = ChannelBridge(
                            channels={"sendblue": sb_channel},
                            session_store=session_store,
                            bus=bus,
                            agent_manager=manager,
                            deep_research_agent=dr_agent,
                            sender_allowlist={
                                "sendblue": allowed_senders,
                            },
                            capability_policy=capability_policy,
                        )
                        request.app.state.channel_bridge = bridge

                    logger.info(
                        "SendBlue channel connected: %s",
                        from_number,
                    )
                except Exception as exc:
                    logger.warning("Failed to init SendBlue channel: %s", exc)

        # Start Slack via slack-bolt Socket Mode
        if req.channel_type == "slack":
            config = req.config or {}
            bot_token = config.get("bot_token", "")
            app_token = config.get("app_token", "")
            if bot_token and app_token:
                try:
                    from openjarvis.channels.slack_daemon import (
                        start_slack_daemon,
                    )
                    from openjarvis.channels.slack_daemon import (
                        stop_daemon as stop_slack,
                    )

                    # Stop any existing daemon first
                    stop_slack()

                    # Spawn as subprocess (reliable)
                    srv_model = (
                        getattr(
                            getattr(
                                request.app.state,
                                "engine",
                                None,
                            ),
                            "_model",
                            "qwen3.5:9b",
                        )
                        or "qwen3.5:9b"
                    )
                    pid = start_slack_daemon(
                        bot_token=bot_token,
                        app_token=app_token,
                        model=srv_model,
                        agent_id=agent_id,
                        allowed_sender_ids=allowed_senders,
                    )
                    logger.info(
                        "Slack daemon started (PID %d)",
                        pid,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to start Slack: %s",
                        exc,
                    )

        return project_channel_binding(binding)

    @agents_router.delete("/{agent_id}/channels/{binding_id}")
    async def unbind_channel(
        agent_id: str,
        binding_id: str,
        request: Request,
    ):
        try:
            binding = manager._get_binding(binding_id)
            if binding:
                ch_type = binding.get("channel_type")
                if ch_type == "imessage":
                    from openjarvis.channels.imessage_daemon import (
                        stop_daemon,
                    )

                    stop_daemon()
                elif ch_type == "slack":
                    from openjarvis.channels.slack_daemon import (
                        stop_daemon as stop_slack_daemon,
                    )

                    stop_slack_daemon()
        except Exception:
            pass
        manager.unbind_channel(binding_id)
        return {"status": "unbound"}

    # ── Messaging ────────────────────────────────────────────

    @agents_router.get("/{agent_id}/messages")
    def list_messages(agent_id: str):
        return {"messages": manager.list_messages(agent_id)}

    @agents_router.post("/{agent_id}/messages")
    async def send_message(agent_id: str, req: SendMessageRequest, request: Request):
        from openjarvis.server.auth_middleware import explicitly_authorized

        principal = getattr(request.state, "api_principal", "")
        if not isinstance(principal, str) or not principal.strip():
            raise HTTPException(
                status_code=503,
                detail="Authenticated API principal is not configured",
            )
        if not explicitly_authorized(
            getattr(request.app.state, "capability_policy", None),
            principal,
            "tool:invoke",
            f"agent:{agent_id}",
        ):
            raise HTTPException(
                status_code=403,
                detail="API principal is not authorized to invoke this agent",
            )

        agent_record = manager.get_agent(agent_id)
        if not agent_record:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Queued and non-streaming modes lose the authenticated operator when
        # a later service-agent tick executes the message.  Until the queue
        # schema carries an immutable principal and grant snapshot, refuse
        # those modes before storing or mutating anything.
        if not req.stream:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Managed-agent API execution requires stream=true so the "
                    "authenticated operator identity remains bound"
                ),
            )

        # Streaming is executed now under the request principal.  Store it as
        # immediate even when an older client leaves mode at its queued default.
        if agent_record["status"] in (
            "error",
            "needs_attention",
        ):
            manager.update_agent(agent_id, status="idle")

        msg = manager.send_message(agent_id, req.content, mode="immediate")

        # --- Streaming mode: run agent and return SSE response ---
        engine = getattr(request.app.state, "engine", None)
        bus = getattr(request.app.state, "bus", None)
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Engine not available for streaming",
            )

        return await _stream_managed_agent(
            manager=manager,
            agent_record=agent_record,
            user_content=req.content,
            message_id=msg["id"],
            engine=engine,
            bus=bus,
            app_state=request.app.state,
            operator_id=principal,
        )

    # ── State inspection ─────────────────────────────────────

    @agents_router.get("/{agent_id}/state")
    def get_agent_state(agent_id: str):
        agent = manager.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {
            "agent": project_managed_agent(agent),
            "tasks": manager.list_tasks(agent_id),
            "channels": [
                project_channel_binding(binding)
                for binding in manager.list_channel_bindings(agent_id)
            ],
            "messages": manager.list_messages(agent_id),
            "checkpoint": (
                {
                    key: checkpoint[key]
                    for key in ("tick_id", "created_at")
                    if key in checkpoint
                }
                if isinstance(
                    (checkpoint := manager.get_latest_checkpoint(agent_id)),
                    dict,
                )
                else None
            ),
        }

    # ── Learning ─────────────────────────────────────────────

    @agents_router.get("/{agent_id}/learning")
    def get_learning_log(agent_id: str):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"learning_log": manager.list_learning_log(agent_id)}

    @agents_router.post("/{agent_id}/learning/run")
    def trigger_learning(agent_id: str):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        from openjarvis.core.events import EventType, get_event_bus

        bus = get_event_bus()
        bus.publish(EventType.AGENT_LEARNING_STARTED, {"agent_id": agent_id})
        return {"status": "triggered"}

    # ── Traces ───────────────────────────────────────────────

    @agents_router.get("/{agent_id}/traces")
    def list_traces(agent_id: str, limit: int = 20):
        if not manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        try:
            from openjarvis.core.config import load_config
            from openjarvis.core.paths import get_config_dir
            from openjarvis.traces.store import TraceStore

            config = load_config()
            store = TraceStore(
                config.traces.db_path or str(get_config_dir() / "traces.db")
            )
            traces = store.list_traces(agent=agent_id, limit=limit)
            return {
                "traces": [
                    {
                        "id": t.trace_id,
                        "outcome": t.outcome,
                        "duration": t.total_latency_seconds,
                        "started_at": t.started_at,
                        "steps": len(t.steps),
                        "metadata": t.metadata,
                    }
                    for t in traces
                ]
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @agents_router.get("/{agent_id}/traces/{trace_id}")
    def get_trace(agent_id: str, trace_id: str):
        try:
            from openjarvis.core.config import load_config
            from openjarvis.core.paths import get_config_dir
            from openjarvis.traces.store import TraceStore

            config = load_config()
            store = TraceStore(
                config.traces.db_path or str(get_config_dir() / "traces.db")
            )
            trace = store.get(trace_id)
            if trace is None:
                raise HTTPException(status_code=404, detail="Trace not found")
            return {
                "id": trace.trace_id,
                "agent": trace.agent,
                "outcome": trace.outcome,
                "duration": trace.total_latency_seconds,
                "started_at": trace.started_at,
                "steps": [
                    {
                        "step_type": s.step_type.value,
                        "input": s.input,
                        "output": s.output,
                        "duration": s.duration_seconds,
                        "metadata": s.metadata,
                    }
                    for s in trace.steps
                ],
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Templates ────────────────────────────────────────────

    @templates_router.get("")
    async def list_templates():
        return {"templates": AgentManager.list_templates()}

    @templates_router.post("/{template_id}/instantiate")
    async def instantiate_template(template_id: str, req: CreateAgentRequest):
        return project_managed_agent(
            manager.create_from_template(
                template_id,
                req.name,
                overrides=req.config,
            )
        )

    # ── Global agent endpoints ───────────────────────────────

    global_router = APIRouter(tags=["agents-global"])

    @global_router.get("/v1/agents/errors")
    def list_error_agents():
        all_agents = manager.list_agents()
        error_agents = [
            a
            for a in all_agents
            if a["status"] in ("error", "needs_attention", "stalled", "budget_exceeded")
        ]
        return {"agents": [project_managed_agent(agent) for agent in error_agents]}

    @global_router.get("/v1/agents/health")
    def agents_health():
        all_agents = manager.list_agents()
        from collections import Counter

        counts = Counter(a["status"] for a in all_agents)
        return {
            "total": len(all_agents),
            "by_status": dict(counts),
        }

    @global_router.get("/v1/recommended-model")
    def recommended_model(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            return {"model": "", "reason": "No engine available"}
        try:
            models = engine.list_models()
        except Exception:
            models = []
        return _pick_recommended_model(models)

    # ── Tools & credentials ──────────────────────────────────

    tools_router = APIRouter(prefix="/v1/tools", tags=["tools"])

    @tools_router.get("")
    def list_tools(request: Request):
        items = build_tools_list()
        try:
            mcp_tools, _ = _get_mcp_tools(request.app.state)
            for tool in mcp_tools:
                fn = tool.get("function", {})
                items.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "category": "mcp",
                        "source": "mcp",
                        "requires_credentials": False,
                        "credential_keys": [],
                        "configured": True,
                    }
                )
        except Exception:
            pass
        return {"tools": items}

    @tools_router.post("/{tool_name}/credentials")
    async def save_tool_credentials(tool_name: str, request: Request):
        from openjarvis.core.credentials import save_credential

        body = await request.json()
        saved = []
        for key, value in body.items():
            save_credential(tool_name, key, value)
            saved.append(key)
        return {"saved": saved}

    @tools_router.get("/{tool_name}/credentials/status")
    def credential_status(tool_name: str):
        from openjarvis.core.credentials import get_credential_status

        return get_credential_status(tool_name)

    # ── SendBlue auto-setup helpers ─────────────────────────

    sendblue_router = APIRouter(prefix="/v1/channels/sendblue", tags=["sendblue"])

    @sendblue_router.post("/verify")
    async def sendblue_verify(request: Request):
        """Verify SendBlue credentials and return assigned phone lines."""
        body = await request.json()
        api_key_id = body.get("api_key_id", "")
        api_secret_key = body.get("api_secret_key", "")
        if not api_key_id or not api_secret_key:
            raise HTTPException(
                status_code=400,
                detail="api_key_id and api_secret_key are required",
            )

        import httpx

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://api.sendblue.co/api/lines",
                    headers={
                        "sb-api-key-id": api_key_id,
                        "sb-api-secret-key": api_secret_key,
                    },
                )
            if resp.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid SendBlue credentials",
                )
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"SendBlue API error: {resp.text[:200]}",
                )
            data = resp.json()
            # data might be a list of lines or {"lines": [...]}
            lines = (
                data
                if isinstance(data, list)
                else data.get("lines", data.get("data", []))
            )
            numbers = []
            for line in lines:
                if isinstance(line, str):
                    num = line
                elif isinstance(line, dict):
                    num = (
                        line.get("number")
                        or line.get("phone_number")
                        or line.get("from_number")
                    )
                else:
                    num = None
                if num:
                    numbers.append(num)
            return {
                "valid": True,
                "numbers": numbers,
                "raw": data,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach SendBlue: {exc}",
            )

    @sendblue_router.post("/register-webhook")
    async def sendblue_register_webhook(request: Request):
        """Auto-register the /webhooks/sendblue endpoint with SendBlue."""
        body = await request.json()
        api_key_id = body.get("api_key_id", "")
        api_secret_key = body.get("api_secret_key", "")
        webhook_url = body.get("webhook_url", "")
        if not webhook_url:
            raise HTTPException(
                status_code=400,
                detail="webhook_url is required",
            )

        import httpx

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.sendblue.co/api/account/webhooks",
                    headers={
                        "sb-api-key-id": api_key_id,
                        "sb-api-secret-key": api_secret_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "receive": webhook_url,
                    },
                )
            return {
                "registered": resp.status_code < 300,
                "status": resp.status_code,
                "response": resp.json() if resp.status_code < 300 else resp.text[:200],
            }
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to register webhook: {exc}",
            )

    @sendblue_router.post("/test")
    async def sendblue_test(request: Request):
        """Send a test message via SendBlue to verify the setup works."""
        body = await request.json()
        api_key_id = body.get("api_key_id", "")
        api_secret_key = body.get("api_secret_key", "")
        from_number = body.get("from_number", "")
        to_number = body.get("to_number", "")
        if not to_number:
            raise HTTPException(
                status_code=400,
                detail="to_number is required",
            )

        import httpx

        try:
            payload: Dict[str, str] = {
                "number": to_number,
                "content": (
                    "Hello from your OpenJarvis agent! "
                    "Text this number anytime to search your "
                    "personal data. Reply with any question to try it."
                ),
            }
            if from_number:
                payload["from_number"] = from_number

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.sendblue.co/api/send-message",
                    headers={
                        "sb-api-key-id": api_key_id,
                        "sb-api-secret-key": api_secret_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            return {
                "sent": resp.status_code < 300,
                "status": resp.status_code,
                "response": resp.json() if resp.status_code < 300 else resp.text[:200],
            }
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to send test message: {exc}",
            )

    @sendblue_router.get("/health")
    async def sendblue_health(request: Request):
        """Check if the SendBlue channel bridge is wired and ready."""
        sb = getattr(request.app.state, "sendblue_channel", None)
        bridge = getattr(request.app.state, "channel_bridge", None)
        has_bridge = bridge is not None and (
            hasattr(bridge, "_channels") and "sendblue" in bridge._channels
        )
        return {
            "channel_connected": sb is not None,
            "bridge_wired": has_bridge,
            "ready": sb is not None and has_bridge,
        }

    return (
        agents_router,
        templates_router,
        global_router,
        tools_router,
        sendblue_router,
    )
