"""WebSocket bridge: EventBus → connected WebSocket clients."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping
from typing import Any

from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.security.taint import redact_sensitive_text
from openjarvis.server.input_limits import (
    MAX_WS_FILTER_BYTES,
    utf8_size_exceeds,
)

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover
    pass  # FastAPI is optional; create_ws_router will fail at call time

logger = logging.getLogger(__name__)

# Agent-related event types to forward
_AGENT_EVENTS = {
    EventType.AGENT_TICK_START,
    EventType.AGENT_TICK_END,
    EventType.AGENT_TICK_ERROR,
    EventType.AGENT_BUDGET_EXCEEDED,
    EventType.AGENT_STALL_DETECTED,
    EventType.AGENT_MESSAGE_RECEIVED,
    EventType.AGENT_CHECKPOINT_SAVED,
    EventType.TOOL_CALL_START,
    EventType.TOOL_CALL_END,
    EventType.INFERENCE_START,
    EventType.INFERENCE_END,
}

_EVENT_FIELD_ALLOWLIST: dict[EventType, frozenset[str]] = {
    EventType.AGENT_TICK_START: frozenset({"agent_id", "agent_name"}),
    EventType.AGENT_TICK_END: frozenset({"agent_id", "duration", "status"}),
    EventType.AGENT_TICK_ERROR: frozenset({"agent_id", "duration", "error_type"}),
    EventType.AGENT_BUDGET_EXCEEDED: frozenset(
        {
            "agent_id",
            "total_cost",
            "total_tokens",
            "max_cost",
            "max_tokens",
        }
    ),
    EventType.AGENT_STALL_DETECTED: frozenset(
        {"agent_id", "last_activity_at", "stall_retries"}
    ),
    EventType.AGENT_MESSAGE_RECEIVED: frozenset({"agent_id", "source"}),
    EventType.AGENT_CHECKPOINT_SAVED: frozenset({"agent_id", "checkpoint_id"}),
    EventType.TOOL_CALL_START: frozenset({"tool", "agent"}),
    EventType.TOOL_CALL_END: frozenset({"tool", "agent", "success", "latency"}),
    EventType.INFERENCE_START: frozenset({"model", "engine", "agent", "message_count"}),
    EventType.INFERENCE_END: frozenset(
        {"model", "agent", "finish_reason", "latency", "usage"}
    ),
}
_USAGE_FIELD_ALLOWLIST = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
    }
)
_MAX_EVENT_TEXT = 256


def _project_scalar(value: Any) -> str | bool | int | float | None:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return redact_sensitive_text(value)[:_MAX_EVENT_TEXT]
    return None


def _project_event(event: Event) -> dict[str, Any] | None:
    """Project an internal event onto a small, redacted public schema."""
    allowed_fields = _EVENT_FIELD_ALLOWLIST.get(event.event_type)
    if allowed_fields is None:
        return None
    raw_data = event.data if isinstance(event.data, Mapping) else {}
    public_data: dict[str, Any] = {}
    for field in allowed_fields:
        value = raw_data.get(field)
        if field == "usage":
            if not isinstance(value, Mapping):
                continue
            usage = {
                key: item
                for key in _USAGE_FIELD_ALLOWLIST
                if isinstance((item := value.get(key)), (int, float))
                and not isinstance(item, bool)
                and (not isinstance(item, float) or math.isfinite(item))
            }
            if usage:
                public_data[field] = usage
            continue
        projected = _project_scalar(value)
        if projected is not None:
            public_data[field] = projected
    timestamp = (
        event.timestamp
        if isinstance(event.timestamp, (int, float))
        and not isinstance(event.timestamp, bool)
        and (not isinstance(event.timestamp, float) or math.isfinite(event.timestamp))
        else 0.0
    )
    return {
        "type": event.event_type.value,
        "timestamp": timestamp,
        "data": public_data,
    }


def create_ws_router(event_bus: EventBus) -> Any:
    """Create a FastAPI router with a WebSocket endpoint for agent events."""
    router = APIRouter()
    # Each connected client gets a queue + loop ref for thread-safe event delivery
    clients: dict[WebSocket, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}

    def _on_event(event: Event) -> None:
        """Forward event to all connected WebSocket client queues (thread-safe)."""
        try:
            payload = _project_event(event)
        except Exception:
            logger.warning("WebSocket event projection failed; event withheld")
            return
        if payload is None:
            return
        for ws, (queue, loop) in list(clients.items()):
            agent_filter = getattr(ws, "_agent_filter", None)
            # Tick events carry "agent_id"; tool-call events carry "agent".
            # Match either so a per-agent subscriber actually receives the
            # tool calls that make up its live trace (without this, only
            # tick_start/end pass the filter and the trace looks empty).
            data = event.data if isinstance(event.data, Mapping) else {}
            event_agent = data.get("agent_id") or data.get("agent")
            if agent_filter and event_agent != agent_filter:
                continue
            try:
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except (RuntimeError, asyncio.QueueFull):
                pass  # Loop closed or client is slow

    # Subscribe to all agent events
    for event_type in _AGENT_EVENTS:
        event_bus.subscribe(event_type, _on_event)

    @router.websocket("/v1/agents/events")
    async def agent_events(websocket: WebSocket) -> None:
        from openjarvis.server.auth_middleware import (
            websocket_authorized,
            websocket_capability_authorized,
        )

        expected_key = getattr(websocket.app.state, "api_key", None)
        authenticated = websocket_authorized(
            websocket,
            expected_key,
            principal=getattr(websocket.app.state, "api_principal", ""),
            allowed_principals=getattr(
                websocket.app.state,
                "api_principal_allowlist",
                (),
            ),
        )
        authorized = authenticated and websocket_capability_authorized(
            websocket,
            "system:admin",
            "/v1/agents/events",
        )
        if not authorized:
            # 1008 = policy violation; reject before accepting the connection.
            await websocket.close(code=1008)
            return
        # Parse agent_id filter from query string
        agent_id = websocket.query_params.get("agent_id")
        if agent_id is not None and utf8_size_exceeds(
            agent_id,
            MAX_WS_FILTER_BYTES,
        ):
            await websocket.close(code=1009)
            return
        await websocket.accept()
        websocket._agent_filter = agent_id  # type: ignore[attr-defined]
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        loop = asyncio.get_running_loop()
        clients[websocket] = (queue, loop)
        try:
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            clients.pop(websocket, None)

    return router


__all__ = ["create_ws_router"]
