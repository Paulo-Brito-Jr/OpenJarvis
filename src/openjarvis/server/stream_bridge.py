"""Bridge sync agent.run() + EventBus events to an async SSE generator.

Subscribes to EventBus callbacks that push events into an asyncio.Queue,
runs agent.run() in a background thread, and yields SSE-formatted strings
from the queue for consumption by FastAPI's StreamingResponse.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse
from starlette.types import Send

from openjarvis.agents._stubs import AgentContext, BaseAgent
from openjarvis.core.cancellation import CancellationToken, cancellation_scope
from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.engine._base import looks_like_context_length_error
from openjarvis.server.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    DeltaMessage,
    StreamChoice,
    UsageInfo,
)

# EventTypes we subscribe to and their corresponding SSE event names
_EVENT_MAP = {
    EventType.AGENT_TURN_START: "agent_turn_start",
    EventType.INFERENCE_START: "inference_start",
    EventType.INFERENCE_END: "inference_end",
    EventType.TOOL_CALL_START: "tool_call_start",
    EventType.TOOL_CALL_END: "tool_call_end",
}

# Sentinel signalling that the agent thread has finished
_DONE = object()


def _estimate_prompt_tokens(messages: list) -> int:
    """Estimate prompt tokens from request messages (incl. system, user, context).

    Uses ~4 chars per token heuristic. Ensures system and all context are counted.
    """
    total = 0
    for m in messages:
        text = getattr(m, "content", None) or ""
        if isinstance(text, str):
            total += max(1, len(text) // 4)
        # Tool call messages may have structured content; still count what we can
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                fn = tc.get("function") if isinstance(tc, dict) else {}
                args = fn.get("arguments", "") if isinstance(fn, dict) else ""
                total += max(1, len(str(args)) // 4)
    return total


class AgentStreamBridge:
    """Bridge between a synchronous agent and an async SSE stream.

    Pattern:
    1. Subscribe EventBus callbacks that push events into an asyncio.Queue
       via ``loop.call_soon_threadsafe()``.
    2. Run ``agent.run()`` in a thread via ``asyncio.to_thread()``.
    3. Async generator reads from queue and yields SSE-formatted strings.
    4. Unsubscribe from EventBus in ``finally`` block.
    """

    def __init__(
        self,
        agent: BaseAgent,
        bus: EventBus,
        model: str,
        request: ChatCompletionRequest,
    ) -> None:
        self._agent = agent
        self._bus = bus
        self._model = model
        self._request = request
        self._chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self._queue: asyncio.Queue = asyncio.Queue()
        self._callbacks: dict[EventType, object] = {}
        self._cancellation_token = CancellationToken()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_callback(self, event_type: EventType):
        """Create a callback that pushes the event onto the async queue."""
        loop = asyncio.get_event_loop()

        def _cb(event: Event) -> None:
            loop.call_soon_threadsafe(self._queue.put_nowait, event)

        self._callbacks[event_type] = _cb
        return _cb

    def _subscribe_all(self) -> None:
        """Subscribe to all relevant EventBus event types."""
        for et in _EVENT_MAP:
            self._bus.subscribe(et, self._make_callback(et))

    def _unsubscribe_all(self) -> None:
        """Remove all registered subscriptions."""
        for et, cb in self._callbacks.items():
            self._bus.unsubscribe(et, cb)
        self._callbacks.clear()

    def _format_named_event(self, name: str, data: dict) -> str:
        """Format an SSE event with an explicit ``event:`` field."""
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    def _run_agent(self) -> object:
        """Execute the agent synchronously (called via asyncio.to_thread)."""
        ctx = AgentContext(cancellation_token=self._cancellation_token)
        # Build conversation context from prior messages
        if len(self._request.messages) > 1:
            from openjarvis.core.types import Message, Role

            for m in self._request.messages[:-1]:
                role = Role(m.role) if m.role in {r.value for r in Role} else Role.USER
                ctx.conversation.add(
                    Message(
                        role=role,
                        content=m.content or "",
                        name=m.name,
                        tool_call_id=m.tool_call_id,
                    )
                )

        input_text = (
            self._request.messages[-1].content if self._request.messages else ""
        )

        # Override agent model for this request if the caller specified one
        original_model = self._agent._model
        if self._model:
            self._agent._model = self._model
        try:
            with cancellation_scope(ctx.cancellation_token):
                return self._agent.run(input_text, context=ctx)
        finally:
            self._agent._model = original_model

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Signal that the response consumer has gone away."""
        self._cancellation_token.cancel()

    async def stream(self) -> AsyncGenerator[str, None]:
        """Async generator that yields SSE-formatted strings."""
        self._subscribe_all()

        # Kick off agent.run() in a background thread
        loop = asyncio.get_event_loop()
        agent_task = asyncio.ensure_future(asyncio.to_thread(self._run_agent))

        def _on_done(fut):
            # Retrieve an exception even if the client has already gone away;
            # this prevents an abandoned worker from producing an unhandled
            # task warning.  ``result()`` below can still retrieve it later.
            if not fut.cancelled():
                fut.exception()
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._queue.put_nowait, _DONE)

        agent_task.add_done_callback(_on_done)

        try:
            # Send initial role chunk (OpenAI-compatible)
            first_chunk = ChatCompletionChunk(
                id=self._chunk_id,
                model=self._model,
                choices=[
                    StreamChoice(
                        delta=DeltaMessage(role="assistant"),
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            # Drain queue until the agent finishes
            while True:
                item = await self._queue.get()

                if item is _DONE:
                    break

                if isinstance(item, Event):
                    sse_name = _EVENT_MAP.get(item.event_type)
                    if sse_name:
                        yield self._format_named_event(sse_name, item.data)

            # Agent is done -- retrieve result
            try:
                agent_result = agent_task.result()
            except Exception as exc:
                import logging

                logger = logging.getLogger("openjarvis.server")
                logger.error("Agent stream error: %s", exc, exc_info=True)

                error_str = str(exc)
                if (
                    getattr(exc, "is_context_length_error", False)
                    or looks_like_context_length_error(error_str)
                    or ("400" in error_str and "too long" in error_str.lower())
                ):
                    error_content = (
                        "The input is too long for the model's context window. "
                        "Please try a shorter message."
                    )
                elif "400" in error_str:
                    error_content = "The model rejected the agent request."
                else:
                    error_content = "Sorry, agent execution failed."
                error_chunk = ChatCompletionChunk(
                    id=self._chunk_id,
                    model=self._model,
                    choices=[
                        StreamChoice(
                            delta=DeltaMessage(content=error_content),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Emit tool results metadata if any
            tool_results_data = []
            for tr in agent_result.tool_results:
                tool_results_data.append(
                    {
                        "tool_name": tr.tool_name,
                        "success": tr.success,
                        "output": tr.content,
                        "latency_ms": tr.latency_seconds * 1000,
                    }
                )

            if tool_results_data:
                yield self._format_named_event(
                    "tool_results",
                    {"results": tool_results_data},
                )

            # The agent has already completed its model/tool loop. Starting a
            # second inference here would discard tool results, double cost,
            # and potentially stream an answer different from the audited
            # AgentResult. Emit that exact result in bounded chunks instead.
            content = agent_result.content or ""
            if content:
                for offset in range(0, len(content), 64):
                    token = content[offset : offset + 64]
                    chunk = ChatCompletionChunk(
                        id=self._chunk_id,
                        model=self._model,
                        choices=[
                            StreamChoice(
                                delta=DeltaMessage(content=token),
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0)

            # Final chunk: finish_reason + usage
            prompt_tokens = agent_result.metadata.get("prompt_tokens", 0)
            completion_tokens = agent_result.metadata.get(
                "completion_tokens",
                0,
            )
            total_tokens = agent_result.metadata.get("total_tokens", 0)
            if total_tokens == 0:
                # Fallback: estimate from request messages (incl. system) + content
                completion_tokens = max(len(content) // 4, 1)
                prompt_tokens = _estimate_prompt_tokens(self._request.messages)
                total_tokens = prompt_tokens + completion_tokens

            final_chunk = ChatCompletionChunk(
                id=self._chunk_id,
                model=self._model,
                choices=[
                    StreamChoice(
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            )
            final_data = json.loads(final_chunk.model_dump_json())
            final_data["usage"] = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ).model_dump()
            yield f"data: {json.dumps(final_data)}\n\n"

            yield "data: [DONE]\n\n"

        finally:
            # Cancelling an asyncio task does not stop ``to_thread`` work.
            # Signal the cooperative token first; the agent and executor will
            # refuse every later inference/tool boundary.
            self.cancel()
            if not agent_task.done():
                agent_task.cancel()
            self._unsubscribe_all()


class AgentStreamingResponse(StreamingResponse):
    """Streaming response that always closes its request-scoped agent.

    ASGI 2.4 reports a disconnected client as ``OSError`` from ``send()``.
    That error occurs in the response loop while the async generator is
    suspended at ``yield``; Python does not guarantee that the generator is
    closed at that point.  Owning cleanup at the response boundary ensures the
    synchronous worker sees cancellation even when Starlette retains the body
    iterator after the failed send.
    """

    def __init__(self, bridge: AgentStreamBridge) -> None:
        self._bridge = bridge
        self._agent_stream = bridge.stream()
        super().__init__(
            self._agent_stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            self._bridge.cancel()
            await self._agent_stream.aclose()


async def create_agent_stream(
    agent: BaseAgent,
    bus: EventBus,
    model: str,
    request: ChatCompletionRequest,
) -> StreamingResponse:
    """Create an AgentStreamBridge and return a FastAPI StreamingResponse."""
    bridge = AgentStreamBridge(agent, bus, model, request)
    return AgentStreamingResponse(bridge)


__all__ = ["AgentStreamBridge", "AgentStreamingResponse", "create_agent_stream"]
