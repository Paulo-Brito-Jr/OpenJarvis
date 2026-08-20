"""Cancellation coverage for managed-agent and research SSE workers."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import ClientDisconnect

from openjarvis.core.cancellation import (
    AgentCancelledError,
    CancellationToken,
    current_cancellation_token,
)
from openjarvis.engine._stubs import StreamChunk
from openjarvis.server import agent_manager_routes, research_router, stream_bridge
from openjarvis.server.stream_bridge import CancelableStreamingResponse


class _RecordingManager:
    def __init__(self) -> None:
        self.delivered: list[str] = []
        self.learning_logs: list[tuple[str, str, str, dict[str, Any]]] = []
        self.stored: list[tuple[str, str, list[dict[str, Any]] | None]] = []

    def list_messages(self, agent_id: str, *, limit: int) -> list[dict[str, Any]]:
        del agent_id, limit
        return []

    def mark_message_delivered(self, message_id: str) -> None:
        self.delivered.append(message_id)

    def add_learning_log(
        self,
        agent_id: str,
        event_type: str,
        summary: str,
        metadata: dict[str, Any],
    ) -> None:
        self.learning_logs.append((agent_id, event_type, summary, metadata))

    def store_agent_response(
        self,
        agent_id: str,
        content: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.stored.append((agent_id, content, tool_calls))


def _managed_app_state(
    *,
    capability_policy: Any = None,
    mcp_cache: tuple[list[dict[str, Any]], dict[str, Any]] = ([], {}),
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(memory_files=None, system_prompt=None),
        capability_policy=capability_policy,
        boundary_guard=None,
        _mcp_tools_cache=mcp_cache,
        model="test-model",
    )


def _agent_record(*, agent_type: str = "simple") -> dict[str, Any]:
    return {
        "id": "managed-agent",
        "agent_type": agent_type,
        "config": {
            "model": "test-model",
            "max_turns": 3,
            "temperature": 0.0,
            "max_tokens": 64,
        },
    }


async def _disconnect_when(
    response: CancelableStreamingResponse,
    ready: threading.Event,
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        entered = await asyncio.to_thread(ready.wait, 2)
        assert entered is True
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await response(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
        },
        receive,
        send,
    )
    return sent


@pytest.mark.asyncio
async def test_sync_close_callback_runs_off_event_loop_after_immediate_cancel() -> None:
    event_loop_thread = threading.get_ident()
    cancel_threads: list[int] = []
    close_threads: list[int] = []

    async def body():
        yield "disconnect me"

    def cancel() -> None:
        cancel_threads.append(threading.get_ident())

    def on_close() -> None:
        close_threads.append(threading.get_ident())

    response = CancelableStreamingResponse(
        body(),
        cancel=cancel,
        on_close=on_close,
    )

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("more_body"):
            raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            },
            receive,
            send,
        )

    assert cancel_threads == [event_loop_thread]
    assert len(close_threads) == 1
    assert close_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_async_close_callback_is_awaited_without_threadpool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_threads: list[int] = []
    event_loop_thread = threading.get_ident()

    async def body():
        yield "complete"

    async def on_close() -> None:
        close_threads.append(threading.get_ident())

    async def reject_threadpool(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("async close callback was sent to threadpool")

    monkeypatch.setattr(stream_bridge, "run_in_threadpool", reject_threadpool)
    response = CancelableStreamingResponse(
        body(),
        cancel=lambda: None,
        on_close=on_close,
    )

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        del message

    await response(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        },
        receive,
        send,
    )

    assert close_threads == [event_loop_thread]


@pytest.mark.asyncio
async def test_asgi23_disconnect_shields_stream_and_close_cleanup() -> None:
    body_started = threading.Event()
    body_closed = threading.Event()
    cancel_called = threading.Event()
    close_called = threading.Event()

    async def body():
        try:
            body_started.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            body_closed.set()

    response = CancelableStreamingResponse(
        body(),
        cancel=cancel_called.set,
        on_close=close_called.set,
    )

    await _disconnect_when(response, body_started)

    assert cancel_called.is_set()
    assert body_closed.is_set()
    assert close_called.is_set()


@pytest.mark.asyncio
async def test_managed_send_failure_persists_partial_without_second_turn() -> None:
    manager = _RecordingManager()

    class PartialEngine:
        engine_id = "partial"
        _model = "test-model"

        def __init__(self) -> None:
            self.calls = 0
            self.resumed_after_partial = False
            self.closed = False

        async def stream_full(self, messages, *, model, **kwargs):
            del messages, model, kwargs
            self.calls += 1
            try:
                yield StreamChunk(content="audited partial")
                self.resumed_after_partial = True
                yield StreamChunk(finish_reason="stop")
            finally:
                self.closed = True

    engine = PartialEngine()
    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(),
        user_content="start",
        message_id="message-1",
        engine=engine,
        bus=None,
        app_state=_managed_app_state(),
        operator_id="api:test",
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("more_body"):
            raise OSError("client disconnected during managed send")

    with pytest.raises(ClientDisconnect):
        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            },
            receive,
            send,
        )

    assert engine.calls == 1
    assert engine.resumed_after_partial is False
    assert engine.closed is True
    assert manager.stored == [
        ("managed-agent", "audited partial", None),
    ]
    assert [entry[1] for entry in manager.learning_logs].count("query_complete") == 1


@pytest.mark.asyncio
async def test_managed_send_failure_retries_transient_store_once() -> None:
    class RetryManager(_RecordingManager):
        def __init__(self) -> None:
            super().__init__()
            self.store_attempts = 0

        def store_agent_response(
            self,
            agent_id: str,
            content: str,
            *,
            tool_calls: list[dict[str, Any]] | None = None,
        ) -> None:
            self.store_attempts += 1
            if self.store_attempts == 1:
                raise RuntimeError("transient store failure")
            super().store_agent_response(
                agent_id,
                content,
                tool_calls=tool_calls,
            )

    class PartialEngine:
        engine_id = "partial-retry"
        _model = "test-model"

        async def stream_full(self, messages, *, model, **kwargs):
            del messages, model, kwargs
            yield StreamChunk(content="retry this response")
            yield StreamChunk(finish_reason="stop")

    manager = RetryManager()
    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(),
        user_content="start",
        message_id="message-retry",
        engine=PartialEngine(),
        bus=None,
        app_state=_managed_app_state(),
        operator_id="api:test",
    )

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("more_body"):
            raise OSError("client disconnected during managed send")

    with pytest.raises(ClientDisconnect):
        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            },
            receive,
            send,
        )

    assert manager.store_attempts == 2
    assert manager.stored == [("managed-agent", "retry this response", None)]
    assert [entry[1] for entry in manager.learning_logs].count("query_complete") == 1


@pytest.mark.asyncio
async def test_managed_disconnect_stops_blocked_authorization_before_tool_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _RecordingManager()
    authorization_entered = threading.Event()
    authorization_finished = threading.Event()

    class ToolCallingEngine:
        engine_id = "tool-calling"
        _model = "test-model"

        def __init__(self) -> None:
            self.calls = 0

        async def stream_full(self, messages, *, model, **kwargs):
            del messages, model, kwargs
            self.calls += 1
            yield StreamChunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-1",
                        "function": {
                            "name": "blocked_tool",
                            "arguments": "{}",
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

    class AuthorizationExecutor:
        def __init__(self) -> None:
            self.authorization_checks = 0
            self.tool_dispatches = 0

        def execute(self, tool_call):
            del tool_call
            token = current_cancellation_token()
            assert token is not None
            self.authorization_checks += 1
            authorization_entered.set()
            try:
                if not token._cancelled.wait(timeout=2):
                    raise TimeoutError("cancellation did not reach authorization")
                token.raise_if_cancelled()
                self.tool_dispatches += 1
                raise AssertionError("tool dispatched after disconnect")
            finally:
                authorization_finished.set()

    engine = ToolCallingEngine()
    executor = AuthorizationExecutor()
    monkeypatch.setattr(
        agent_manager_routes,
        "_build_server_tool_executor",
        lambda **kwargs: executor,
    )
    mcp_spec = {
        "type": "function",
        "function": {
            "name": "blocked_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(),
        user_content="use the tool",
        message_id="message-2",
        engine=engine,
        bus=None,
        app_state=_managed_app_state(
            mcp_cache=([mcp_spec], {"blocked_tool": object()}),
        ),
        operator_id="api:test",
    )

    await _disconnect_when(response, authorization_entered)
    finished = await asyncio.to_thread(authorization_finished.wait, 2)

    assert finished is True
    assert engine.calls == 1
    assert executor.authorization_checks == 1
    assert executor.tool_dispatches == 0
    assert not manager.stored
    assert [entry[1] for entry in manager.learning_logs].count("query_complete") == 0


@pytest.mark.asyncio
async def test_managed_disconnect_persists_dispatched_tool_after_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EventManager(_RecordingManager):
        def __init__(self) -> None:
            super().__init__()
            self.stored_event = threading.Event()

        def store_agent_response(
            self,
            agent_id: str,
            content: str,
            *,
            tool_calls: list[dict[str, Any]] | None = None,
        ) -> None:
            super().store_agent_response(
                agent_id,
                content,
                tool_calls=tool_calls,
            )
            self.stored_event.set()

    class ToolCallingEngine:
        engine_id = "tool-calling"
        _model = "test-model"

        async def stream_full(self, messages, *, model, **kwargs):
            del messages, model, kwargs
            yield StreamChunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-in-flight",
                        "function": {
                            "name": "slow_tool",
                            "arguments": '{"value":1}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

    worker_entered = threading.Event()
    release_worker = threading.Event()

    class CompletingExecutor:
        def execute(self, tool_call):
            del tool_call
            assert current_cancellation_token() is not None
            worker_entered.set()
            if not release_worker.wait(timeout=2):
                raise TimeoutError("test did not release dispatched tool")
            return SimpleNamespace(success=True, content="effect completed")

    manager = EventManager()
    monkeypatch.setattr(
        agent_manager_routes,
        "_build_server_tool_executor",
        lambda **kwargs: CompletingExecutor(),
    )
    mcp_spec = {
        "type": "function",
        "function": {
            "name": "slow_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(),
        user_content="dispatch the tool",
        message_id="message-in-flight",
        engine=ToolCallingEngine(),
        bus=None,
        app_state=_managed_app_state(
            mcp_cache=([mcp_spec], {"slow_tool": object()}),
        ),
        operator_id="api:test",
    )

    await _disconnect_when(response, worker_entered)
    assert not manager.stored

    release_worker.set()
    assert await asyncio.to_thread(manager.stored_event.wait, 2) is True

    assert len(manager.stored) == 1
    stored_agent, stored_content, stored_tools = manager.stored[0]
    assert stored_agent == "managed-agent"
    assert stored_content == ""
    assert stored_tools is not None
    assert len(stored_tools) == 1
    assert stored_tools[0]["tool"] == "slow_tool"
    assert stored_tools[0]["arguments"] == '{"value":1}'
    assert stored_tools[0]["result"] == "effect completed"
    assert stored_tools[0]["success"] is True
    assert stored_tools[0]["latency"] >= 0


@pytest.mark.asyncio
async def test_deep_research_disconnect_cancels_blocked_provider_without_error_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.agents import deep_research

    manager = _RecordingManager()
    provider_entered = threading.Event()
    worker_finished = threading.Event()

    class BlockingDeepResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self._executor = SimpleNamespace(execute=lambda tool_call: tool_call)

        def bind_security(self, *args: Any) -> None:
            del args

        def run(self, query: str):
            del query
            token = current_cancellation_token()
            assert token is not None
            provider_entered.set()
            try:
                if not token._cancelled.wait(timeout=2):
                    raise TimeoutError("cancellation did not reach provider")
                token.raise_if_cancelled()
                raise AssertionError("provider continued after disconnect")
            finally:
                worker_finished.set()

    monkeypatch.setattr(
        agent_manager_routes,
        "_build_deep_research_tools",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        deep_research,
        "DeepResearchAgent",
        BlockingDeepResearchAgent,
    )

    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(agent_type="deep_research"),
        user_content="research",
        message_id="message-3",
        engine=SimpleNamespace(_model="test-model"),
        bus=None,
        app_state=_managed_app_state(capability_policy=object()),
        operator_id="api:test",
    )

    await _disconnect_when(response, provider_entered)
    finished = await asyncio.to_thread(worker_finished.wait, 2)

    assert finished is True
    assert not manager.stored
    event_types = [entry[1] for entry in manager.learning_logs]
    assert "query_cancelled" in event_types
    assert "query_error" not in event_types


@pytest.mark.asyncio
async def test_deep_research_asgi23_disconnect_persists_late_tool_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.agents import deep_research

    class EventManager(_RecordingManager):
        def __init__(self) -> None:
            super().__init__()
            self.stored_event = threading.Event()

        def store_agent_response(
            self,
            agent_id: str,
            content: str,
            *,
            tool_calls: list[dict[str, Any]] | None = None,
        ) -> None:
            super().store_agent_response(
                agent_id,
                content,
                tool_calls=tool_calls,
            )
            self.stored_event.set()

    tool_entered = threading.Event()
    release_tool = threading.Event()

    class ToolBlockingDeepResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

            def execute(tool_call):
                del tool_call
                tool_entered.set()
                if not release_tool.wait(timeout=2):
                    raise TimeoutError("test did not release deep tool")
                return SimpleNamespace(
                    success=True,
                    content="deep effect completed",
                )

            self._executor = SimpleNamespace(execute=execute)

        def bind_security(self, *args: Any) -> None:
            del args

        def run(self, query: str):
            del query
            self._executor.execute(
                SimpleNamespace(
                    name="knowledge_search",
                    arguments='{"query":"late effect"}',
                )
            )
            token = current_cancellation_token()
            assert token is not None
            token.raise_if_cancelled()
            return SimpleNamespace(content="unreachable", metadata={})

    manager = EventManager()
    monkeypatch.setattr(
        agent_manager_routes,
        "_build_deep_research_tools",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        deep_research,
        "DeepResearchAgent",
        ToolBlockingDeepResearchAgent,
    )
    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(agent_type="deep_research"),
        user_content="research",
        message_id="message-deep-asgi23",
        engine=SimpleNamespace(_model="test-model"),
        bus=None,
        app_state=_managed_app_state(capability_policy=object()),
        operator_id="api:test",
    )

    await _disconnect_when(response, tool_entered)
    assert not manager.stored

    release_tool.set()
    assert await asyncio.to_thread(manager.stored_event.wait, 2) is True

    assert len(manager.stored) == 1
    _, stored_content, stored_tools = manager.stored[0]
    assert stored_content == ""
    assert stored_tools is not None
    assert stored_tools[0]["tool"] == "knowledge_search"
    assert stored_tools[0]["result"] == "deep effect completed"


@pytest.mark.parametrize(
    "failure_marker",
    [
        b'"content": "completed"',
        b'"finish_reason": "stop"',
    ],
    ids=["answer-token", "terminal-chunk"],
)
@pytest.mark.asyncio
async def test_deep_research_send_failure_persists_completed_state_once(
    monkeypatch: pytest.MonkeyPatch,
    failure_marker: bytes,
) -> None:
    from openjarvis.agents import deep_research

    manager = _RecordingManager()

    class CompletedDeepResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self._executor = SimpleNamespace(
                execute=lambda tool_call: SimpleNamespace(
                    success=True,
                    content=f"result for {tool_call.name}",
                )
            )

        def bind_security(self, *args: Any) -> None:
            del args

        def run(self, query: str):
            del query
            self._executor.execute(
                SimpleNamespace(
                    name="knowledge_search",
                    arguments='{"query":"auditable"}',
                )
            )
            return SimpleNamespace(
                content="completed answer",
                metadata={
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            )

    monkeypatch.setattr(
        agent_manager_routes,
        "_build_deep_research_tools",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        deep_research,
        "DeepResearchAgent",
        CompletedDeepResearchAgent,
    )

    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(agent_type="deep_research"),
        user_content="research",
        message_id="message-completed",
        engine=SimpleNamespace(_model="test-model"),
        bus=None,
        app_state=_managed_app_state(capability_policy=object()),
        operator_id="api:test",
    )

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if failure_marker in message.get("body", b""):
            raise OSError("client disconnected after deep research completed")

    with pytest.raises(ClientDisconnect):
        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            },
            receive,
            send,
        )

    assert len(manager.stored) == 1
    stored_agent, stored_content, stored_tools = manager.stored[0]
    assert stored_agent == "managed-agent"
    assert stored_content == "completed answer"
    assert stored_tools is not None
    assert len(stored_tools) == 1
    assert stored_tools[0] == {
        "tool": "knowledge_search",
        "arguments": '{"query":"auditable"}',
        "result": "result for knowledge_search",
        "success": True,
        "latency": stored_tools[0]["latency"],
    }
    assert stored_tools[0]["latency"] >= 0


@pytest.mark.asyncio
async def test_deep_research_deadline_uses_daemon_worker_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.agents import deep_research

    manager = _RecordingManager()
    release_provider = threading.Event()
    real_thread_class = threading.Thread
    deep_threads: list[threading.Thread] = []

    class RecordingThread(real_thread_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.name.startswith("deep-research-"):
                deep_threads.append(self)

    class BlockingDeepResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self._executor = SimpleNamespace(execute=lambda tool_call: tool_call)

        def bind_security(self, *args: Any) -> None:
            del args

        def run(self, query: str):
            del query
            release_provider.wait(timeout=2)
            return SimpleNamespace(content="late result", metadata={})

    monkeypatch.setattr(
        agent_manager_routes,
        "_DEEP_RESEARCH_STREAM_TIMEOUT_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        agent_manager_routes.threading,
        "Thread",
        RecordingThread,
    )
    monkeypatch.setattr(
        agent_manager_routes,
        "_build_deep_research_tools",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        deep_research,
        "DeepResearchAgent",
        BlockingDeepResearchAgent,
    )

    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(agent_type="deep_research"),
        user_content="research forever",
        message_id="message-timeout",
        engine=SimpleNamespace(_model="test-model"),
        bus=None,
        app_state=_managed_app_state(capability_policy=object()),
        operator_id="api:test",
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    try:
        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            },
            receive,
            send,
        )
    finally:
        release_provider.set()
        for worker in deep_threads:
            worker.join(timeout=2)

    payload = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert b"Deep" in payload, payload
    assert b"research" in payload
    assert b"timed" in payload
    assert b"out." in payload
    assert b"data: [DONE]" in payload
    assert len(deep_threads) == 1
    assert deep_threads[0].daemon is True
    assert deep_threads[0].is_alive() is False
    # The client deadline remains a timeout, but a provider result that
    # subsequently completes is the canonical persisted terminal state.
    assert manager.stored == [("managed-agent", "late result", None)]


@pytest.mark.asyncio
async def test_deep_research_timeout_waits_for_in_flight_tool_before_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.agents import deep_research

    class EventManager(_RecordingManager):
        def __init__(self) -> None:
            super().__init__()
            self.stored_event = threading.Event()

        def store_agent_response(
            self,
            agent_id: str,
            content: str,
            *,
            tool_calls: list[dict[str, Any]] | None = None,
        ) -> None:
            super().store_agent_response(
                agent_id,
                content,
                tool_calls=tool_calls,
            )
            self.stored_event.set()

    tool_entered = threading.Event()
    release_tool = threading.Event()
    deep_threads: list[threading.Thread] = []
    real_thread_class = threading.Thread

    class CoordinatedThread(real_thread_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.name.startswith("deep-research-"):
                deep_threads.append(self)

        def start(self) -> None:
            super().start()
            assert tool_entered.wait(timeout=2) is True

    class ToolBlockingDeepResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

            def execute(tool_call):
                del tool_call
                tool_entered.set()
                if not release_tool.wait(timeout=2):
                    raise TimeoutError("test did not release deep tool")
                return SimpleNamespace(
                    success=True,
                    content="tool result after timeout",
                )

            self._executor = SimpleNamespace(execute=execute)

        def bind_security(self, *args: Any) -> None:
            del args

        def run(self, query: str):
            del query
            self._executor.execute(
                SimpleNamespace(
                    name="knowledge_search",
                    arguments='{"query":"timeout"}',
                )
            )
            token = current_cancellation_token()
            assert token is not None
            token.raise_if_cancelled()
            return SimpleNamespace(content="unreachable", metadata={})

    manager = EventManager()
    monkeypatch.setattr(
        agent_manager_routes,
        "_DEEP_RESEARCH_STREAM_TIMEOUT_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        agent_manager_routes,
        "threading",
        SimpleNamespace(Lock=threading.Lock, Thread=CoordinatedThread),
    )
    monkeypatch.setattr(
        agent_manager_routes,
        "_build_deep_research_tools",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        deep_research,
        "DeepResearchAgent",
        ToolBlockingDeepResearchAgent,
    )
    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(agent_type="deep_research"),
        user_content="research",
        message_id="message-deep-timeout-tool",
        engine=SimpleNamespace(_model="test-model"),
        bus=None,
        app_state=_managed_app_state(capability_policy=object()),
        operator_id="api:test",
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await response(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        },
        receive,
        send,
    )

    assert not manager.stored
    release_tool.set()
    for worker in deep_threads:
        await asyncio.to_thread(worker.join, 2)
    assert await asyncio.to_thread(manager.stored_event.wait, 2) is True

    payload = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert b"timed" in payload
    assert len(manager.stored) == 1
    _, stored_content, stored_tools = manager.stored[0]
    assert stored_content == ""
    assert stored_tools is not None
    assert stored_tools[0]["result"] == "tool result after timeout"


@pytest.mark.asyncio
async def test_deep_research_deadline_prefers_terminal_before_queue_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.agents import deep_research

    manager = _RecordingManager()
    real_loop = asyncio.get_running_loop()
    scheduled_callbacks: list[tuple[Any, tuple[Any, ...]]] = []

    class DirectLoop:
        def time(self) -> float:
            return real_loop.time()

        def is_closed(self) -> bool:
            return real_loop.is_closed()

        def call_soon_threadsafe(self, callback, *args) -> None:
            scheduled_callbacks.append((callback, args))

    class AsyncioProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        def get_running_loop(self) -> DirectLoop:
            return DirectLoop()

    class InlineThread:
        def __init__(
            self,
            *,
            target,
            name: str,
            daemon: bool,
        ) -> None:
            del name, daemon
            self._target = target

        def start(self) -> None:
            self._target()

    class CompletedDeepResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self._executor = SimpleNamespace(execute=lambda tool_call: tool_call)

        def bind_security(self, *args: Any) -> None:
            del args

        def run(self, query: str):
            del query
            return SimpleNamespace(content="boundary result", metadata={})

    monkeypatch.setattr(
        agent_manager_routes,
        "_DEEP_RESEARCH_STREAM_TIMEOUT_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        agent_manager_routes,
        "asyncio",
        AsyncioProxy(),
    )
    monkeypatch.setattr(
        agent_manager_routes,
        "threading",
        SimpleNamespace(Lock=threading.Lock, Thread=InlineThread),
    )
    monkeypatch.setattr(
        agent_manager_routes,
        "_build_deep_research_tools",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        deep_research,
        "DeepResearchAgent",
        CompletedDeepResearchAgent,
    )

    response = await agent_manager_routes._stream_managed_agent(
        manager=manager,  # type: ignore[arg-type]
        agent_record=_agent_record(agent_type="deep_research"),
        user_content="boundary",
        message_id="message-boundary",
        engine=SimpleNamespace(_model="test-model"),
        bus=None,
        app_state=_managed_app_state(capability_policy=object()),
        operator_id="api:test",
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await response(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        },
        receive,
        send,
    )

    payload = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert b"boundary" in payload
    assert b"timed" not in payload
    assert len(scheduled_callbacks) == 1
    assert manager.stored == [("managed-agent", "boundary result", None)]


class _ResearchSampler:
    available = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        return

    def stop(self) -> dict[str, float]:
        return {
            "energy_j": 0.0,
            "mean_power_w": 0.0,
            "peak_power_w": 0.0,
        }


@pytest.mark.asyncio
async def test_research_close_stops_sampler_and_drops_late_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_entered = threading.Event()
    release_provider = threading.Event()
    worker_finished = threading.Event()
    worker_done_scheduled = threading.Event()
    sampler_instances: list[Any] = []
    metric_schedules: list[dict[str, Any]] = []

    class RecordingSampler:
        available = True

        def __init__(self, on_sample, **kwargs: Any) -> None:
            del kwargs
            self.on_sample = on_sample
            self.start_calls = 0
            self.stop_calls = 0
            sampler_instances.append(self)

        def start(self) -> None:
            self.start_calls += 1
            self.on_sample(25.0, 12.5, 0.5)

        def stop(self) -> dict[str, float]:
            self.stop_calls += 1
            return {
                "energy_j": 12.5,
                "mean_power_w": 25.0,
                "peak_power_w": 25.0,
                "duration_s": 0.5,
            }

    class BlockingResearchAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.last_usage: dict[str, int] = {}

        def run(self, query: str):
            del query
            provider_entered.set()
            try:
                if not release_provider.wait(timeout=2):
                    raise TimeoutError("test did not release research provider")
                return SimpleNamespace(
                    answer="late provider result",
                    usage={},
                    sources=[],
                )
            finally:
                worker_finished.set()

    class AllowPolicy:
        def check(self, *args: Any) -> bool:
            del args
            return True

    monkeypatch.setattr(research_router, "load_config", lambda: object())
    monkeypatch.setattr(
        research_router,
        "_build_planner_engine",
        lambda *args, **kwargs: ("research-test", object(), "test-model"),
    )
    monkeypatch.setattr(research_router, "KnowledgeStore", lambda: object())
    monkeypatch.setattr(
        research_router,
        "OllamaEmbedder",
        lambda: SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(
        research_router,
        "HybridSearch",
        lambda store, embedder: object(),
    )
    monkeypatch.setattr(research_router, "ResearchAgent", BlockingResearchAgent)
    monkeypatch.setattr(research_router, "_LiveGPUSampler", RecordingSampler)

    loop = asyncio.get_running_loop()
    original_call_soon_threadsafe = loop.call_soon_threadsafe

    def record_call_soon_threadsafe(callback, *args, **kwargs):
        if args and isinstance(args[0], dict):
            if args[0].get("type") == "system_metrics":
                metric_schedules.append(args[0])
        if args and args[0] is research_router._DONE:
            worker_done_scheduled.set()
        return original_call_soon_threadsafe(callback, *args, **kwargs)

    monkeypatch.setattr(
        loop,
        "call_soon_threadsafe",
        record_call_soon_threadsafe,
    )

    request = SimpleNamespace(
        state=SimpleNamespace(api_principal="api:test"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine=object(),
                engine_name="research-test",
                model="test-model",
                capability_policy=AllowPolicy(),
            )
        ),
    )
    response = await research_router.research(
        research_router.ResearchRequest(query="cancel sampling"),
        request,  # type: ignore[arg-type]
    )

    try:
        await _disconnect_when(response, provider_entered)

        assert len(sampler_instances) == 1
        sampler = sampler_instances[0]
        assert sampler.start_calls == 1
        # The provider is deliberately non-cooperative, so only the shielded
        # response on_close callback can stop the sampler at this point.
        assert worker_finished.is_set() is False
        assert sampler.stop_calls == 1
        assert len(metric_schedules) == 1

        sampler.on_sample(50.0, 50.0, 1.0)
        assert len(metric_schedules) == 1
    finally:
        release_provider.set()

    assert await asyncio.to_thread(worker_finished.wait, 2) is True
    assert await asyncio.to_thread(worker_done_scheduled.wait, 2) is True


@pytest.mark.asyncio
async def test_research_send_failure_cancels_blocked_search_and_keeps_usage_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_entered = threading.Event()
    authorization_finished = threading.Event()
    telemetry_recorded = threading.Event()
    telemetry: dict[str, Any] = {}

    class ResearchPolicy:
        def __init__(self) -> None:
            self.checks = 0

        def check(self, agent_id: str, capability: str, resource: str) -> bool:
            del agent_id, capability, resource
            self.checks += 1
            if self.checks < 5:
                return True
            token = current_cancellation_token()
            assert token is not None
            authorization_entered.set()
            try:
                if not token._cancelled.wait(timeout=2):
                    raise TimeoutError(
                        "cancellation did not reach search authorization"
                    )
                return True
            finally:
                authorization_finished.set()

    class ResearchEngine:
        engine_id = "research-test"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search-1",
                        "name": "search",
                        "arguments": '{"query":"cancel me"}',
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            }

    class RecordingSearch:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, *args: Any, **kwargs: Any) -> list[Any]:
            del args, kwargs
            self.calls += 1
            return []

    engine = ResearchEngine()
    policy = ResearchPolicy()
    search = RecordingSearch()

    monkeypatch.setattr(research_router, "load_config", lambda: object())
    monkeypatch.setattr(
        research_router,
        "_build_planner_engine",
        lambda *args, **kwargs: ("research-test", engine, "test-model"),
    )
    monkeypatch.setattr(research_router, "KnowledgeStore", object)
    monkeypatch.setattr(
        research_router,
        "OllamaEmbedder",
        lambda: SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(
        research_router,
        "HybridSearch",
        lambda store, embedder: search,
    )
    monkeypatch.setattr(research_router, "_LiveGPUSampler", _ResearchSampler)

    def record_telemetry(**kwargs: Any) -> None:
        telemetry.update(kwargs)
        telemetry_recorded.set()

    monkeypatch.setattr(
        research_router,
        "_record_research_telemetry",
        record_telemetry,
    )

    request = SimpleNamespace(
        state=SimpleNamespace(api_principal="api:test"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine=engine,
                engine_name="research-test",
                model="test-model",
                capability_policy=policy,
            )
        ),
    )
    response = await research_router.research(
        research_router.ResearchRequest(query="find something"),
        request,  # type: ignore[arg-type]
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("more_body"):
            entered = await asyncio.to_thread(authorization_entered.wait, 2)
            assert entered is True
            raise OSError("client disconnected during research send")

    with pytest.raises(ClientDisconnect):
        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
            },
            receive,
            send,
        )

    authorization_done = await asyncio.to_thread(authorization_finished.wait, 2)
    audit_done = await asyncio.to_thread(telemetry_recorded.wait, 2)

    assert authorization_done is True
    assert audit_done is True
    assert engine.calls == 1
    assert policy.checks == 5
    assert search.calls == 0
    assert telemetry["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.asyncio
async def test_pre_cancelled_research_stream_skips_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_calls = 0

    def load_config() -> object:
        nonlocal setup_calls
        setup_calls += 1
        return object()

    monkeypatch.setattr(research_router, "load_config", load_config)
    token = CancellationToken()
    token.cancel()
    stream = research_router._stream_research(
        "cancelled",
        capability_policy=object(),
        agent_id="api:test",
        cancellation_token=token,
    )

    with pytest.raises(AgentCancelledError):
        await anext(stream)
    await stream.aclose()

    assert setup_calls == 0
