"""Cancellation guarantees for the synchronous agent SSE bridge."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

import pytest
from starlette.requests import ClientDisconnect

from openjarvis.agents._stubs import AgentContext, AgentResult
from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.core.cancellation import (
    AgentCancelledError,
    CancellationToken,
    cancellation_scope,
    current_cancellation_token,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.server.models import ChatCompletionRequest
from openjarvis.server.stream_bridge import AgentStreamBridge, create_agent_stream
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


class _RecordingMutationTool(BaseTool):
    tool_id = "record_mutation"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="record_mutation",
            description="Record a mutation for cancellation tests.",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **params: Any) -> ToolResult:
        self.calls.append(dict(params))
        return ToolResult(
            tool_name=self.spec.name,
            content="mutated",
            success=True,
        )


class _BlockingToolCallEngine:
    engine_id = "blocking-test"

    def __init__(self) -> None:
        self.generate_entered = threading.Event()
        self.release_generate = threading.Event()
        self.generate_calls = 0

    def generate(self, messages, **kwargs):
        del messages, kwargs
        self.generate_calls += 1
        self.generate_entered.set()
        if not self.release_generate.wait(timeout=5):
            raise TimeoutError("test did not release the blocking engine")
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "record_mutation",
                    "arguments": "{}",
                },
                {
                    "id": "call_2",
                    "name": "record_mutation",
                    "arguments": "{}",
                },
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "finish_reason": "tool_calls",
        }


class _ObservedOrchestrator(OrchestratorAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_finished = threading.Event()
        self.context_token: Optional[CancellationToken] = None
        self.current_token: Optional[CancellationToken] = None

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        assert context is not None
        self.context_token = context.cancellation_token
        self.current_token = current_cancellation_token()
        try:
            return super().run(input, context=context, **kwargs)
        finally:
            self.run_finished.set()


def _allow_agent(agent: OrchestratorAgent) -> None:
    policy = CapabilityPolicy()
    policy.grant("stream-cancellation-test", "*")
    agent.bind_security(policy, "stream-cancellation-test")


@pytest.mark.asyncio
async def test_disconnect_stops_agent_before_mutating_tool_invocation() -> None:
    engine = _BlockingToolCallEngine()
    tool = _RecordingMutationTool()
    bus = EventBus()
    agent = _ObservedOrchestrator(
        engine,
        "test-model",
        tools=[tool],
        # Keep the bridge bus quiet so the pending ``anext`` remains blocked
        # until the simulated disconnect instead of consuming a turn event.
        bus=None,
        max_turns=2,
        temperature=0.0,
        max_tokens=64,
    )
    _allow_agent(agent)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "perform the mutation"}],
        stream=True,
    )
    bridge = AgentStreamBridge(agent, bus, "test-model", request)
    stream = bridge.stream()
    pending_chunk: Optional[asyncio.Task[str]] = None

    try:
        first_chunk = await anext(stream)
        assert '"role":"assistant"' in first_chunk

        # Wait inside the generator just as Starlette does while a client is
        # connected.  Cancelling this task injects asyncio.CancelledError into
        # the bridge and exercises its unconditional ``finally`` cleanup.
        pending_chunk = asyncio.create_task(anext(stream))
        entered = await asyncio.wait_for(
            asyncio.to_thread(engine.generate_entered.wait, 2),
            timeout=3,
        )
        assert entered is True

        pending_chunk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_chunk

        engine.release_generate.set()
        finished = await asyncio.wait_for(
            asyncio.to_thread(agent.run_finished.wait, 2),
            timeout=3,
        )
        assert finished is True
    finally:
        engine.release_generate.set()
        if pending_chunk is not None and not pending_chunk.done():
            pending_chunk.cancel()
        await stream.aclose()

    assert engine.generate_calls == 1
    assert tool.calls == []
    assert agent.context_token is bridge._cancellation_token
    assert agent.current_token is bridge._cancellation_token
    assert bridge._cancellation_token.is_cancelled is True


@pytest.mark.asyncio
async def test_asgi_send_disconnect_closes_suspended_agent_stream() -> None:
    engine = _BlockingToolCallEngine()
    tool = _RecordingMutationTool()
    bus = EventBus()
    agent = _ObservedOrchestrator(
        engine,
        "test-model",
        tools=[tool],
        bus=None,
        max_turns=2,
        temperature=0.0,
        max_tokens=64,
    )
    _allow_agent(agent)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "perform the mutation"}],
        stream=True,
    )
    response = await create_agent_stream(
        agent,
        bus,
        "test-model",
        request,
    )
    sent_messages: list[dict[str, Any]] = []

    async def _receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def _send(message: dict[str, Any]) -> None:
        sent_messages.append(message)
        if message["type"] != "http.response.body" or not message.get("more_body"):
            return
        entered = await asyncio.wait_for(
            asyncio.to_thread(engine.generate_entered.wait, 2),
            timeout=3,
        )
        assert entered is True
        raise OSError("client disconnected during send")

    try:
        with pytest.raises(ClientDisconnect):
            await response(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.4"},
                },
                _receive,
                _send,
            )
        engine.release_generate.set()
        finished = await asyncio.wait_for(
            asyncio.to_thread(agent.run_finished.wait, 2),
            timeout=3,
        )
        assert finished is True
    finally:
        engine.release_generate.set()

    assert [message["type"] for message in sent_messages[:2]] == [
        "http.response.start",
        "http.response.body",
    ]
    assert tool.calls == []
    assert response._bridge._cancellation_token.is_cancelled is True


class _CancelAfterGenerateOrchestrator(OrchestratorAgent):
    def __init__(
        self,
        *args: Any,
        cancellation_token: CancellationToken,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._test_cancellation_token = cancellation_token

    def _generate(self, messages, **extra_kwargs):
        result = super()._generate(messages, **extra_kwargs)
        self._test_cancellation_token.cancel()
        return result


def test_parallel_tool_workers_inherit_cancelled_request_context() -> None:
    engine = _BlockingToolCallEngine()
    engine.release_generate.set()
    tool = _RecordingMutationTool()
    token = CancellationToken()
    agent = _CancelAfterGenerateOrchestrator(
        engine,
        "test-model",
        tools=[tool],
        max_turns=2,
        temperature=0.0,
        max_tokens=64,
        cancellation_token=token,
    )
    _allow_agent(agent)
    context = AgentContext(cancellation_token=token)

    with cancellation_scope(token):
        with pytest.raises(AgentCancelledError):
            agent.run("perform both mutations", context=context)

    assert engine.generate_calls == 1
    assert tool.calls == []


class _BlockingAuthorizationMutationTool(_RecordingMutationTool):
    def __init__(self) -> None:
        super().__init__()
        self._authorization_lock = threading.Lock()
        self.authorization_calls = 0
        self.both_authorizations_entered = threading.Event()
        self.release_authorization = threading.Event()

    def authorization_resource(self, params: dict[str, Any]) -> str:
        del params
        with self._authorization_lock:
            self.authorization_calls += 1
            if self.authorization_calls == 2:
                self.both_authorizations_entered.set()
        if not self.release_authorization.wait(timeout=5):
            raise TimeoutError("test did not release tool authorization")
        return "tool:record_mutation"


@pytest.mark.asyncio
async def test_disconnect_reaches_parallel_workers_blocked_in_authorization() -> None:
    engine = _BlockingToolCallEngine()
    engine.release_generate.set()
    tool = _BlockingAuthorizationMutationTool()
    bus = EventBus()
    agent = _ObservedOrchestrator(
        engine,
        "test-model",
        tools=[tool],
        bus=None,
        max_turns=2,
        temperature=0.0,
        max_tokens=64,
    )
    _allow_agent(agent)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "perform both mutations"}],
        stream=True,
    )
    bridge = AgentStreamBridge(agent, bus, "test-model", request)
    stream = bridge.stream()
    pending_chunk: Optional[asyncio.Task[str]] = None

    try:
        await anext(stream)
        pending_chunk = asyncio.create_task(anext(stream))
        both_entered = await asyncio.wait_for(
            asyncio.to_thread(tool.both_authorizations_entered.wait, 2),
            timeout=3,
        )
        assert both_entered is True

        pending_chunk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_chunk

        tool.release_authorization.set()
        finished = await asyncio.wait_for(
            asyncio.to_thread(agent.run_finished.wait, 2),
            timeout=3,
        )
        assert finished is True
    finally:
        tool.release_authorization.set()
        if pending_chunk is not None and not pending_chunk.done():
            pending_chunk.cancel()
        await stream.aclose()

    assert tool.authorization_calls == 2
    assert tool.calls == []
    assert bridge._cancellation_token.is_cancelled is True


class _CancelDuringAuthorizationPolicy:
    enabled = True

    def __init__(self, token: CancellationToken) -> None:
        self._token = token

    def check(self, agent_id: str, capability: str, resource: str) -> bool:
        del agent_id, capability, resource
        self._token.cancel()
        return True


def test_executor_rechecks_cancellation_after_authorization() -> None:
    token = CancellationToken()
    tool = _RecordingMutationTool()
    executor = ToolExecutor(
        [tool],
        capability_policy=_CancelDuringAuthorizationPolicy(token),
        agent_id="stream-cancellation-test",
    )

    with cancellation_scope(token):
        with pytest.raises(AgentCancelledError):
            executor.execute(
                ToolCall(
                    id="call_1",
                    name="record_mutation",
                    arguments="{}",
                )
            )

    assert tool.calls == []


class _InFlightMutationTool(_RecordingMutationTool):
    def __init__(self) -> None:
        super().__init__()
        self.execution_entered = threading.Event()
        self.release_execution = threading.Event()

    def execute(self, **params: Any) -> ToolResult:
        self.calls.append(dict(params))
        self.execution_entered.set()
        if not self.release_execution.wait(timeout=5):
            raise TimeoutError("test did not release in-flight mutation")
        return ToolResult(
            tool_name=self.spec.name,
            content="mutated",
            success=True,
        )


def test_in_flight_action_keeps_result_and_audit_after_disconnect() -> None:
    token = CancellationToken()
    tool = _InFlightMutationTool()
    bus = EventBus(record_history=True)
    policy = CapabilityPolicy()
    policy.grant("stream-cancellation-test", "*")
    executor = ToolExecutor(
        [tool],
        bus=bus,
        capability_policy=policy,
        agent_id="stream-cancellation-test",
    )
    outcome: dict[str, Any] = {}

    def _run_executor() -> None:
        try:
            with cancellation_scope(token):
                outcome["result"] = executor.execute(
                    ToolCall(
                        id="call_1",
                        name="record_mutation",
                        arguments="{}",
                    )
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            outcome["error"] = exc

    executor_thread = threading.Thread(target=_run_executor)
    executor_thread.start()
    try:
        assert tool.execution_entered.wait(timeout=2) is True
        token.cancel()
        tool.release_execution.set()
        executor_thread.join(timeout=3)
    finally:
        tool.release_execution.set()
        executor_thread.join(timeout=3)

    assert executor_thread.is_alive() is False
    assert "error" not in outcome
    assert tool.calls == [{}]
    result = outcome["result"]
    assert result.success is True
    assert result.metadata["cancellation_requested_after_dispatch"] is True
    end_events = [
        event for event in bus.history if event.event_type == EventType.TOOL_CALL_END
    ]
    assert len(end_events) == 1
    assert (
        end_events[0].data["metadata"]["cancellation_requested_after_dispatch"] is True
    )


class _CooperativelyCancelledInFlightTool(_InFlightMutationTool):
    def execute(self, **params: Any) -> ToolResult:
        self.calls.append(dict(params))
        self.execution_entered.set()
        if not self.release_execution.wait(timeout=5):
            raise TimeoutError("test did not release in-flight mutation")
        token = current_cancellation_token()
        assert token is not None
        token.raise_if_cancelled()
        raise AssertionError("cancelled tool execution unexpectedly continued")


def test_cooperative_in_flight_cancellation_requires_reconciliation() -> None:
    token = CancellationToken()
    tool = _CooperativelyCancelledInFlightTool()
    bus = EventBus(record_history=True)
    policy = CapabilityPolicy()
    policy.grant("stream-cancellation-test", "*")
    executor = ToolExecutor(
        [tool],
        bus=bus,
        capability_policy=policy,
        agent_id="stream-cancellation-test",
    )
    outcome: dict[str, Any] = {}

    def _run_executor() -> None:
        try:
            with cancellation_scope(token):
                executor.execute(
                    ToolCall(
                        id="call_1",
                        name="record_mutation",
                        arguments="{}",
                    )
                )
        except BaseException as exc:
            outcome["error"] = exc

    executor_thread = threading.Thread(target=_run_executor)
    executor_thread.start()
    try:
        assert tool.execution_entered.wait(timeout=2) is True
        token.cancel()
        tool.release_execution.set()
        executor_thread.join(timeout=3)
    finally:
        tool.release_execution.set()
        executor_thread.join(timeout=3)

    assert executor_thread.is_alive() is False
    assert isinstance(outcome.get("error"), AgentCancelledError)
    assert tool.calls == [{}]
    end_events = [
        event for event in bus.history if event.event_type == EventType.TOOL_CALL_END
    ]
    assert len(end_events) == 1
    metadata = end_events[0].data["metadata"]
    assert metadata["cancellation_during_execution"] is True
    assert metadata["outcome"] == "unknown"
    assert metadata["reconcile_required"] is True
    assert "cancelled_before_invocation" not in metadata
