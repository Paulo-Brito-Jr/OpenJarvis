"""Deterministic cooperative-cancellation tests for hybrid agent boundaries."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.hybrid import _base, _openai_retry
from openjarvis.agents.hybrid._base import (
    LocalCloudAgent,
    _OpenRouterLimiter,
    tavily_search_context,
)
from openjarvis.agents.hybrid.minions import (
    _bind_client_cancellation,
    _request_cancellation_token,
)
from openjarvis.agents.hybrid.skillorchestra.pool import ModelSpec
from openjarvis.agents.hybrid.skillorchestra.tools import run_search
from openjarvis.agents.hybrid.toolorchestra import (
    ToolOrchestraAgent,
    _call_modal_python,
)
from openjarvis.core.cancellation import (
    AgentCancelledError,
    CancellationToken,
    cancellation_scope,
)
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor


class _PassBoundaryGuard:
    def scan_outbound(self, content, destination):
        return content


class _HybridProbe(LocalCloudAgent):
    agent_id = "hybrid-cancellation-probe"

    def _run_paradigm(self, input, context, **kwargs):
        self.seen_context = context
        return input, {"turns": 1}


def _authorizer(*capabilities: str) -> ToolExecutor:
    agent_id = "hybrid-cancellation-agent"
    policy = CapabilityPolicy()
    policy.grant(agent_id, "tool:invoke", "*")
    for capability in capabilities:
        policy.grant(agent_id, capability, "*")
    return ToolExecutor(
        [],
        capability_policy=policy,
        agent_id=agent_id,
        boundary_guard=_PassBoundaryGuard(),
    )


def test_guarded_agent_context_preserves_cancellation_token() -> None:
    engine = MagicMock(engine_id="mock")
    agent = _HybridProbe(engine, "test-model", cloud_endpoint="anthropic")
    policy = CapabilityPolicy()
    for capability in ("tool:invoke", "network:fetch", "code:execute"):
        policy.grant("hybrid-cancellation-agent", capability, "*")
    agent.bind_security(
        policy,
        "hybrid-cancellation-agent",
        _PassBoundaryGuard(),
    )
    token = CancellationToken()
    context = AgentContext(
        metadata={"task": {"question": "private"}},
        cancellation_token=token,
    )

    result = agent.run("private input", context)

    assert result.content == "private input"
    assert agent.seen_context is not context
    assert agent.seen_context.cancellation_token is token


def test_cancellation_during_web_search_guard_prevents_dispatch(monkeypatch) -> None:
    token = CancellationToken()
    execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )
    authorizer = _authorizer("network:fetch")
    authorizer.bind_boundary_guard(
        SimpleNamespace(
            scan_outbound=lambda content, destination: token.cancel() or content
        )
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        tavily_search_context(
            "query",
            action_authorizer=authorizer,
        )

    execute.assert_not_called()


def test_cancelled_provider_result_is_audited_without_second_call(
    monkeypatch,
) -> None:
    token = CancellationToken()
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="first result")],
        usage=SimpleNamespace(
            input_tokens=3,
            output_tokens=2,
            server_tool_use=None,
        ),
        stop_reason="max_tokens",
    )

    def complete_first_call(**kwargs):
        token.cancel()
        return message

    create = MagicMock(side_effect=complete_first_call)
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(
            Anthropic=MagicMock(
                return_value=SimpleNamespace(
                    messages=SimpleNamespace(create=create),
                )
            )
        ),
    )
    record_event = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid._base._record_event",
        record_event,
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        LocalCloudAgent._call_anthropic_agent(
            "test-model",
            user="query",
            max_turns=2,
            action_authorizer=_authorizer("network:fetch"),
        )

    create.assert_called_once()
    assert record_event.call_count == 1
    assert record_event.call_args.args[0]["response"] == "first result"


def test_openrouter_rpm_wait_is_cooperatively_cancelled(monkeypatch) -> None:
    token = CancellationToken()
    limiter = _OpenRouterLimiter(max_concurrent=1, rpm=1)
    limiter.record_call()
    sleep_delays = []

    def cancel_on_first_poll(delay):
        sleep_delays.append(delay)
        if len(sleep_delays) > 1:
            raise AssertionError("RPM gate slept again after cancellation")
        token.cancel()

    monkeypatch.setattr(_base.time, "sleep", cancel_on_first_poll)

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        limiter.wait_for_rpm_slot()

    assert sleep_delays
    assert 0 < sleep_delays[0] <= limiter._POLL_INTERVAL_S


def test_openrouter_concurrency_wait_releases_only_acquired_slot() -> None:
    token = CancellationToken()
    limiter = _OpenRouterLimiter(max_concurrent=1, rpm=1)
    semaphore = SimpleNamespace(
        acquire=MagicMock(),
        release=MagicMock(),
    )
    limiter._sem = semaphore

    def cancel_while_waiting(*args, **kwargs):
        assert args == ()
        assert kwargs == {"timeout": limiter._POLL_INTERVAL_S}
        token.cancel()
        return False

    semaphore.acquire.side_effect = cancel_while_waiting

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        limiter.acquire_concurrency()

    semaphore.acquire.assert_called_once()
    semaphore.release.assert_not_called()


def test_openrouter_without_token_uses_blocking_concurrency_fast_path() -> None:
    limiter = _OpenRouterLimiter(max_concurrent=1, rpm=1)
    semaphore = SimpleNamespace(
        acquire=MagicMock(return_value=True),
        release=MagicMock(),
    )
    limiter._sem = semaphore

    limiter.acquire_concurrency()

    semaphore.acquire.assert_called_once_with()
    semaphore.release.assert_not_called()


def test_openrouter_pre_dispatch_cancellation_does_not_record_rpm(
    monkeypatch,
) -> None:
    token = CancellationToken()
    create = MagicMock()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=MagicMock(return_value=client)),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    limiter = SimpleNamespace(
        wait_for_rpm_slot=MagicMock(),
        acquire_concurrency=MagicMock(side_effect=token.cancel),
        release_concurrency=MagicMock(),
        record_call=MagicMock(),
    )
    monkeypatch.setattr(_base, "_openrouter_limiter", lambda: limiter)

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        LocalCloudAgent._call_openrouter(
            "provider/model",
            user="query",
            action_authorizer=_authorizer("network:fetch"),
        )

    create.assert_not_called()
    limiter.release_concurrency.assert_called_once_with()
    limiter.record_call.assert_not_called()


def test_toolorchestra_cancelled_orchestrator_never_dispatches_worker(
    monkeypatch,
) -> None:
    token = CancellationToken()
    agent = ToolOrchestraAgent(
        MagicMock(engine_id="mock"),
        "test-model",
        cfg={
            "max_turns": 2,
            "workers": [
                {
                    "id": 0,
                    "name": "worker",
                    "type": "anthropic",
                    "model": "worker-model",
                    "description": "test worker",
                }
            ],
        },
    )

    def complete_orchestrator(**kwargs):
        token.cancel()
        return ('{"action":"call_worker","worker_id":0,"input":"next"}', 1, 1)

    agent._call_cloud = MagicMock(side_effect=complete_orchestrator)
    dispatch = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._call_worker",
        dispatch,
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        agent._run_prompted("question", None)

    agent._call_cloud.assert_called_once()
    dispatch.assert_not_called()


def test_cancellation_during_retriever_guard_prevents_http_dispatch(
    monkeypatch,
) -> None:
    token = CancellationToken()
    http_execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.skillorchestra.tools.HttpRequestTool.execute",
        http_execute,
    )
    action_authorizer = SimpleNamespace(
        confirm_action=MagicMock(return_value=None),
        guard_outbound_content=lambda content, destination: token.cancel() or content,
    )
    agent = SimpleNamespace(
        _action_authorizer=action_authorizer,
        _require_action=MagicMock(),
        _call_vllm=MagicMock(return_value=("<query>lookup value</query>", 2, 1)),
        record_trace_event=MagicMock(),
    )
    spec = ModelSpec(
        alias="search-3",
        model="local-model",
        endpoint="http://localhost:8001/v1",
        kind="local",
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        run_search(
            agent,
            spec,
            context_str="context",
            problem="problem",
            retriever_url="https://retriever.example",
        )

    http_execute.assert_not_called()


def test_cancellation_after_modal_lookup_prevents_sandbox_dispatch(
    monkeypatch,
) -> None:
    token = CancellationToken()

    def lookup(*args, **kwargs):
        token.cancel()
        return object()

    sandbox_create = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=SimpleNamespace(lookup=lookup),
            Image=SimpleNamespace(debian_slim=MagicMock(return_value=object())),
            Sandbox=SimpleNamespace(create=sandbox_create),
        ),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._require_action_authorized",
        MagicMock(),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._guard_provider_text",
        lambda authorizer, content, destination: content,
    )
    record_event = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._record_event",
        record_event,
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        _call_modal_python("print('never dispatched')")

    sandbox_create.assert_not_called()
    assert record_event.call_args.args[0]["kind"] == "modal_app_lookup"


def test_cancellation_after_modal_create_terminates_without_wait(
    monkeypatch,
) -> None:
    token = CancellationToken()
    sandbox = SimpleNamespace(
        wait=MagicMock(),
        terminate=MagicMock(),
    )

    def create(*args, **kwargs):
        token.cancel()
        return sandbox

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=SimpleNamespace(lookup=MagicMock(return_value=object())),
            Image=SimpleNamespace(debian_slim=MagicMock(return_value=object())),
            Sandbox=SimpleNamespace(create=MagicMock(side_effect=create)),
        ),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._require_action_authorized",
        MagicMock(),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._guard_provider_text",
        lambda authorizer, content, destination: content,
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._record_event",
        MagicMock(),
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        _call_modal_python("print('created')")

    sandbox.wait.assert_not_called()
    sandbox.terminate.assert_called_once_with()


def test_cancellation_during_modal_wait_terminates_promptly_once(
    monkeypatch,
) -> None:
    token = CancellationToken()
    wait_started = threading.Event()
    release_wait = threading.Event()
    caller_finished = threading.Event()
    wait_thread_daemon = []
    caller_errors = []

    def wait():
        wait_thread_daemon.append(threading.current_thread().daemon)
        wait_started.set()
        release_wait.wait(timeout=2)

    def terminate():
        release_wait.set()

    sandbox = SimpleNamespace(
        wait=MagicMock(side_effect=wait),
        terminate=MagicMock(side_effect=terminate),
        stdout=SimpleNamespace(read=MagicMock(return_value="")),
        stderr=SimpleNamespace(read=MagicMock(return_value="")),
        returncode=0,
    )
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=SimpleNamespace(lookup=MagicMock(return_value=object())),
            Image=SimpleNamespace(debian_slim=MagicMock(return_value=object())),
            Sandbox=SimpleNamespace(create=MagicMock(return_value=sandbox)),
        ),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._require_action_authorized",
        MagicMock(),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._guard_provider_text",
        lambda authorizer, content, destination: content,
    )
    record_event = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._record_event",
        record_event,
    )

    def call_modal():
        try:
            with cancellation_scope(token):
                _call_modal_python("print('in flight')")
        except BaseException as exc:  # noqa: BLE001
            caller_errors.append(exc)
        finally:
            caller_finished.set()

    caller = threading.Thread(target=call_modal)
    caller.start()
    try:
        assert wait_started.wait(timeout=1)
        token.cancel()
        assert caller_finished.wait(timeout=1)
    finally:
        token.cancel()
        release_wait.set()
        caller.join(timeout=1)

    assert not caller.is_alive()
    assert wait_thread_daemon == [True]
    assert len(caller_errors) == 1
    assert isinstance(caller_errors[0], AgentCancelledError)
    sandbox.terminate.assert_called_once_with()
    assert not any(
        call.args[0].get("kind") == "modal_python"
        for call in record_event.call_args_list
    )


def test_completed_modal_wait_is_audited_and_terminated_once(
    monkeypatch,
) -> None:
    token = CancellationToken()
    sandbox = SimpleNamespace(
        wait=MagicMock(side_effect=token.cancel),
        terminate=MagicMock(),
        stdout=SimpleNamespace(read=MagicMock(return_value="completed")),
        stderr=SimpleNamespace(read=MagicMock(return_value="warning")),
        returncode=0,
    )
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=SimpleNamespace(lookup=MagicMock(return_value=object())),
            Image=SimpleNamespace(debian_slim=MagicMock(return_value=object())),
            Sandbox=SimpleNamespace(create=MagicMock(return_value=sandbox)),
        ),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._require_action_authorized",
        MagicMock(),
    )
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._guard_provider_text",
        lambda authorizer, content, destination: content,
    )
    record_event = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.toolorchestra._record_event",
        record_event,
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        _call_modal_python("print('completed')")

    sandbox.terminate.assert_called_once_with()
    modal_events = [
        call.args[0]
        for call in record_event.call_args_list
        if call.args[0].get("kind") == "modal_python"
    ]
    assert len(modal_events) == 1
    assert modal_events[0]["response"] == "completed\nwarning"
    assert modal_events[0]["returncode"] == 0


@pytest.mark.parametrize("method_name", ["chat", "schat"])
def test_minions_worker_token_stops_second_round(method_name) -> None:
    token = CancellationToken()
    calls = []
    errors = []

    def complete_first_round(*args, **kwargs):
        calls.append((args, kwargs))
        token.cancel()
        return "completed first round"

    client = SimpleNamespace()
    setattr(client, method_name, complete_first_round)
    _bind_client_cancellation(client, token)

    def run_worker_rounds():
        try:
            for _round in range(2):
                getattr(client, method_name)("prompt")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=run_worker_rounds)
    worker.start()
    worker.join()

    assert len(calls) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], AgentCancelledError)


def test_minions_prefers_live_scope_token_over_context_default() -> None:
    live_token = CancellationToken()
    context_token = CancellationToken()
    context = AgentContext(cancellation_token=context_token)

    with cancellation_scope(live_token):
        selected = _request_cancellation_token(context)

    assert selected is live_token


def test_openai_retry_cancelled_attempt_never_retries(monkeypatch) -> None:
    token = CancellationToken()
    calls = MagicMock()
    sleep = MagicMock()

    def retryable_first_attempt(self, *args, **kwargs):
        calls()
        token.cancel()
        raise RuntimeError("retryable")

    monkeypatch.setattr(_openai_retry, "_is_retryable", lambda exc: True)
    monkeypatch.setattr(_openai_retry.time, "sleep", sleep)
    wrapped = _openai_retry._wrap_create(retryable_first_attempt)
    resource = SimpleNamespace(
        _client=SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        wrapped(resource)

    calls.assert_called_once_with()
    sleep.assert_not_called()


def test_openai_retry_cancelled_while_waiting_never_dispatches(
    monkeypatch,
) -> None:
    token = CancellationToken()
    dispatch = MagicMock()
    semaphore = threading.BoundedSemaphore(1)
    assert semaphore.acquire(blocking=False)
    acquire_attempted = threading.Event()
    worker_finished = threading.Event()
    worker_errors = []
    original_acquire = semaphore.acquire

    def observed_acquire(*args, **kwargs):
        acquire_attempted.set()
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(semaphore, "acquire", observed_acquire)
    monkeypatch.setattr(_openai_retry, "_SEM", semaphore)
    wrapped = _openai_retry._wrap_create(dispatch)
    resource = SimpleNamespace(
        _client=SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
    )

    def wait_for_cloud_slot():
        try:
            with cancellation_scope(token):
                wrapped(resource)
        except BaseException as exc:  # noqa: BLE001
            worker_errors.append(exc)
        finally:
            worker_finished.set()

    worker = threading.Thread(target=wait_for_cloud_slot)
    worker.start()
    try:
        assert acquire_attempted.wait(timeout=1)
        token.cancel()
        assert worker_finished.wait(timeout=1)
        assert not worker.is_alive()
    finally:
        token.cancel()
        try:
            semaphore.release()
        finally:
            worker.join(timeout=1)

    dispatch.assert_not_called()
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], AgentCancelledError)


def test_openai_retry_returns_completed_result_for_caller_audit(
    monkeypatch,
) -> None:
    token = CancellationToken()
    completed = object()
    calls = MagicMock()
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(_openai_retry, "_SEM", semaphore)

    def complete_in_flight(self):
        calls()
        token.cancel()
        return completed

    wrapped = _openai_retry._wrap_create(complete_in_flight)
    resource = SimpleNamespace(
        _client=SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
    )

    with cancellation_scope(token):
        result = wrapped(resource)

    assert result is completed
    calls.assert_called_once_with()
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def test_openai_retry_null_semaphore_dispatches_once(monkeypatch) -> None:
    completed = object()
    dispatch = MagicMock(return_value=completed)
    monkeypatch.setattr(_openai_retry, "_SEM", _openai_retry._NullSem())
    wrapped = _openai_retry._wrap_create(dispatch)
    resource = SimpleNamespace(
        _client=SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
    )

    assert wrapped(resource) is completed
    dispatch.assert_called_once_with(resource)


def test_openai_retry_real_semaphore_without_token_dispatches_once(
    monkeypatch,
) -> None:
    completed = object()
    dispatch = MagicMock(return_value=completed)
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(_openai_retry, "_SEM", semaphore)
    wrapped = _openai_retry._wrap_create(dispatch)
    resource = SimpleNamespace(
        _client=SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
    )

    assert wrapped(resource) is completed
    dispatch.assert_called_once_with(resource)
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def test_openai_retry_backoff_is_cooperatively_cancelled(monkeypatch) -> None:
    token = CancellationToken()
    calls = MagicMock(side_effect=RuntimeError("retryable"))

    def cancel_instead_of_sleep(_delay):
        token.cancel()

    sleep = MagicMock(side_effect=cancel_instead_of_sleep)
    monkeypatch.setattr(_openai_retry, "_is_retryable", lambda exc: True)
    monkeypatch.setattr(_openai_retry, "_sleep_for", lambda attempt, exc: 1.0)
    monkeypatch.setattr(_openai_retry.time, "sleep", sleep)
    wrapped = _openai_retry._wrap_create(calls)
    resource = SimpleNamespace(
        _client=SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
    )

    with cancellation_scope(token), pytest.raises(AgentCancelledError):
        wrapped(resource)

    calls.assert_called_once_with(resource)
    sleep.assert_called_once()
