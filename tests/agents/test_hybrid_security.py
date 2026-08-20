"""Fail-closed authorization tests for hybrid provider-side actions."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.hybrid._base import (
    LocalCloudAgent,
    build_web_search_tool,
    tavily_search_context,
)
from openjarvis.agents.hybrid.mini_swe_agent import (
    _run_bash,
    run_swe_agent_loop,
)
from openjarvis.agents.hybrid.runner import _build_agent
from openjarvis.agents.hybrid.skillorchestra.pool import ModelSpec
from openjarvis.agents.hybrid.skillorchestra.tools import (
    run_code,
    run_search,
)
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor


class _HybridProbe(LocalCloudAgent):
    agent_id = "hybrid-probe"

    def _run_paradigm(self, input, context, **kwargs):
        return input, {}


class _PassBoundaryGuard:
    def scan_outbound(self, content, destination):
        return content


def _web_search_authorizer(
    *,
    grant_network: bool,
    agent_id: str = "hybrid-test-agent",
) -> ToolExecutor:
    policy = CapabilityPolicy()
    policy.grant(agent_id, "tool:invoke", "*")
    if grant_network:
        policy.grant(agent_id, "network:fetch", "*")
    return ToolExecutor(
        [],
        capability_policy=policy,
        agent_id=agent_id,
        boundary_guard=_PassBoundaryGuard(),
    )


def test_tavily_missing_security_context_denies_before_tool_execution(
    monkeypatch,
):
    execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )

    with pytest.raises(PermissionError, match="policy unavailable"):
        tavily_search_context("query")

    execute.assert_not_called()


def test_tavily_missing_identity_denies_before_tool_execution(monkeypatch):
    execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )
    authorizer = _web_search_authorizer(
        grant_network=True,
        agent_id="",
    )

    with pytest.raises(PermissionError, match="identity unavailable"):
        tavily_search_context(
            "query",
            action_authorizer=authorizer,
        )

    execute.assert_not_called()


def test_tavily_missing_network_capability_denies(monkeypatch):
    execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )
    authorizer = _web_search_authorizer(grant_network=False)

    with pytest.raises(PermissionError, match="network:fetch"):
        tavily_search_context(
            "query",
            action_authorizer=authorizer,
        )

    execute.assert_not_called()


def test_tavily_missing_boundary_guard_denies_before_tool_execution(monkeypatch):
    execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )
    policy = CapabilityPolicy()
    policy.grant("hybrid-test-agent", "tool:invoke", "*")
    policy.grant("hybrid-test-agent", "network:fetch", "*")
    authorizer = ToolExecutor(
        [],
        capability_policy=policy,
        agent_id="hybrid-test-agent",
    )

    with pytest.raises(PermissionError, match="Boundary guard unavailable"):
        tavily_search_context(
            "query",
            action_authorizer=authorizer,
        )

    execute.assert_not_called()


def test_tavily_minimal_grants_allow_execution(monkeypatch):
    execute = MagicMock(
        return_value=ToolResult(
            tool_name="web_search",
            content="grounded result",
            success=True,
            metadata={"engine": "duckduckgo"},
        )
    )
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )
    authorizer = _web_search_authorizer(grant_network=True)

    result = tavily_search_context(
        "query",
        action_authorizer=authorizer,
    )

    execute.assert_called_once_with(query="query", max_results=5)
    assert result["text"] == "grounded result"
    assert result["success"] is True


def test_tavily_uses_boundary_guard_output(monkeypatch):
    execute = MagicMock(
        return_value=ToolResult(
            tool_name="web_search",
            content="safe",
            success=True,
            metadata={"engine": "duckduckgo"},
        )
    )
    monkeypatch.setattr(
        "openjarvis.tools.web_search.WebSearchTool.execute",
        execute,
    )
    authorizer = _web_search_authorizer(grant_network=True)
    authorizer.bind_boundary_guard(
        SimpleNamespace(scan_outbound=lambda content, destination: "[REDACTED]")
    )

    tavily_search_context(
        "secret query",
        action_authorizer=authorizer,
    )

    execute.assert_called_once_with(query="[REDACTED]", max_results=5)


@pytest.mark.parametrize(
    "provider_call",
    [
        lambda: LocalCloudAgent._call_anthropic(
            "test-model",
            user="query",
            tools=[build_web_search_tool(1)],
        ),
        lambda: LocalCloudAgent._call_openai_agent(
            "test-model",
            user="query",
        ),
        lambda: LocalCloudAgent._call_gemini_agent(
            "test-model",
            user="query",
        ),
    ],
)
def test_provider_web_search_without_authorizer_fails_closed(provider_call):
    with pytest.raises(PermissionError, match="policy unavailable"):
        provider_call()


def test_provider_web_search_missing_network_capability_never_calls_sdk(
    monkeypatch,
):
    anthropic_ctor = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=anthropic_ctor),
    )
    authorizer = _web_search_authorizer(grant_network=False)

    with pytest.raises(PermissionError, match="network:fetch"):
        LocalCloudAgent._call_anthropic(
            "test-model",
            user="query",
            tools=[build_web_search_tool(1)],
            action_authorizer=authorizer,
        )

    anthropic_ctor.assert_not_called()


def test_provider_web_search_minimal_grants_reach_sdk(monkeypatch):
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="grounded answer")],
        usage=SimpleNamespace(
            input_tokens=3,
            output_tokens=2,
            server_tool_use=SimpleNamespace(web_search_requests=1),
        ),
        stop_reason="end_turn",
    )
    create = MagicMock(return_value=message)
    anthropic_ctor = MagicMock(
        return_value=SimpleNamespace(
            messages=SimpleNamespace(create=create),
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=anthropic_ctor),
    )
    authorizer = _web_search_authorizer(grant_network=True)

    text, prompt_tokens, completion_tokens, searches = LocalCloudAgent._call_anthropic(
        "test-model",
        user="query",
        tools=[build_web_search_tool(1)],
        action_authorizer=authorizer,
    )

    anthropic_ctor.assert_called_once()
    create.assert_called_once()
    assert (text, prompt_tokens, completion_tokens, searches) == (
        "grounded answer",
        3,
        2,
        1,
    )


def test_hybrid_runner_binds_explicit_policy_and_identity(monkeypatch):
    monkeypatch.setattr(
        AgentRegistry,
        "contains",
        classmethod(lambda cls, key: key == "hybrid-probe"),
    )
    monkeypatch.setattr(
        AgentRegistry,
        "get",
        classmethod(lambda cls, key: _HybridProbe),
    )
    policy = CapabilityPolicy()
    policy.grant("cell-1", "tool:invoke", "web_search")
    policy.grant("cell-1", "network:fetch", "web_search")

    agent = _build_agent(
        {
            "method": "hybrid-probe",
            "cloud": {"model": "test-model"},
        },
        capability_policy=policy,
        agent_id="cell-1",
        boundary_guard=_PassBoundaryGuard(),
    )

    agent._require_action("web_search")


def test_hybrid_run_without_bound_policy_never_reaches_paradigm():
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(
        engine,
        "test-model",
        cloud_endpoint="anthropic",
    )
    run_paradigm = MagicMock()
    agent._run_paradigm = run_paradigm

    result = agent.run("do work")

    assert result.turns == 0
    assert result.metadata["security_denied"] is True
    assert result.metadata["resource"] == "https://api.anthropic.com"
    assert "policy unavailable" in result.content.lower()
    run_paradigm.assert_not_called()


def test_hybrid_run_requires_code_execution_before_paradigm():
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(
        engine,
        "test-model",
        cloud_endpoint="anthropic",
    )
    policy = CapabilityPolicy()
    resource = "https://api.anthropic.com"
    policy.grant("hybrid-agent", "tool:invoke", resource)
    policy.grant("hybrid-agent", "network:fetch", resource)
    agent.bind_security(policy, "hybrid-agent")
    run_paradigm = MagicMock()
    agent._run_paradigm = run_paradigm

    result = agent.run("do work")

    assert result.metadata["security_denied"] is True
    assert "code:execute" in result.content
    run_paradigm.assert_not_called()


def test_hybrid_run_missing_boundary_guard_never_reaches_paradigm():
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(
        engine,
        "test-model",
        cloud_endpoint="anthropic",
    )
    policy = CapabilityPolicy()
    resource = "https://api.anthropic.com"
    for capability in ("tool:invoke", "network:fetch", "code:execute"):
        policy.grant("hybrid-agent", capability, resource)
    agent.bind_security(policy, "hybrid-agent")
    run_paradigm = MagicMock()
    agent._run_paradigm = run_paradigm

    result = agent.run("do work")

    assert result.metadata["security_denied"] is True
    assert "boundary guard unavailable" in result.content.lower()
    run_paradigm.assert_not_called()


def test_hybrid_security_rebind_clears_stale_boundary_guard():
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(
        engine,
        "test-model",
        cloud_endpoint="anthropic",
    )
    policy = CapabilityPolicy()
    resource = "https://api.anthropic.com"
    for capability in ("tool:invoke", "network:fetch", "code:execute"):
        policy.grant("hybrid-agent", capability, resource)
    agent.bind_security(policy, "hybrid-agent", _PassBoundaryGuard())
    agent.bind_security(policy, "hybrid-agent")
    run_paradigm = MagicMock()
    agent._run_paradigm = run_paradigm

    result = agent.run("do work")

    assert result.metadata["security_denied"] is True
    assert "boundary guard unavailable" in result.content.lower()
    run_paradigm.assert_not_called()


def test_hybrid_run_guards_input_and_task_metadata_before_paradigm():
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(
        engine,
        "test-model",
        cloud_endpoint="anthropic",
    )
    policy = CapabilityPolicy()
    resource = "https://api.anthropic.com"
    for capability in ("tool:invoke", "network:fetch", "code:execute"):
        policy.grant("hybrid-agent", capability, resource)
    boundary_guard = SimpleNamespace(
        scan_outbound=lambda content, destination: f"[SAFE:{destination}]"
    )
    agent.bind_security(policy, "hybrid-agent", boundary_guard)
    run_paradigm = MagicMock(return_value=("safe answer", {}))
    agent._run_paradigm = run_paradigm
    context = AgentContext(
        metadata={
            "task": {
                "problem_statement": "private task",
                "nested": ["private hint"],
            },
            "task_id": "task-1",
        }
    )

    result = agent.run("private input", context)

    assert result.content == "safe answer"
    run_paradigm.assert_called_once()
    guarded_input, guarded_context = run_paradigm.call_args.args
    assert guarded_input == f"[SAFE:{resource}]"
    assert guarded_context.metadata["task"] == {
        "problem_statement": f"[SAFE:{resource}]",
        "nested": [f"[SAFE:{resource}]"],
    }
    assert guarded_context.metadata["task_id"] == "task-1"
    assert context.metadata["task"]["problem_statement"] == "private task"


def test_hybrid_trace_requires_file_write_before_paradigm(tmp_path):
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(engine, "test-model", cloud_endpoint="anthropic")
    policy = CapabilityPolicy()
    resource = "https://api.anthropic.com"
    policy.grant("hybrid-agent", "tool:invoke", "*")
    for capability in ("network:fetch", "code:execute"):
        policy.grant("hybrid-agent", capability, resource)
    agent.bind_security(policy, "hybrid-agent", _PassBoundaryGuard())
    run_paradigm = MagicMock(return_value=("safe answer", {}))
    agent._run_paradigm = run_paradigm
    context = AgentContext(metadata={"log_dir": str(tmp_path), "task_id": "task-1"})

    result = agent.run("private input", context)

    assert result.metadata["security_denied"] is True
    assert "file:write" in result.content
    run_paradigm.assert_not_called()
    assert not (tmp_path / "task-1.json").exists()


def test_hybrid_trace_rejects_task_id_traversal_before_paradigm(tmp_path):
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(engine, "test-model", cloud_endpoint="anthropic")
    policy = CapabilityPolicy()
    for capability in (
        "tool:invoke",
        "network:fetch",
        "code:execute",
        "file:write",
    ):
        policy.grant("hybrid-agent", capability, "*")
    agent.bind_security(policy, "hybrid-agent", _PassBoundaryGuard())
    run_paradigm = MagicMock(return_value=("safe answer", {}))
    agent._run_paradigm = run_paradigm
    context = AgentContext(metadata={"log_dir": str(tmp_path), "task_id": "../escaped"})

    result = agent.run("private input", context)

    assert result.metadata["security_denied"] is True
    assert "trace configuration invalid" in result.content.lower()
    run_paradigm.assert_not_called()
    assert not (tmp_path.parent / "escaped.json").exists()


def test_hybrid_trace_writes_only_to_authorized_canonical_path(tmp_path):
    engine = MagicMock()
    engine.engine_id = "mock"
    agent = _HybridProbe(engine, "test-model", cloud_endpoint="anthropic")
    policy = CapabilityPolicy()
    resource = "https://api.anthropic.com"
    target = tmp_path / "task-1.json"
    policy.grant("hybrid-agent", "tool:invoke", "*")
    policy.grant("hybrid-agent", "network:fetch", resource)
    policy.grant("hybrid-agent", "code:execute", resource)
    policy.grant("hybrid-agent", "file:write", str(target))
    agent.bind_security(policy, "hybrid-agent", _PassBoundaryGuard())
    agent._run_paradigm = MagicMock(return_value=("safe answer", {"turns": 1}))
    context = AgentContext(metadata={"log_dir": str(tmp_path), "task_id": "task-1"})

    result = agent.run("private input", context)

    assert result.content == "safe answer"
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text())["task_id"] == "task-1"


def test_skillorchestra_generated_python_never_executes_on_host(monkeypatch):
    host_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", host_run)
    agent = SimpleNamespace(
        _require_action=MagicMock(),
        _call_vllm=MagicMock(
            return_value=("```python\nprint('must not run')\n```", 2, 1)
        ),
    )
    spec = ModelSpec(
        alias="reasoner-3",
        model="local-model",
        endpoint="http://localhost:8001/v1",
        kind="local",
    )

    result = run_code(
        agent,
        spec,
        context_str="context",
        problem="problem",
    )

    host_run.assert_not_called()
    agent._require_action.assert_called_once_with(
        "code:skillorchestra",
        ["code:execute"],
        tool_name="skillorchestra_code",
    )
    assert result["security_disabled"] is True
    assert "isolated sandbox" in result["exec_result"]


def test_mini_swe_public_and_shell_helpers_never_execute_on_host(
    monkeypatch,
    tmp_path,
):
    host_run = MagicMock()
    host_popen = MagicMock()
    monkeypatch.setattr(subprocess, "run", host_run)
    monkeypatch.setattr(subprocess, "Popen", host_popen)

    result = run_swe_agent_loop(
        {
            "repo": "owner/repo",
            "base_commit": "deadbeef",
            "task_id": "task-1",
        },
        backbone="cloud",
        backbone_model="test-model",
    )
    shell_result = _run_bash("touch escaped", tmp_path)

    host_run.assert_not_called()
    host_popen.assert_not_called()
    assert result["security_disabled"] is True
    assert result["patch"] == ""
    assert shell_result["security_disabled"] is True
    assert not (tmp_path / "escaped").exists()


def test_skillorchestra_retriever_uses_ssrf_safe_transport_and_dlp(
    monkeypatch,
):
    captured = {}

    def safe_http_execute(_self, **params):
        captured.update(params)
        return ToolResult(
            tool_name="http_request",
            content=json.dumps(
                [[{"document": {"content": "safe retrieved document"}}]]
            ),
            success=True,
            metadata={"status_code": 200},
        )

    monkeypatch.setattr(
        "openjarvis.agents.hybrid.skillorchestra.tools.HttpRequestTool.execute",
        safe_http_execute,
    )
    action_authorizer = SimpleNamespace(
        guard_outbound_content=lambda content, destination: "[REDACTED]",
        confirm_action=MagicMock(return_value=None),
    )
    agent = SimpleNamespace(
        _action_authorizer=action_authorizer,
        _require_action=MagicMock(),
        _call_vllm=MagicMock(
            return_value=("<query>private lookup value</query>", 2, 1)
        ),
    )
    spec = ModelSpec(
        alias="search-3",
        model="local-model",
        endpoint="http://localhost:8001/v1",
        kind="local",
    )

    result = run_search(
        agent,
        spec,
        context_str="context",
        problem="problem",
        retriever_url="https://retriever.example",
    )

    endpoint = "https://retriever.example/retrieve"
    agent._require_action.assert_called_once_with(
        endpoint,
        ["network:fetch", "network:mutate"],
        tool_name="skillorchestra_retriever",
    )
    action_authorizer.confirm_action.assert_called_once_with(
        "skillorchestra_retriever",
        {"url": endpoint, "method": "POST"},
    )
    assert captured["url"] == endpoint
    assert captured["method"] == "POST"
    assert json.loads(captured["body"])["queries"] == ["[REDACTED]"]
    assert result["search_results_data"] == ["safe retrieved document"]


def test_skillorchestra_retriever_requires_central_confirmation(monkeypatch):
    http_execute = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.hybrid.skillorchestra.tools.HttpRequestTool.execute",
        http_execute,
    )
    denied = ToolResult(
        tool_name="skillorchestra_retriever",
        content="live confirmation unavailable",
        success=False,
    )
    action_authorizer = SimpleNamespace(
        guard_outbound_content=MagicMock(),
        confirm_action=MagicMock(return_value=denied),
    )
    agent = SimpleNamespace(
        _action_authorizer=action_authorizer,
        _require_action=MagicMock(),
        _call_vllm=MagicMock(
            return_value=("<query>private lookup value</query>", 2, 1)
        ),
    )
    spec = ModelSpec(
        alias="search-3",
        model="local-model",
        endpoint="http://localhost:8001/v1",
        kind="local",
    )

    with pytest.raises(PermissionError, match="confirmation unavailable"):
        run_search(
            agent,
            spec,
            context_str="context",
            problem="problem",
            retriever_url="https://retriever.example",
        )

    agent._require_action.assert_called_once_with(
        "https://retriever.example/retrieve",
        ["network:fetch", "network:mutate"],
        tool_name="skillorchestra_retriever",
    )
    http_execute.assert_not_called()
