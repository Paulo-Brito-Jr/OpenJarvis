"""Tests for worker-scoped, revocable tool authorization receipts."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._execution_context import _authorized_resource
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec
from openjarvis.tools.file_write import FileWriteTool

_AGENT_ID = "receipt-test"
_DIRECT_DENIAL = "Security block: authenticated ToolExecutor dispatch required."


def _resource(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _executor(
    tool: BaseTool,
    resource: str,
    *capabilities: str,
) -> ToolExecutor:
    policy = CapabilityPolicy()
    policy.grant(_AGENT_ID, "tool:invoke", resource)
    for capability in capabilities:
        policy.grant(_AGENT_ID, capability, resource)
    return ToolExecutor(
        [tool],
        capability_policy=policy,
        agent_id=_AGENT_ID,
    )


class _SingleUseProbeTool(BaseTool):
    tool_id = "receipt_probe"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="receipt_probe", description="Probe receipt lifetime.")

    def execute(self, **params) -> ToolResult:
        requested = params["path"]
        first = _authorized_resource(self.spec.name, requested)
        second = _authorized_resource(self.spec.name, requested)
        return ToolResult(
            tool_name=self.spec.name,
            content=json.dumps({"first": first, "second": second}),
            success=True,
        )


class _DelayedFileWriteTool(FileWriteTool):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.finished = Event()
        self.observed_after_timeout: list[ToolResult] = []

    @property
    def spec(self) -> ToolSpec:
        spec = super().spec
        spec.timeout_seconds = 0.5
        return spec

    def execute(self, **params) -> ToolResult:
        self.started.set()
        self.release.wait(timeout=3)
        result = super().execute(**params)
        self.observed_after_timeout.append(result)
        self.finished.set()
        return result


def test_receipt_is_single_use_and_bound_to_worker(tmp_path):
    requested = str(tmp_path / "target.txt")
    canonical = _resource(requested)
    tool = _SingleUseProbeTool()

    result = _executor(tool, canonical).execute(
        ToolCall(
            id="single-use",
            name=tool.spec.name,
            arguments=json.dumps({"path": requested}),
        )
    )
    observed = json.loads(result.content)

    assert result.success is True
    assert observed == {"first": canonical, "second": None}
    assert _authorized_resource(tool.spec.name, requested) is None


def test_timeout_revokes_receipt_before_late_worker_consumes_it(tmp_path):
    target = tmp_path / "target.txt"
    requested = str(target)
    tool = _DelayedFileWriteTool()
    results: list[ToolResult] = []
    executor = _executor(tool, _resource(requested), "file:write")
    call = ToolCall(
        id="timeout",
        name=tool.spec.name,
        arguments=json.dumps(
            {
                "path": requested,
                "content": "late write must be blocked",
            }
        ),
    )

    executor_thread = Thread(
        target=lambda: results.append(executor.execute(call)),
        daemon=True,
    )
    executor_thread.start()
    assert tool.started.wait(timeout=2)
    executor_thread.join(timeout=2)

    assert executor_thread.is_alive() is False
    assert len(results) == 1
    result = results[0]
    assert tool.started.is_set()
    assert result.success is False
    assert result.metadata["reconcile_required"] is True
    assert _authorized_resource(tool.spec.name, requested) is None
    assert not target.exists()

    tool.release.set()
    assert tool.finished.wait(timeout=2)
    assert len(tool.observed_after_timeout) == 1
    assert tool.observed_after_timeout[0].success is False
    assert tool.observed_after_timeout[0].content == _DIRECT_DENIAL
    assert not target.exists()

    direct = FileWriteTool().execute(
        path=requested,
        content="direct write must remain blocked",
    )
    assert direct.success is False
    assert direct.content == _DIRECT_DENIAL
    assert not target.exists()
