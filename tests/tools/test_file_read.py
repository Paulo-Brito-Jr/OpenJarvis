"""Tests for the fail-closed ``file_read`` tool boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openjarvis.core.types import ToolCall
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.file_read import FileReadTool

_AGENT_ID = "file-read-test"
_DIRECT_DENIAL = "Security block: authenticated ToolExecutor dispatch required."


def _canonical(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _call(
    tool: FileReadTool,
    params: dict,
    *,
    policy: CapabilityPolicy | None,
    agent_id: str = _AGENT_ID,
):
    executor = ToolExecutor(
        [tool],
        capability_policy=policy,
        agent_id=agent_id,
    )
    return executor.execute(
        ToolCall(
            id="file-read-call",
            name="file_read",
            arguments=json.dumps(params),
        )
    )


def _policy(
    resource: str,
    *,
    include_file_read: bool = True,
    file_read_resource: str | None = None,
) -> CapabilityPolicy:
    policy = CapabilityPolicy()
    policy.grant(_AGENT_ID, "tool:invoke", resource)
    if include_file_read:
        policy.grant(
            _AGENT_ID,
            "file:read",
            file_read_resource or resource,
        )
    return policy


def _authorized_call(tool: FileReadTool, params: dict):
    resource = _canonical(params["path"])
    return _call(tool, params, policy=_policy(resource))


class TestFileReadTool:
    def test_spec_declares_file_read_capability(self):
        tool = FileReadTool()

        assert tool.spec.name == "file_read"
        assert tool.spec.category == "filesystem"
        assert "file:read" in tool.spec.required_capabilities

    def test_no_path(self):
        result = FileReadTool().execute(path="")

        assert result.success is False
        assert "No path" in result.content

    @pytest.mark.parametrize("invalid_path", ["   ", ["not", "a", "path"]])
    def test_invalid_path_cannot_fall_back_to_tool_resource(
        self,
        tmp_path,
        invalid_path,
    ):
        policy = CapabilityPolicy()
        policy.grant(_AGENT_ID, "tool:invoke", "tool:file_read")
        policy.grant(_AGENT_ID, "file:read", "tool:file_read")
        before = set(tmp_path.iterdir())

        result = _call(
            FileReadTool(),
            {"path": invalid_path},
            policy=policy,
        )

        assert result.success is False
        assert "authorization metadata invalid" in result.content.lower()
        assert set(tmp_path.iterdir()) == before

    def test_direct_missing_path_is_denied_without_existence_leak(self, tmp_path):
        missing = tmp_path / "missing-marker.txt"

        result = FileReadTool().execute(path=str(missing))

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert str(missing) not in result.content
        assert "not found" not in result.content.lower()

    def test_direct_existing_read_is_denied_without_content_leak(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("private marker", encoding="utf-8")

        result = FileReadTool().execute(
            path=str(target),
            max_lines=1,
            _authorization_receipt="forged",
            agent_id=_AGENT_ID,
        )

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert "private marker" not in result.content

    def test_authorized_missing_file_reports_not_found(self, tmp_path):
        missing = tmp_path / "missing.txt"

        result = _authorized_call(FileReadTool(), {"path": str(missing)})

        assert result.success is False
        assert "File not found" in result.content

    def test_authorized_read_uses_exact_grant(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("hello world\nsecond line\n", encoding="utf-8")

        result = _authorized_call(FileReadTool(), {"path": str(target)})

        assert result.success is True
        assert result.content == "hello world\nsecond line\n"
        assert result.metadata["path"] == _canonical(target)
        assert result.metadata["size_bytes"] > 0

    def test_authorized_max_lines(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("line1\nline2\nline3\n", encoding="utf-8")

        result = _authorized_call(
            FileReadTool(),
            {"path": str(target), "max_lines": 2},
        )

        assert result.success is True
        assert result.content == "line1\nline2\n"

    def test_allowed_dirs_blocks_even_after_executor_authorization(self, tmp_path):
        target = tmp_path / "secret.txt"
        target.write_text("secret data", encoding="utf-8")
        tool = FileReadTool(allowed_dirs=[str(tmp_path / "allowed")])

        result = _authorized_call(tool, {"path": str(target)})

        assert result.success is False
        assert "outside allowed directories" in result.content
        assert "secret data" not in result.content

    def test_allowed_dirs_permits_authorized_target(self, tmp_path):
        target = tmp_path / "ok.txt"
        target.write_text("ok data", encoding="utf-8")

        result = _authorized_call(
            FileReadTool(allowed_dirs=[str(tmp_path)]),
            {"path": str(target)},
        )

        assert result.success is True
        assert result.content == "ok data"

    def test_directory_is_not_a_file(self, tmp_path):
        result = _authorized_call(FileReadTool(), {"path": str(tmp_path)})

        assert result.success is False
        assert "Not a file" in result.content

    @pytest.mark.parametrize(
        ("filename", "content"),
        [
            (".env", "SECRET=env-marker"),
            ("server.pem", "pem-marker"),
            ("credentials.json", "credential-marker"),
        ],
    )
    def test_sensitive_files_remain_blocked_after_authorization(
        self,
        tmp_path,
        filename,
        content,
    ):
        target = tmp_path / filename
        target.write_text(content, encoding="utf-8")

        result = _authorized_call(FileReadTool(), {"path": str(target)})

        assert result.success is False
        assert "sensitive" in result.content.lower()
        assert content not in result.content

    def test_oversized_file_is_blocked_after_authorization(self, tmp_path):
        target = tmp_path / "large.txt"
        target.write_bytes(b"x" * 1_048_577)

        result = _authorized_call(FileReadTool(), {"path": str(target)})

        assert result.success is False
        assert "too large" in result.content.lower()

    def test_missing_policy_denies_before_read(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("policy marker", encoding="utf-8")

        result = _call(
            FileReadTool(),
            {"path": str(target)},
            policy=None,
        )

        assert result.success is False
        assert "policy unavailable" in result.content.lower()
        assert "policy marker" not in result.content

    def test_empty_agent_identity_denies_before_read(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("identity marker", encoding="utf-8")
        resource = _canonical(target)

        result = _call(
            FileReadTool(),
            {"path": str(target)},
            policy=_policy(resource),
            agent_id="",
        )

        assert result.success is False
        assert "identity unavailable" in result.content.lower()
        assert "identity marker" not in result.content

    def test_missing_file_read_grant_denies_before_read(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("capability marker", encoding="utf-8")
        resource = _canonical(target)

        result = _call(
            FileReadTool(),
            {"path": str(target)},
            policy=_policy(resource, include_file_read=False),
        )

        assert result.success is False
        assert "file:read" in result.content
        assert "capability marker" not in result.content

    def test_wrong_resource_grant_denies_before_read(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("wrong grant marker", encoding="utf-8")
        resource = _canonical(target)

        result = _call(
            FileReadTool(),
            {"path": str(target)},
            policy=_policy(
                resource,
                file_read_resource=_canonical(tmp_path / "other.txt"),
            ),
        )

        assert result.success is False
        assert "file:read" in result.content
        assert "wrong grant marker" not in result.content

    def test_traversal_spelling_uses_canonical_authorization_resource(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        target = tmp_path / "target.txt"
        target.write_text("canonical marker", encoding="utf-8")
        traversing_path = nested / ".." / "target.txt"
        canonical_target = _canonical(target)

        result = _call(
            FileReadTool(),
            {"path": str(traversing_path)},
            policy=_policy(canonical_target),
        )

        assert result.success is True
        assert result.content == "canonical marker"
        assert result.metadata["path"] == canonical_target

    def test_path_mutation_after_authorization_is_denied(self, tmp_path):
        authorized = tmp_path / "authorized.txt"
        requested_later = tmp_path / "mutated.txt"
        authorized.write_text("authorized marker", encoding="utf-8")
        requested_later.write_text("mutated marker", encoding="utf-8")

        class MutatingPathReadTool(FileReadTool):
            def authorization_resource(self, params):
                resource = super().authorization_resource(params)
                params["path"] = str(requested_later)
                return resource

        result = _call(
            MutatingPathReadTool(),
            {"path": str(authorized)},
            policy=_policy(_canonical(authorized)),
        )

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert "authorized marker" not in result.content
        assert "mutated marker" not in result.content

    def test_receipt_is_not_reused_after_executor_returns(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("one-shot marker", encoding="utf-8")
        tool = FileReadTool()

        authorized = _authorized_call(tool, {"path": str(target)})
        direct = tool.execute(path=str(target))

        assert authorized.success is True
        assert direct.success is False
        assert direct.content == _DIRECT_DENIAL
        assert "one-shot marker" not in direct.content
