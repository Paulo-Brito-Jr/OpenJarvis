"""Tests for the fail-closed ``file_write`` tool boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openjarvis.core.types import ToolCall
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.file_write import FileWriteTool

_AGENT_ID = "file-write-test"
_DIRECT_DENIAL = "Security block: authenticated ToolExecutor dispatch required."


def _canonical(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _call(
    tool: FileWriteTool,
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
            id="file-write-call",
            name="file_write",
            arguments=json.dumps(params),
        )
    )


def _policy(
    resource: str,
    *,
    include_file_write: bool = True,
    file_write_resource: str | None = None,
) -> CapabilityPolicy:
    policy = CapabilityPolicy()
    policy.grant(_AGENT_ID, "tool:invoke", resource)
    if include_file_write:
        policy.grant(
            _AGENT_ID,
            "file:write",
            file_write_resource or resource,
        )
    return policy


def _authorized_call(tool: FileWriteTool, params: dict):
    resource = _canonical(params["path"])
    return _call(tool, params, policy=_policy(resource))


class TestFileWriteTool:
    def test_spec_declares_file_write_capability(self):
        tool = FileWriteTool()

        assert tool.spec.name == "file_write"
        assert tool.spec.category == "filesystem"
        assert "file:write" in tool.spec.required_capabilities

    def test_no_path(self):
        result = FileWriteTool().execute(path="", content="hello")

        assert result.success is False
        assert "No path" in result.content

    @pytest.mark.parametrize("invalid_path", ["   ", ["not", "a", "path"]])
    def test_invalid_path_cannot_fall_back_to_tool_resource(
        self,
        tmp_path,
        monkeypatch,
        invalid_path,
    ):
        monkeypatch.chdir(tmp_path)
        policy = CapabilityPolicy()
        policy.grant(_AGENT_ID, "tool:invoke", "tool:file_write")
        policy.grant(_AGENT_ID, "file:write", "tool:file_write")
        before = set(tmp_path.iterdir())

        result = _call(
            FileWriteTool(),
            {"path": invalid_path, "content": "must not be written"},
            policy=policy,
        )

        assert result.success is False
        assert "authorization metadata invalid" in result.content.lower()
        assert set(tmp_path.iterdir()) == before

    def test_no_content(self):
        result = FileWriteTool().execute(path="/tmp/test.txt")

        assert result.success is False
        assert "No content" in result.content

    def test_invalid_mode(self, tmp_path):
        result = FileWriteTool().execute(
            path=str(tmp_path / "test.txt"),
            content="data",
            mode="invalid",
        )

        assert result.success is False
        assert "Invalid mode" in result.content

    def test_direct_write_is_denied_without_effect(self, tmp_path):
        target = tmp_path / "test.txt"

        result = FileWriteTool().execute(
            path=str(target),
            content="write marker",
            _authorization_receipt="forged",
            agent_id=_AGENT_ID,
        )

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert not target.exists()

    def test_direct_append_is_denied_without_effect(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("original\n", encoding="utf-8")

        result = FileWriteTool().execute(
            path=str(target),
            content="append marker\n",
            mode="append",
        )

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert target.read_text(encoding="utf-8") == "original\n"

    def test_direct_create_dirs_is_denied_before_directory_creation(self, tmp_path):
        parent = tmp_path / "sub" / "deep"
        target = parent / "test.txt"

        result = FileWriteTool().execute(
            path=str(target),
            content="nested marker",
            create_dirs=True,
        )

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert not parent.exists()
        assert not target.exists()

    def test_authorized_write_uses_exact_grant(self, tmp_path):
        target = tmp_path / "test.txt"

        result = _authorized_call(
            FileWriteTool(),
            {"path": str(target), "content": "hello world\n"},
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "hello world\n"
        assert result.metadata["path"] == _canonical(target)
        assert result.metadata["size_bytes"] > 0

    def test_authorized_append_uses_exact_grant(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("line1\n", encoding="utf-8")

        result = _authorized_call(
            FileWriteTool(),
            {
                "path": str(target),
                "content": "line2\n",
                "mode": "append",
            },
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "line1\nline2\n"

    def test_authorized_overwrite_replaces_existing_file(self, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("old content", encoding="utf-8")

        result = _authorized_call(
            FileWriteTool(),
            {"path": str(target), "content": "new content"},
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "new content"

    def test_authorized_create_dirs(self, tmp_path):
        target = tmp_path / "sub" / "deep" / "test.txt"

        result = _authorized_call(
            FileWriteTool(),
            {
                "path": str(target),
                "content": "nested",
                "create_dirs": True,
            },
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "nested"

    def test_missing_parent_without_create_dirs_is_blocked(self, tmp_path):
        target = tmp_path / "nonexistent" / "test.txt"

        result = _authorized_call(
            FileWriteTool(),
            {"path": str(target), "content": "data"},
        )

        assert result.success is False
        assert "Parent directory does not exist" in result.content
        assert not target.parent.exists()

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

        result = _authorized_call(
            FileWriteTool(),
            {"path": str(target), "content": content},
        )

        assert result.success is False
        assert "sensitive" in result.content.lower()
        assert not target.exists()

    def test_allowed_dirs_blocks_even_after_executor_authorization(self, tmp_path):
        target = tmp_path / "blocked.txt"
        tool = FileWriteTool(allowed_dirs=[str(tmp_path / "allowed")])

        result = _authorized_call(
            tool,
            {"path": str(target), "content": "blocked marker"},
        )

        assert result.success is False
        assert "outside allowed directories" in result.content
        assert not target.exists()

    def test_allowed_dirs_permits_authorized_target(self, tmp_path):
        target = tmp_path / "ok.txt"

        result = _authorized_call(
            FileWriteTool(allowed_dirs=[str(tmp_path)]),
            {"path": str(target), "content": "ok data"},
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "ok data"

    def test_file_size_limit_is_enforced_after_authorization(self, tmp_path):
        target = tmp_path / "big.txt"

        class ExpandingWriteTool(FileWriteTool):
            def execute(self, **params):
                original_content = params["content"]
                try:
                    params["content"] = "x" * 10_485_761
                    return super().execute(**params)
                finally:
                    params["content"] = original_content

        result = _authorized_call(
            ExpandingWriteTool(),
            {"path": str(target), "content": "expand-in-tool"},
        )

        assert result.success is False
        assert "too large" in result.content.lower()
        assert not target.exists()

    def test_missing_policy_denies_without_effect(self, tmp_path):
        target = tmp_path / "target.txt"

        result = _call(
            FileWriteTool(),
            {"path": str(target), "content": "policy marker"},
            policy=None,
        )

        assert result.success is False
        assert "policy unavailable" in result.content.lower()
        assert not target.exists()

    def test_empty_agent_identity_denies_without_effect(self, tmp_path):
        target = tmp_path / "target.txt"
        resource = _canonical(target)

        result = _call(
            FileWriteTool(),
            {"path": str(target), "content": "identity marker"},
            policy=_policy(resource),
            agent_id="",
        )

        assert result.success is False
        assert "identity unavailable" in result.content.lower()
        assert not target.exists()

    def test_missing_file_write_grant_denies_without_effect(self, tmp_path):
        target = tmp_path / "target.txt"
        resource = _canonical(target)

        result = _call(
            FileWriteTool(),
            {"path": str(target), "content": "capability marker"},
            policy=_policy(resource, include_file_write=False),
        )

        assert result.success is False
        assert "file:write" in result.content
        assert not target.exists()

    def test_wrong_resource_grant_denies_without_effect(self, tmp_path):
        target = tmp_path / "target.txt"
        resource = _canonical(target)

        result = _call(
            FileWriteTool(),
            {"path": str(target), "content": "wrong grant marker"},
            policy=_policy(
                resource,
                file_write_resource=_canonical(tmp_path / "other.txt"),
            ),
        )

        assert result.success is False
        assert "file:write" in result.content
        assert not target.exists()

    def test_traversal_spelling_uses_canonical_authorization_resource(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        target = tmp_path / "target.txt"
        traversing_path = nested / ".." / "target.txt"
        canonical_target = _canonical(target)

        result = _call(
            FileWriteTool(),
            {"path": str(traversing_path), "content": "canonical marker"},
            policy=_policy(canonical_target),
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "canonical marker"
        assert result.metadata["path"] == canonical_target

    def test_path_mutation_after_authorization_is_denied_without_effect(
        self,
        tmp_path,
    ):
        authorized = tmp_path / "authorized.txt"
        mutated = tmp_path / "mutated.txt"

        class MutatingPathWriteTool(FileWriteTool):
            def authorization_resource(self, params):
                resource = super().authorization_resource(params)
                params["path"] = str(mutated)
                return resource

        result = _call(
            MutatingPathWriteTool(),
            {"path": str(authorized), "content": "mutation marker"},
            policy=_policy(_canonical(authorized)),
        )

        assert result.success is False
        assert result.content == _DIRECT_DENIAL
        assert not authorized.exists()
        assert not mutated.exists()

    def test_receipt_is_not_reused_after_executor_returns(self, tmp_path):
        target = tmp_path / "target.txt"
        tool = FileWriteTool()

        authorized = _authorized_call(
            tool,
            {"path": str(target), "content": "authorized"},
        )
        direct = tool.execute(path=str(target), content="direct")

        assert authorized.success is True
        assert direct.success is False
        assert direct.content == _DIRECT_DENIAL
        assert target.read_text(encoding="utf-8") == "authorized"
