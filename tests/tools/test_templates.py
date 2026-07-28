"""Tests for MCP templates (Phase 16.3)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from openjarvis.core.types import ToolCall
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.templates.loader import ToolTemplate, discover_templates


class TestToolTemplate:
    def test_create_template(self):
        template = ToolTemplate(
            {
                "name": "test_tool",
                "description": "A test tool",
                "action": {"type": "transform", "transform": "upper"},
            }
        )
        assert template.spec.name == "test_tool"

    def test_execute_upper_transform(self):
        template = ToolTemplate(
            {
                "name": "upper",
                "description": "Uppercase",
                "action": {"type": "transform", "transform": "upper"},
            }
        )
        result = template.execute(input="hello")
        assert result.success
        assert result.content == "HELLO"

    def test_execute_lower_transform(self):
        template = ToolTemplate(
            {
                "name": "lower",
                "description": "Lowercase",
                "action": {"type": "transform", "transform": "lower"},
            }
        )
        result = template.execute(input="HELLO")
        assert result.success
        assert result.content == "hello"

    def test_execute_reverse_transform(self):
        template = ToolTemplate(
            {
                "name": "reverse",
                "description": "Reverse",
                "action": {"type": "transform", "transform": "reverse"},
            }
        )
        result = template.execute(input="hello")
        assert result.success
        assert result.content == "olleh"

    def test_execute_length_transform(self):
        template = ToolTemplate(
            {
                "name": "length",
                "description": "Length",
                "action": {"type": "transform", "transform": "length"},
            }
        )
        result = template.execute(input="hello")
        assert result.success
        assert result.content == "5"

    def test_execute_json_pretty_transform(self):
        template = ToolTemplate(
            {
                "name": "json",
                "description": "JSON pretty",
                "action": {"type": "transform", "transform": "json_pretty"},
            }
        )
        result = template.execute(input='{"a":1}')
        assert result.success
        assert '"a": 1' in result.content

    def test_execute_json_pretty_invalid(self):
        template = ToolTemplate(
            {
                "name": "json",
                "description": "JSON pretty",
                "action": {"type": "transform", "transform": "json_pretty"},
            }
        )
        result = template.execute(input="not json")
        assert not result.success

    def test_execute_python_action(self):
        template = ToolTemplate(
            {
                "name": "py",
                "description": "Python",
                "action": {"type": "python", "expression": "str(len(input))"},
            }
        )
        result = template.execute(input="hello")
        assert result.success
        assert result.content == "5"

    def test_execute_identity_transform(self):
        template = ToolTemplate(
            {
                "name": "identity",
                "description": "Identity",
                "action": {"type": "transform", "transform": "identity"},
            }
        )
        result = template.execute(input="unchanged")
        assert result.success
        assert result.content == "unchanged"

    def test_unknown_action_type(self):
        template = ToolTemplate(
            {
                "name": "bad",
                "description": "Bad",
                "action": {"type": "unknown"},
            }
        )
        result = template.execute()
        assert not result.success

    def test_template_metadata(self):
        template = ToolTemplate(
            {
                "name": "test",
                "description": "test",
                "action": {"type": "transform"},
            }
        )
        assert template.spec.metadata.get("template") is True

    def test_every_template_requires_code_execute_capability(self):
        for action in (
            {"type": "transform", "transform": "upper"},
            {"type": "python", "expression": "str(input)"},
            {"type": "shell", "command": "echo {input}"},
        ):
            template = ToolTemplate(
                {
                    "name": "guarded",
                    "description": "Guarded template",
                    "action": action,
                }
            )
            assert template.spec.required_capabilities == ["code:execute"]

    def test_shell_template_denied_before_subprocess_without_capability(
        self,
        monkeypatch,
    ):
        template = ToolTemplate(
            {
                "name": "shell_guard",
                "description": "Guarded shell",
                "action": {"type": "shell", "command": "echo {input}"},
            }
        )
        run = MagicMock()
        monkeypatch.setattr(
            "openjarvis.tools.templates.loader.subprocess.run",
            run,
        )
        policy = CapabilityPolicy()
        policy.grant("template-agent", "tool:invoke", "echo {input}")
        executor = ToolExecutor(
            [template],
            capability_policy=policy,
            agent_id="template-agent",
        )

        result = executor.execute(
            ToolCall(
                id="1",
                name="shell_guard",
                arguments='{"input":"hello"}',
            )
        )

        assert result.success is False
        assert "code:execute" in result.content
        run.assert_not_called()

    def test_shell_template_requires_live_confirmation_with_capability(
        self,
        monkeypatch,
    ):
        template = ToolTemplate(
            {
                "name": "shell_guard",
                "description": "Guarded shell",
                "action": {"type": "shell", "command": "echo {input}"},
            }
        )
        run = MagicMock()
        confirmation = MagicMock(return_value=True)
        monkeypatch.setattr(
            "openjarvis.tools.templates.loader.subprocess.run",
            run,
        )
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        policy = CapabilityPolicy()
        policy.grant("template-agent", "tool:invoke", "echo {input}")
        policy.grant("template-agent", "code:execute", "echo {input}")
        executor = ToolExecutor(
            [template],
            capability_policy=policy,
            agent_id="template-agent",
            interactive=True,
            confirm_callback=confirmation,
        )

        result = executor.execute(
            ToolCall(
                id="1",
                name="shell_guard",
                arguments='{"input":"hello"}',
            )
        )

        assert result.success is False
        assert "live TTY" in result.content
        confirmation.assert_not_called()
        run.assert_not_called()

    def test_transform_template_runs_with_exact_explicit_grants(self):
        template = ToolTemplate(
            {
                "name": "upper_guard",
                "description": "Guarded transform",
                "action": {"type": "transform", "transform": "upper"},
            }
        )
        resource = "template:upper_guard:transform"
        policy = CapabilityPolicy()
        policy.grant("template-agent", "tool:invoke", resource)
        policy.grant("template-agent", "code:execute", resource)
        executor = ToolExecutor(
            [template],
            capability_policy=policy,
            agent_id="template-agent",
        )

        result = executor.execute(
            ToolCall(
                id="1",
                name="upper_guard",
                arguments='{"input":"hello"}',
            )
        )

        assert result.success is True
        assert result.content == "HELLO"


class TestDiscoverTemplates:
    def test_discover_builtin(self):
        templates = discover_templates()
        # Should find at least some builtin templates
        if templates:
            names = {t.spec.name for t in templates}
            assert len(names) > 0

    def test_discover_nonexistent_dir(self, tmp_path):
        templates = discover_templates(tmp_path / "nonexistent")
        assert templates == []

    def test_discover_custom_dir(self, tmp_path):
        # Create a custom template
        toml_content = """
[tool]
name = "custom"
description = "Custom tool"
[tool.action]
type = "transform"
transform = "upper"
"""
        (tmp_path / "custom.toml").write_text(toml_content)
        templates = discover_templates(tmp_path)
        assert len(templates) == 1
        assert templates[0].spec.name == "custom"
