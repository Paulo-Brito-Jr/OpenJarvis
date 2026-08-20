from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tomllib

from openjarvis.core.types import ToolCall
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


def test_skill_create(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    result = tool.execute(
        action="create",
        name="api_health",
        description="Check API health",
        steps=[
            {
                "tool_name": "http_request",
                "arguments_template": '{"url": "{endpoint}/health"}',
            }
        ],
    )
    assert result.success
    assert (skills_dir / "api_health.toml").exists()


def test_skill_list(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    tool.execute(
        action="create",
        name="skill_a",
        description="Skill A",
        steps=[{"tool_name": "calculator"}],
    )
    tool.execute(
        action="create",
        name="skill_b",
        description="Skill B",
        steps=[{"tool_name": "calculator"}],
    )
    result = tool.execute(action="list")
    assert "skill_a" in result.content
    assert "skill_b" in result.content


def test_skill_delete(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    tool.execute(
        action="create",
        name="temp_skill",
        description="Temp",
        steps=[{"tool_name": "calculator"}],
    )
    assert (skills_dir / "temp_skill.toml").exists()
    result = tool.execute(action="delete", name="temp_skill")
    assert result.success
    assert not (skills_dir / "temp_skill.toml").exists()


def test_skill_load(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    tool.execute(
        action="create",
        name="my_skill",
        description="My skill desc",
        steps=[
            {
                "tool_name": "web_search",
                "arguments_template": '{"q": "test"}',
            }
        ],
    )
    result = tool.execute(action="load", name="my_skill")
    assert "web_search" in result.content
    assert "My skill desc" in result.content


@pytest.mark.parametrize(
    "action",
    ["create", "load", "delete"],
)
@pytest.mark.parametrize(
    "name",
    ["../escape", "..", "nested/escape", "/tmp/escape", "bad.toml"],
)
def test_skill_name_traversal_is_rejected(
    skills_dir: Path,
    action: str,
    name: str,
):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    result = tool.execute(
        action=action,
        name=name,
        description="No escape",
        steps=[{"tool_name": "calculator"}],
    )

    assert result.success is False
    assert "Invalid skill request" in result.content
    assert list(skills_dir.iterdir()) == []


def test_skill_create_serializes_toml_strings_without_injection(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    description = 'quoted"\n[[skill.steps]]\ntool_name = "injected"'
    arguments = '{"value": "\\"\\n[[skill.steps]]\\ntool_name = \\"evil\\""}'
    tool = SkillManageTool(skills_dir=skills_dir)

    result = tool.execute(
        action="create",
        name="safe_skill",
        description=description,
        steps=[
            {
                "tool_name": "calculator",
                "arguments_template": arguments,
            }
        ],
    )

    assert result.success is True
    with (skills_dir / "safe_skill.toml").open("rb") as handle:
        parsed = tomllib.load(handle)
    assert parsed["skill"]["description"] == description
    assert parsed["skill"]["steps"] == [
        {
            "tool_name": "calculator",
            "arguments_template": arguments,
        }
    ]


def test_skill_load_rejects_symlink_escape(skills_dir: Path, tmp_path: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    outside = tmp_path / "outside.toml"
    outside.write_text("secret", encoding="utf-8")
    (skills_dir / "linked.toml").symlink_to(outside)
    tool = SkillManageTool(skills_dir=skills_dir)

    result = tool.execute(action="load", name="linked")

    assert result.success is False
    assert "escapes" in result.content
    assert outside.read_text(encoding="utf-8") == "secret"


def test_skill_create_requires_write_capability_before_confirmation(
    skills_dir: Path,
):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    policy = CapabilityPolicy()
    resource = str((skills_dir / "created.toml").resolve())
    policy.grant("skill-agent", "tool:invoke", resource)
    executor = ToolExecutor(
        [tool],
        capability_policy=policy,
        agent_id="skill-agent",
        interactive=True,
        confirm_callback=MagicMock(return_value=True),
    )

    result = executor.execute(
        ToolCall(
            id="1",
            name="skill_manage",
            arguments=(
                '{"action":"create","name":"created","description":"x",'
                '"steps":[{"tool_name":"calculator"}]}'
            ),
        )
    )

    assert result.success is False
    assert "file:write" in result.content
    assert not (skills_dir / "created.toml").exists()


def test_skill_create_requires_live_tty_even_with_write_capability(
    skills_dir: Path,
    monkeypatch,
):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    policy = CapabilityPolicy()
    resource = str((skills_dir / "created.toml").resolve())
    policy.grant("skill-agent", "tool:invoke", resource)
    policy.grant("skill-agent", "file:write", resource)
    confirmation = MagicMock(return_value=True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    executor = ToolExecutor(
        [tool],
        capability_policy=policy,
        agent_id="skill-agent",
        interactive=True,
        confirm_callback=confirmation,
    )

    result = executor.execute(
        ToolCall(
            id="1",
            name="skill_manage",
            arguments=(
                '{"action":"create","name":"created","description":"x",'
                '"steps":[{"tool_name":"calculator"}]}'
            ),
        )
    )

    assert result.success is False
    assert "live TTY" in result.content
    confirmation.assert_not_called()
    assert not (skills_dir / "created.toml").exists()
