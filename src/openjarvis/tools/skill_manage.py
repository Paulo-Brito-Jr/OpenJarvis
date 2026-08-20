"""SkillManageTool — create, list, load, or delete agent-authored skills."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("skill_manage")
class SkillManageTool(BaseTool):
    """Manage agent-authored procedural skills."""

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        if skills_dir is None:
            skills_dir = get_config_dir() / "skills"
        self._skills_dir = Path(skills_dir).expanduser()

    _SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    _SAFE_TOOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="skill_manage",
            description="Create, list, load, or delete agent-authored skills.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "load", "delete"],
                        "description": "Action to perform.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name (for create/load/delete).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Skill description (for create).",
                    },
                    "steps": {
                        "type": "array",
                        "description": (
                            "List of step dicts with tool_name and optional"
                            " arguments_template (for create)."
                        ),
                    },
                },
                "required": ["action"],
            },
            category="skill",
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        name = params.get("name")
        if isinstance(name, str) and self._SAFE_NAME_RE.fullmatch(name):
            return str(self._skill_path(name))
        return str(self._skills_dir.resolve(strict=False))

    def authorization_capabilities(self, params: dict[str, Any]) -> List[str]:
        action = params.get("action", "list")
        if action in {"create", "delete"}:
            return ["file:write"]
        return ["file:read"]

    def requires_confirmation_for(self, params: dict[str, Any]) -> bool:
        return params.get("action") in {"create", "delete"}

    def execute(self, **params: Any) -> ToolResult:
        action = params.get("action", "list")
        name = params.get("name", "")
        try:
            if action == "create":
                return self._create(
                    name, params.get("description", ""), params.get("steps", [])
                )
            elif action == "list":
                return self._list()
            elif action == "load":
                return self._load(name)
            elif action == "delete":
                return self._delete(name)
        except (TypeError, ValueError) as exc:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Invalid skill request: {exc}",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=False,
            content=f"Unknown action: {action}",
        )

    def _create(self, name: str, description: str, steps: List[dict]) -> ToolResult:
        path = self._skill_path(name)
        if not isinstance(description, str) or len(description) > 4096:
            raise ValueError("description must be a string of at most 4096 characters")
        if not isinstance(steps, list) or len(steps) > 100:
            raise ValueError("steps must be a list containing at most 100 entries")

        self._skills_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "[skill]",
            f"name = {json.dumps(name, ensure_ascii=False)}",
            f"description = {json.dumps(description, ensure_ascii=False)}",
            "",
        ]
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("each step must be an object")
            tool_name = step.get("tool_name", "")
            if not isinstance(tool_name, str) or not self._SAFE_TOOL_RE.fullmatch(
                tool_name
            ):
                raise ValueError("step tool_name is invalid")
            lines.append("[[skill.steps]]")
            lines.append(f"tool_name = {json.dumps(tool_name, ensure_ascii=False)}")
            if "arguments_template" in step:
                arguments_template = step["arguments_template"]
                if not isinstance(arguments_template, str):
                    raise ValueError("arguments_template must be a string")
                lines.append(
                    "arguments_template = "
                    f"{json.dumps(arguments_template, ensure_ascii=False)}"
                )
            if "output_key" in step:
                output_key = step["output_key"]
                if not isinstance(output_key, str):
                    raise ValueError("output_key must be a string")
                lines.append(
                    f"output_key = {json.dumps(output_key, ensure_ascii=False)}"
                )
            lines.append("")

        serialized = "\n".join(lines)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=str(self._skills_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=f"Created skill: {name}",
        )

    def _list(self) -> ToolResult:
        if not self._skills_dir.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                content="No skills directory found.",
            )
        skills = []
        for f in sorted(self._skills_dir.glob("*.toml")):
            skills.append(f.stem)
        if not skills:
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                content="No skills found.",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content="Available skills:\n" + "\n".join(f"- {s}" for s in skills),
        )

    def _load(self, name: str) -> ToolResult:
        path = self._skill_path(name)
        if not path.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Skill not found: {name}",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=path.read_text(encoding="utf-8"),
        )

    def _delete(self, name: str) -> ToolResult:
        path = self._skill_path(name)
        if not path.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Skill not found: {name}",
            )
        path.unlink()
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=f"Deleted skill: {name}",
        )

    def _skill_path(self, name: str) -> Path:
        """Resolve a canonical child path or reject traversal/aliases."""
        if not isinstance(name, str) or not self._SAFE_NAME_RE.fullmatch(name):
            raise ValueError(
                "name must contain only letters, digits, underscores, or hyphens"
            )
        root = self._skills_dir.resolve(strict=False)
        candidate = (root / f"{name}.toml").resolve(strict=False)
        if candidate.parent != root:
            raise ValueError("skill path escapes the configured skills directory")
        return candidate
