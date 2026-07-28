"""Docker-sandboxed code interpreter tool."""

from __future__ import annotations

import re
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_PINNED_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


@ToolRegistry.register("code_interpreter_docker")
class DockerCodeInterpreterTool(BaseTool):
    """Execute Python code in a disposable Docker container."""

    tool_id = "code_interpreter_docker"

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        timeout: int = 30,
        max_output: int = 10000,
        memory_limit: str = "512m",
        cpu_count: int = 1,
        network_disabled: bool = True,
        pids_limit: int = 100,
    ) -> None:
        self._image = image
        self._timeout = timeout
        self._max_output = max_output
        self._memory_limit = memory_limit
        self._cpu_count = cpu_count
        self._network_disabled = network_disabled
        self._pids_limit = pids_limit

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter_docker",
            description=(
                "Execute Python code in an isolated Docker container. "
                "Provides sandboxed execution with resource limits "
                "(512MB memory, 1 CPU, no network, PID limit 100)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
            },
            category="code",
            requires_confirmation=True,
            timeout_seconds=60.0,
            required_capabilities=["code:execute"],
            metadata={"sandbox_required": True, "sandbox": "docker"},
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        """Authorize the sandbox class without exposing source text."""
        del params
        return f"container:{self._image}"

    def execute(self, **params: Any) -> ToolResult:
        code = params.get("code", "")
        if not code:
            return ToolResult(
                tool_name="code_interpreter_docker",
                content="No code provided.",
                success=False,
            )
        if not _PINNED_IMAGE_RE.fullmatch(self._image):
            return ToolResult(
                tool_name="code_interpreter_docker",
                content=(
                    "Docker execution disabled: image must be pinned to an "
                    "immutable sha256 digest."
                ),
                success=False,
                metadata={
                    "security_disabled": True,
                    "reason": "immutable_image_digest_required",
                },
            )
        if not self._network_disabled:
            return ToolResult(
                tool_name="code_interpreter_docker",
                content=(
                    "Docker execution disabled: arbitrary-code sandboxes "
                    "must not have network access."
                ),
                success=False,
                metadata={
                    "security_disabled": True,
                    "reason": "network_isolation_required",
                },
            )

        try:
            import docker
        except ImportError:
            return ToolResult(
                tool_name="code_interpreter_docker",
                content=(
                    "Docker SDK not available. "
                    "Install with: uv sync --extra sandbox-docker"
                ),
                success=False,
            )

        try:
            client = docker.from_env()

            container = client.containers.run(
                self._image,
                ["python", "-I", "-S", "-B", "-c", code],
                detach=True,
                mem_limit=self._memory_limit,
                nano_cpus=self._cpu_count * 10**9,
                network_disabled=True,
                pids_limit=self._pids_limit,
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                user="65534:65534",
                privileged=False,
                init=True,
                ipc_mode="none",
                # Writable scratch space, but never executable or device-backed.
                tmpfs={
                    "/tmp": "rw,noexec,nosuid,nodev,size=64m,mode=1777"
                },
                stderr=True,
                stdout=True,
            )

            try:
                result = container.wait(timeout=self._timeout)
                exit_code = result.get("StatusCode", -1)
                stdout = container.logs(
                    stdout=True,
                    stderr=False,
                ).decode("utf-8", errors="replace")
                stderr = container.logs(
                    stdout=False,
                    stderr=True,
                ).decode("utf-8", errors="replace")
            finally:
                container.remove(force=True)

            output = stdout
            if stderr:
                output += ("\n" if output else "") + stderr
            if len(output) > self._max_output:
                output = output[: self._max_output] + "\n... (output truncated)"

            return ToolResult(
                tool_name="code_interpreter_docker",
                content=output or "(no output)",
                success=exit_code == 0,
                metadata={"exit_code": exit_code},
            )

        except Exception as exc:
            error_type = type(exc).__name__
            if "timeout" in str(exc).lower() or "read timed out" in str(exc).lower():
                return ToolResult(
                    tool_name="code_interpreter_docker",
                    content=(f"Execution timed out after {self._timeout} seconds."),
                    success=False,
                )
            return ToolResult(
                tool_name="code_interpreter_docker",
                content=f"Docker execution error ({error_type}): {exc}",
                success=False,
            )


__all__ = ["DockerCodeInterpreterTool"]
