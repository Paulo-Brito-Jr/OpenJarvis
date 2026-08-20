"""Tests for the Docker-sandboxed code interpreter tool."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

PINNED_IMAGE = "python@sha256:" + ("a" * 64)


def _make_docker_mock():
    """Create a mock docker module and return (mock_module, mock_client)."""
    mock_docker = MagicMock()
    mock_client = MagicMock()
    mock_docker.from_env.return_value = mock_client
    return mock_docker, mock_client


class TestDockerCodeInterpreterTool:
    def test_spec(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        tool = DockerCodeInterpreterTool()
        spec = tool.spec
        assert spec.name == "code_interpreter_docker"
        assert "code" in spec.parameters["properties"]
        assert spec.category == "code"
        assert spec.requires_confirmation is True
        assert spec.required_capabilities == ["code:execute"]
        assert tool.authorization_resource({}) == "container:python:3.12-slim"

    def test_empty_code(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        tool = DockerCodeInterpreterTool()
        result = tool.execute(code="")
        assert not result.success
        assert "No code" in result.content

    def test_successful_execution(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, mock_client = _make_docker_mock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [
            b"Hello World\n",  # stdout
            b"",  # stderr
        ]
        mock_client.containers.run.return_value = mock_container

        with patch.dict(sys.modules, {"docker": mock_docker}):
            tool = DockerCodeInterpreterTool(image=PINNED_IMAGE)
            result = tool.execute(code="print('Hello World')")

        assert result.success
        assert "Hello World" in result.content
        mock_container.remove.assert_called_once_with(force=True)

    def test_execution_error(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, mock_client = _make_docker_mock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.side_effect = [
            b"",
            b"NameError: name 'foo' is not defined\n",
        ]
        mock_client.containers.run.return_value = mock_container

        with patch.dict(sys.modules, {"docker": mock_docker}):
            tool = DockerCodeInterpreterTool(image=PINNED_IMAGE)
            result = tool.execute(code="print(foo)")

        assert not result.success
        assert "NameError" in result.content

    def test_container_resource_limits(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, mock_client = _make_docker_mock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"ok\n", b""]
        mock_client.containers.run.return_value = mock_container

        tool = DockerCodeInterpreterTool(
            image=PINNED_IMAGE,
            memory_limit="256m",
            cpu_count=2,
            network_disabled=True,
            pids_limit=50,
        )

        with patch.dict(sys.modules, {"docker": mock_docker}):
            tool.execute(code="print('ok')")

        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs[1]["mem_limit"] == "256m"
        assert call_kwargs[1]["nano_cpus"] == 2 * 10**9
        assert call_kwargs[1]["network_disabled"] is True
        assert call_kwargs[1]["pids_limit"] == 50
        assert call_kwargs[1]["read_only"] is True
        assert call_kwargs[1]["cap_drop"] == ["ALL"]
        assert call_kwargs[1]["security_opt"] == ["no-new-privileges:true"]
        assert call_kwargs[1]["user"] == "65534:65534"
        assert call_kwargs[1]["privileged"] is False
        assert call_kwargs[1]["ipc_mode"] == "none"
        assert call_kwargs[1]["tmpfs"]["/tmp"].startswith("rw,noexec,nosuid,nodev,")
        assert call_kwargs[0][1][:4] == ["python", "-I", "-S", "-B"]

    def test_output_truncation(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, mock_client = _make_docker_mock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"x" * 20000, b""]
        mock_client.containers.run.return_value = mock_container

        with patch.dict(sys.modules, {"docker": mock_docker}):
            tool = DockerCodeInterpreterTool(
                image=PINNED_IMAGE,
                max_output=100,
            )
            result = tool.execute(code="print('x' * 20000)")

        assert result.success
        assert len(result.content) < 200
        assert "truncated" in result.content

    def test_container_cleanup_on_error(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, mock_client = _make_docker_mock()

        mock_container = MagicMock()
        mock_container.wait.side_effect = Exception("timeout")
        mock_client.containers.run.return_value = mock_container

        with patch.dict(sys.modules, {"docker": mock_docker}):
            tool = DockerCodeInterpreterTool(image=PINNED_IMAGE)
            result = tool.execute(code="import time; time.sleep(999)")

        assert not result.success
        mock_container.remove.assert_called_once_with(force=True)

    def test_mutable_image_tag_is_rejected_before_docker_access(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, _ = _make_docker_mock()
        with patch.dict(sys.modules, {"docker": mock_docker}):
            result = DockerCodeInterpreterTool(image="python:3.12-slim").execute(
                code="print('must not run')"
            )

        assert result.success is False
        assert result.metadata["reason"] == "immutable_image_digest_required"
        mock_docker.from_env.assert_not_called()

    def test_network_enabled_sandbox_is_rejected_before_docker_access(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        mock_docker, _ = _make_docker_mock()
        with patch.dict(sys.modules, {"docker": mock_docker}):
            result = DockerCodeInterpreterTool(
                image=PINNED_IMAGE,
                network_disabled=False,
            ).execute(code="print('must not run')")

        assert result.success is False
        assert result.metadata["reason"] == "network_isolation_required"
        mock_docker.from_env.assert_not_called()
