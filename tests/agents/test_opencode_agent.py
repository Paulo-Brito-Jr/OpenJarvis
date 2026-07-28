"""Tests for OpenCodeAgent (wraps the `opencode` coding agent).

The pure helpers, provider-config wiring, graceful degradation, and response
parsing are tested without the `opencode` binary. SPIKE_RESPONSE is the actual
message shape captured from a live `opencode serve` session.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from openjarvis.agents.opencode import (
    OpenCodeAgent,
    _derive_openai_base_url,
    _extract_text,
    _extract_tool_results,
    is_opencode_available,
)

SPIKE_RESPONSE = {
    "info": {
        "role": "assistant",
        "agent": "build",
        "modelID": "local-model",
        "providerID": "openjarvis",
        "finish": "stop",
        "tokens": {"input": 0, "output": 0},
        "sessionID": "ses_x",
        "id": "msg_x",
    },
    "parts": [
        {"type": "step-start"},
        {"type": "text", "text": "Hello from the local model. "},
        {"type": "step-finish", "reason": "stop"},
    ],
}


class TestPartParsing:
    def test_extract_text_joins_text_parts(self):
        assert _extract_text(SPIKE_RESPONSE["parts"]) == "Hello from the local model."

    def test_extract_text_ignores_non_text(self):
        assert _extract_text([{"type": "step-start"}, {"type": "tool"}]) == ""

    def test_extract_tool_results_success(self):
        parts = [
            {
                "type": "tool",
                "tool": "bash",
                "state": {"status": "completed", "output": "ok"},
            }
        ]
        tr = _extract_tool_results(parts)
        assert len(tr) == 1
        assert tr[0].tool_name == "bash"
        assert tr[0].content == "ok"
        assert tr[0].success is True

    def test_extract_tool_results_error(self):
        parts = [
            {
                "type": "tool",
                "tool": "edit",
                "state": {"status": "error", "output": "boom"},
            }
        ]
        assert _extract_tool_results(parts)[0].success is False


class TestDeriveBaseUrl:
    def test_from_host_appends_v1(self):
        eng = SimpleNamespace(_host="http://localhost:11434")
        assert _derive_openai_base_url(eng) == "http://localhost:11434/v1"

    def test_host_already_v1(self):
        eng = SimpleNamespace(_host="http://x:8000/v1")
        assert _derive_openai_base_url(eng) == "http://x:8000/v1"

    def test_explicit_base_url_attr(self):
        eng = SimpleNamespace(base_url="http://y/v1")
        assert _derive_openai_base_url(eng) == "http://y/v1"

    def test_unwraps_wrapper_engine(self):
        # InstrumentedEngine wraps the real engine at `_inner`; must unwrap.
        wrapped = SimpleNamespace(
            _inner=SimpleNamespace(_host="http://localhost:11434")
        )
        assert _derive_openai_base_url(wrapped) == "http://localhost:11434/v1"

    def test_none_when_unknown(self):
        assert _derive_openai_base_url(SimpleNamespace()) == ""


class TestAvailability:
    def test_true(self, monkeypatch):
        monkeypatch.setattr(
            "openjarvis.agents.opencode.shutil.which", lambda n: "/usr/bin/opencode"
        )
        assert is_opencode_available() is True

    def test_false(self, monkeypatch):
        monkeypatch.setattr("openjarvis.agents.opencode.shutil.which", lambda n: None)
        assert is_opencode_available() is False


class TestConfigBuilding:
    def test_includes_provider_when_base_url(self, tmp_path):
        cfg = OpenCodeAgent(
            SimpleNamespace(_host="http://localhost:11434"),
            "qwen3:8b",
            workspace=str(tmp_path),
        )._build_config()
        prov = cfg["provider"]["openjarvis"]
        assert prov["npm"] == "@ai-sdk/openai-compatible"
        assert prov["options"]["baseURL"] == "http://localhost:11434/v1"
        assert "qwen3:8b" in prov["models"]

    def test_no_provider_when_no_base_url(self, tmp_path):
        # Pass-through model -> rely on opencode's own provider; no provider block.
        cfg = OpenCodeAgent(
            SimpleNamespace(), "ollama/llama3", workspace=str(tmp_path)
        )._build_config()
        assert "provider" not in cfg

    def test_build_mode_permission_denies_all_effects(self, tmp_path):
        cfg = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "m",
            workspace=str(tmp_path),
            agent="build",
        )._build_config()
        assert cfg["permission"] == {
            "edit": "deny",
            "bash": "deny",
            "webfetch": "deny",
        }

    def test_plan_mode_permission_denies_edit_and_bash(self, tmp_path):
        cfg = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "m",
            workspace=str(tmp_path),
            agent="plan",
        )._build_config()
        assert cfg["permission"]["edit"] == "deny"
        assert cfg["permission"]["bash"] == "deny"
        assert cfg["permission"]["webfetch"] == "deny"

    def test_custom_permission_cannot_override_fail_closed_policy(self, tmp_path):
        cfg = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "m",
            workspace=str(tmp_path),
            permission={
                "bash": "allow",
                "edit": "allow",
                "webfetch": "allow",
            },
        )._build_config()
        assert cfg["permission"] == {
            "edit": "deny",
            "bash": "deny",
            "webfetch": "deny",
        }

    def test_does_not_pollute_workspace(self, tmp_path):
        # The config goes to a private OPENCODE_CONFIG file, never the workspace.
        OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"), "m", workspace=str(tmp_path)
        )._build_config()
        assert not (tmp_path / "opencode.json").exists()


class TestRunFailClosed:
    def test_constructor_does_not_retain_disabled_adapter_credentials(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "environment-secret")
        agent = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "m",
            workspace=str(tmp_path),
            api_key="provider-secret",
            server_password="server-secret",
        )

        assert agent._api_key == ""
        assert agent._server_password == ""

    def test_missing_binary_still_reports_security_disabled(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(
            "openjarvis.agents.opencode.shutil.which",
            lambda n: None,
        )
        agent = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "m",
            workspace=str(tmp_path),
            opencode_bin="/nonexistent/opencode",
        )
        res = agent.run("do something")
        assert res.metadata.get("error") is True
        assert res.metadata["security_disabled"] is True
        assert res.metadata["reason"] == "unverified_external_sandbox"

    def test_unresolvable_provider_is_not_reached(self, tmp_path):
        agent = OpenCodeAgent(SimpleNamespace(), "qwen3:8b", workspace=str(tmp_path))
        res = agent.run("do something")
        assert res.metadata["security_disabled"] is True
        assert "disabled" in res.content.lower()

    def test_declares_security_context_requirement(self):
        assert OpenCodeAgent.requires_security_context is True

    def test_ensure_server_refuses_before_spawn(self, tmp_path):
        agent = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "m",
            workspace=str(tmp_path),
        )

        try:
            agent._ensure_server()
        except RuntimeError as exc:
            assert "disabled" in str(exc).lower()
        else:
            raise AssertionError("disabled adapter unexpectedly started")


class _FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


# Full-turn message list as returned by GET /session/{id}/message: the tool
# executes in an intermediate assistant message, not the final one (this shape
# was captured from a live opencode session).
TURN_MESSAGES = [
    {"info": {"role": "user"}, "parts": [{"type": "text", "text": "..."}]},
    {
        "info": {"role": "assistant"},
        "parts": [
            {"type": "step-start"},
            {
                "type": "tool",
                "tool": "write",
                "callID": "c1",
                "state": {
                    "status": "completed",
                    "output": "Wrote file successfully.",
                    "input": {"filePath": "greet.py", "content": "x"},
                },
            },
            {"type": "step-finish", "reason": "tool"},
        ],
    },
    SPIKE_RESPONSE,  # final assistant message (text only)
]


class _FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, path, json=None):
        if path == "/session":
            return _FakeResp({"id": "ses_x"})
        if path.endswith("/message"):
            _FakeClient.last_body = json
            return _FakeResp(SPIKE_RESPONSE)
        return _FakeResp({})

    def get(self, path, **kw):
        return _FakeResp(TURN_MESSAGES)


class TestDisabledRunDoesNotReachLegacyClient:
    def test_run_never_starts_server_or_client(self, monkeypatch, tmp_path):
        agent = OpenCodeAgent(
            SimpleNamespace(_host="http://h:1"),
            "local-model",
            workspace=str(tmp_path),
            agent="build",
        )
        ensure_server = MagicMock()
        client = MagicMock()
        monkeypatch.setattr(agent, "_ensure_server", ensure_server)
        monkeypatch.setattr(agent, "_client", client)

        res = agent.run("Write a hello world")

        assert res.metadata["security_disabled"] is True
        assert res.tool_results == []
        ensure_server.assert_not_called()
        client.assert_not_called()
