"""Tests for extended API routes."""

from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server.api_routes import include_all_routes  # noqa: E402


def _make_app(capability_policy=None, *, authenticated=False):
    app = FastAPI()
    app.state.capability_policy = capability_policy
    app.state.bus = None
    if authenticated:

        @app.middleware("http")
        async def _authenticated_request(request, call_next):
            request.state.api_principal = "server-api"
            return await call_next(request)

    include_all_routes(app)
    return app


class TestAgentRoutes:
    def test_list_agents(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "registered" in data
        assert "running" in data

    def test_list_agents_projects_spawned_private_fields(self):
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        _SPAWNED_AGENTS["private-agent"] = {
            "agent_type": "simple",
            "status": "running",
            "initial_query": "private household request",
            "tools": "shell_exec,file_read",
        }
        try:
            response = TestClient(_make_app()).get("/v1/agents")
            running = next(
                item
                for item in response.json()["running"]
                if item["id"] == "private-agent"
            )
            assert running == {
                "id": "private-agent",
                "status": "running",
                "agent_type": "simple",
            }
            assert "private household" not in response.text
            assert "shell_exec" not in response.text
        finally:
            _SPAWNED_AGENTS.pop("private-agent", None)

    def test_create_agent(self):
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        client = TestClient(_make_app())
        resp = client.post(
            "/v1/agents",
            json={
                "agent_type": "simple",
                "agent_id": "denied-server-api-agent",
            },
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == (
            "Authenticated API principal is not configured"
        )
        assert "denied-server-api-agent" not in _SPAWNED_AGENTS

    def test_create_agent_with_explicit_server_policy(self):
        from openjarvis.security.capabilities import CapabilityPolicy
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        policy = CapabilityPolicy()
        policy.grant("server-api", "tool:invoke", "tool:agent_spawn")
        policy.grant("server-api", "system:admin", "tool:agent_spawn")
        client = TestClient(_make_app(policy, authenticated=True))
        try:
            resp = client.post(
                "/v1/agents",
                json={
                    "agent_type": "simple",
                    "agent_id": "allowed-server-api-agent",
                },
            )
            assert resp.status_code == 200
            assert "allowed-server-api-agent" in _SPAWNED_AGENTS
        finally:
            _SPAWNED_AGENTS.pop("allowed-server-api-agent", None)

    def test_kill_nonexistent(self):
        client = TestClient(_make_app())
        resp = client.delete("/v1/agents/nonexistent")
        assert resp.status_code == 503
        assert resp.json()["detail"] == (
            "Authenticated API principal is not configured"
        )


class TestMemoryRoutes:
    def test_search(self):
        client = TestClient(_make_app())
        resp = client.post("/v1/memory/search", json={"query": "test"})
        # The optional Rust backend may be absent in a pure-Python test env.
        assert resp.status_code in (200, 503)

    def test_stats(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/memory/stats")
        assert resp.status_code in (200, 503)


class TestMemoryRustMissing:
    """Regression for #502: when the native ``openjarvis_rust`` extension is
    missing from the serving venv, memory ops must surface a CLEAR, ACTIONABLE
    error — never the misleading "Failed to index path" or a 200 silent no-op.
    """

    @staticmethod
    def _client(monkeypatch):
        # Force the same failure mode as a venv without the compiled extension.
        def _boom():
            raise ImportError("No module named 'openjarvis_rust'")

        import openjarvis._rust_bridge as bridge

        monkeypatch.setattr(bridge, "get_rust_module", _boom)
        from openjarvis.security.capabilities import CapabilityPolicy

        policy = CapabilityPolicy()
        for capability in ("tool:invoke", "file:read", "memory:write"):
            policy.grant("server-api", capability, "*")
        return TestClient(_make_app(policy, authenticated=True))

    def test_store_is_not_a_silent_noop(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post("/v1/memory/store", json={"content": "hi"})
        # Must NOT return the old 200 {"status":"stored","note":"no backend..."}.
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "openjarvis_rust" in detail
        assert "maturin develop" in detail

    def test_index_surfaces_actionable_detail(self, monkeypatch, tmp_path):
        (tmp_path / "note.txt").write_text("hello world some content here")
        monkeypatch.setenv("OPENJARVIS_WORKSPACE", str(tmp_path))
        client = self._client(monkeypatch)
        resp = client.post("/v1/memory/index", json={"path": str(tmp_path)})
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        # The frontend reads this `detail`; it must point at the real cause,
        # not blame the indexed path.
        assert "openjarvis_rust" in detail
        assert detail != "Failed to index path"
        assert detail != "No memory backend available"


class TestMemoryIndexAuthorization:
    @staticmethod
    def _client(capabilities):
        from openjarvis.security.capabilities import CapabilityPolicy

        policy = CapabilityPolicy()
        for capability in capabilities:
            policy.grant("server-api", capability, "*")
        app = _make_app(policy, authenticated=True)
        app.state.memory_backend = MagicMock()
        return TestClient(app), app.state.memory_backend

    def test_workspace_is_mandatory(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENJARVIS_WORKSPACE", raising=False)
        target = tmp_path / "note.txt"
        target.write_text("content long enough to be indexed")
        client, backend = self._client({"tool:invoke", "file:read", "memory:write"})

        response = client.post("/v1/memory/index", json={"path": str(target)})

        assert response.status_code == 503
        assert "workspace" in response.json()["detail"].lower()
        backend.store.assert_not_called()

    def test_target_must_stay_inside_workspace(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("content long enough to be indexed")
        monkeypatch.setenv("OPENJARVIS_WORKSPACE", str(workspace))
        client, backend = self._client({"tool:invoke", "file:read", "memory:write"})

        response = client.post("/v1/memory/index", json={"path": str(outside)})

        assert response.status_code == 403
        backend.store.assert_not_called()

    def test_file_read_capability_is_required(self, monkeypatch, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("content long enough to be indexed")
        monkeypatch.setenv("OPENJARVIS_WORKSPACE", str(tmp_path))
        client, backend = self._client({"tool:invoke", "memory:write"})

        response = client.post("/v1/memory/index", json={"path": str(target)})

        assert response.status_code == 403
        backend.store.assert_not_called()

    def test_authorized_index_uses_tool_executor(self, monkeypatch, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("content long enough to be indexed")
        monkeypatch.setenv("OPENJARVIS_WORKSPACE", str(tmp_path))
        client, backend = self._client({"tool:invoke", "file:read", "memory:write"})

        response = client.post("/v1/memory/index", json={"path": str(target)})

        assert response.status_code == 200
        assert response.json()["chunks_indexed"] >= 1
        backend.store.assert_called()

    def test_config_reports_unavailable(self, monkeypatch):
        def _boom():
            raise ImportError("No module named 'openjarvis_rust'")

        import openjarvis._rust_bridge as bridge

        monkeypatch.setattr(bridge, "get_rust_module", _boom)
        client = TestClient(_make_app())
        resp = client.get("/v1/memory/config")
        assert resp.status_code == 200
        data = resp.json()
        # Must not falsely report a healthy backend when none could be built.
        assert data["available"] is False
        assert "openjarvis_rust" in (data["detail"] or "")


class TestBudgetRoutes:
    def test_get_budget(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/budget")
        assert resp.status_code == 200
        data = resp.json()
        assert "limits" in data
        assert "usage" in data

    def test_set_limits(self):
        client = TestClient(_make_app())
        resp = client.put("/v1/budget/limits", json={"max_tokens_per_day": 100000})
        assert resp.status_code == 200
        assert resp.json()["limits"]["max_tokens_per_day"] == 100000


class TestMetricsRoute:
    def test_metrics_endpoint(self):
        client = TestClient(_make_app())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "openjarvis" in resp.text or "No metrics" in resp.text


class TestSkillRoutes:
    def test_list_skills(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/skills")
        assert resp.status_code == 200
        assert "skills" in resp.json()


class TestSessionRoutes:
    def test_list_sessions(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/sessions")
        assert resp.status_code == 200


class TestTraceRoutes:
    def test_list_traces(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/traces")
        assert resp.status_code == 200
