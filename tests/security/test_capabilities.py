"""Tests for RBAC capabilities system (Phase 14.4)."""

from __future__ import annotations

from openjarvis import _rust_bridge
from openjarvis.security.capabilities import (
    DEFAULT_TOOL_CAPABILITIES,
    Capability,
    CapabilityPolicy,
)


class TestCapability:
    def test_capability_values(self):
        assert Capability.FILE_READ == "file:read"
        assert Capability.NETWORK_FETCH == "network:fetch"
        assert Capability.CODE_EXECUTE == "code:execute"
        assert Capability.SYSTEM_ADMIN == "system:admin"

    def test_all_capabilities_exist(self):
        expected = {
            "file:read",
            "file:write",
            "network:fetch",
            "code:execute",
            "memory:read",
            "memory:write",
            "channel:send",
            "email:write",
            "calendar:write",
            "message:send",
            "approval:decide",
            "tool:invoke",
            "schedule:create",
            "system:admin",
        }
        actual = {c.value for c in Capability}
        assert expected == actual


class TestCapabilityPolicy:
    def test_default_deny(self):
        policy = CapabilityPolicy()
        assert not policy.check("agent1", "file:read")
        assert not policy.check("agent1", "code:execute")

    def test_explicit_legacy_allow(self):
        policy = CapabilityPolicy(default_deny=False)
        assert policy.check("agent1", "file:read")

    def test_explicit_grant(self):
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:read")
        assert policy.check("agent1", "file:read")
        assert not policy.check("agent1", "code:execute")

    def test_explicit_deny(self):
        policy = CapabilityPolicy(default_deny=False)
        policy.deny("agent1", "code:execute")
        assert not policy.check("agent1", "code:execute")
        assert policy.check("agent1", "file:read")

    def test_deny_overrides_grant(self):
        policy = CapabilityPolicy(default_deny=False)
        policy.grant("agent1", "code:execute")
        policy.deny("agent1", "code:execute")
        assert not policy.check("agent1", "code:execute")

    def test_resource_pattern(self):
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:read", pattern="/safe/*")
        assert policy.check("agent1", "file:read", "/safe/data.txt")
        assert not policy.check("agent1", "file:read", "/etc/passwd")
        assert not policy.check("agent1", "file:read", "")

    def test_glob_pattern(self):
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:*")
        assert policy.check("agent1", "file:read")
        assert policy.check("agent1", "file:write")
        assert not policy.check("agent1", "code:execute")

    def test_glob_is_case_sensitive_and_only_star_is_special(self):
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:*", "/safe/*/report.txt")
        assert policy.check(
            "agent1",
            "file:read",
            "/safe/team/report.txt",
        )
        assert not policy.check(
            "agent1",
            "FILE:READ",
            "/safe/team/report.txt",
        )
        assert not policy.check(
            "agent1",
            "file:read",
            "/safe/team/REPORT.txt",
        )

    def test_unsupported_glob_metacharacters_fail_closed(self):
        malformed_grant = CapabilityPolicy(default_deny=False)
        malformed_grant.grant("agent1", "file:?", "*")
        assert not malformed_grant.check("agent1", "file:read")

        malformed_resource = CapabilityPolicy(default_deny=False)
        malformed_resource.grant("agent1", "file:read", "/safe/[ab]*")
        assert not malformed_resource.check(
            "agent1",
            "file:read",
            "/safe/a.txt",
        )

        malformed_deny = CapabilityPolicy(default_deny=False)
        malformed_deny.deny("agent1", "code:[a-z]*")
        assert not malformed_deny.check("agent1", "file:read")

    def test_list_grants(self):
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:read")
        policy.grant("agent1", "code:execute")
        grants = policy.list_grants("agent1")
        assert len(grants) == 2

    def test_list_agents(self):
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:read")
        policy.grant("agent2", "code:execute")
        agents = policy.list_agents()
        assert set(agents) == {"agent1", "agent2"}

    def test_no_policy_agent(self):
        policy = CapabilityPolicy()
        assert policy.list_grants("unknown") == []

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "policy.json"
        policy = CapabilityPolicy()
        policy.grant("agent1", "file:read")
        policy.deny("agent1", "code:execute")
        policy.save(path)

        loaded = CapabilityPolicy(policy_path=str(path))
        assert loaded.check("agent1", "file:read")
        assert not loaded.check("agent1", "code:execute")

    def test_load_nonexistent_file(self):
        policy = CapabilityPolicy(policy_path="/nonexistent/path.json")
        # Should not raise, just have no policies
        assert not policy.check("agent1", "file:read")

    def test_python_fallback_is_deny_by_default(self, monkeypatch):
        def _missing_rust():
            raise ImportError("native extension missing")

        monkeypatch.setattr(_rust_bridge, "get_rust_module", _missing_rust)
        policy = CapabilityPolicy()

        assert policy._rust_impl is None
        assert not policy.check("agent1", "tool:invoke", "safe")
        policy.grant("agent1", "tool:invoke", "safe")
        assert policy.check("agent1", "tool:invoke", "safe")

    def test_explicitly_disabled_policy_fails_closed(self):
        policy = CapabilityPolicy(enabled=False)
        assert not policy.check("", "tool:invoke", "anything")

    def test_default_tool_capabilities(self):
        assert "file:read" in DEFAULT_TOOL_CAPABILITIES.get("file_read", [])
        assert "network:fetch" in DEFAULT_TOOL_CAPABILITIES.get("web_search", [])
        assert "code:execute" in DEFAULT_TOOL_CAPABILITIES.get("code_interpreter", [])
