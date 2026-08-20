"""Deployment configs must not ship an unauthenticated public server (#221).

Every shipped deployment method must either bind loopback (no network
exposure) or require an API key, so that following the docs never yields an
open `0.0.0.0:8000` server. `check_bind_safety` is the runtime backstop;
these tests guard the static config files that drive it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = REPO_ROOT / "deploy"
TAURI_LIB = REPO_ROOT / "frontend" / "src-tauri" / "src" / "lib.rs"
SETTINGS_PAGE = REPO_ROOT / "frontend" / "src" / "pages" / "SettingsPage.tsx"


def _read(rel: str) -> str:
    return (DEPLOY / rel).read_text()


def test_docker_compose_requires_api_key():
    text = _read("docker/docker-compose.yml")
    # The container binds 0.0.0.0, so the key must be a *required* variable
    # (compose's ${VAR:?...} fails fast when unset).
    assert "OPENJARVIS_API_KEY" in text
    assert "OPENJARVIS_API_KEY:?" in text


def test_docker_env_example_present():
    assert (DEPLOY / "docker" / ".env.example").is_file()
    assert "OPENJARVIS_API_KEY" in _read("docker/.env.example")


def test_systemd_unit_binds_public_and_requires_env_file():
    text = _read("systemd/openjarvis.service")
    # Public bind -> must pull in an EnvironmentFile (no leading '-', so the
    # unit fails to start if it's missing).
    assert "--host 0.0.0.0" in text
    assert "EnvironmentFile=/etc/openjarvis/env" in text
    assert "\n-EnvironmentFile" not in text and "=-/etc" not in text


def test_launchd_plist_binds_loopback():
    text = _read("launchd/com.openjarvis.plist")
    # Personal-device default: loopback, not the network.
    assert "<string>127.0.0.1</string>" in text
    assert "<string>0.0.0.0</string>" not in text


def test_skynet_l99_policy_is_narrow_and_default_deny_compatible():
    policy_path = DEPLOY / "launchd" / "skynet-l99-capabilities.json"
    policy = json.loads(policy_path.read_text())

    assert set(policy) == {"agents"}
    assert len(policy["agents"]) == 2
    agents = {agent["agent_id"]: agent for agent in policy["agents"]}
    agent = agents["orchestrator"]
    assert agent["agent_id"] == "orchestrator"
    assert set(agent) == {"agent_id", "grants", "deny"}

    grants = {(grant["capability"], grant["pattern"]) for grant in agent["grants"]}
    assert ("tool:invoke", "skynet://*") in grants
    assert ("network:fetch", "skynet://*") in grants
    assert (
        "skynet:casa:write",
        "skynet://casa/tomada_cozinha_backlight",
    ) in grants
    assert not any(
        capability in {"code:execute", "file:write", "channel:send"}
        for capability, _ in grants
    )
    assert not any(pattern in {"*", "http://*", "https://*"} for _, pattern in grants)

    desktop = agents["api:l99-desktop"]
    desktop_grants = {
        (grant["capability"], grant["pattern"]) for grant in desktop["grants"]
    }
    assert desktop_grants == {
        ("tool:invoke", "/v1/chat/completions"),
        ("tool:invoke", "/v1/speech/transcribe"),
    }
    assert "system:*" in desktop["deny"]
    assert not any(pattern == "*" for _, pattern in desktop_grants)


def test_tauri_server_uses_the_scoped_l99_principal_and_fail_closed_client():
    text = TAURI_LIB.read_text()
    compact = "".join(text.split())

    assert 'const OPENJARVIS_API_PRINCIPAL: &str = "api:l99-desktop";' in text
    assert (
        'command.env("OPENJARVIS_API_PRINCIPAL",OPENJARVIS_API_PRINCIPAL);' in compact
    )
    assert '"OPENJARVIS_API_PRINCIPAL_ALLOWLIST",OPENJARVIS_API_PRINCIPAL' in compact
    assert "required_local_api_key()?" in text
    assert ".redirect(reqwest::redirect::Policy::none())" in text


def test_desktop_settings_sync_the_local_api_key_with_keychain():
    text = SETTINGS_PAGE.read_text()

    assert "saveCloudKey('OPENJARVIS_API_KEY', next)" in text
    assert "saveCloudKey('OPENJARVIS_API_KEY', '')" in text
    assert "Saved securely in Keychain." in text
    assert (
        "Removed from Keychain. This session keeps using its current key "
        "until OpenJarvis restarts."
    ) in text


@pytest.mark.parametrize(
    ("host", "api_key", "should_exit"),
    [
        ("127.0.0.1", "", False),
        ("localhost", "", False),
        ("0.0.0.0", "oj_sk_x", False),
        ("0.0.0.0", "", True),
        ("192.168.1.10", "", True),
    ],
)
def test_check_bind_safety(host, api_key, should_exit):
    from openjarvis.server.auth_middleware import check_bind_safety

    if should_exit:
        with pytest.raises(SystemExit):
            check_bind_safety(host, api_key=api_key)
    else:
        check_bind_safety(host, api_key=api_key)  # must not raise
