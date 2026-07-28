from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import runpy
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = REPO_ROOT / "deploy"
CONFIG = DEPLOY / "config" / "skynet-l99-shadow.toml"
POLICY = DEPLOY / "config" / "skynet-l99-shadow-capabilities.json"
HELPER = DEPLOY / "launchd" / "openjarvis-skynet-l99-shadow.py"
PLIST = DEPLOY / "launchd" / "com.openjarvis.skynet-l99-shadow.plist"
MANAGER = DEPLOY / "launchd" / "manage-skynet-l99-shadow.sh"
TAURI_LIB = REPO_ROOT / "frontend" / "src-tauri" / "src" / "lib.rs"

READ_TOOLS = {
    "skynet_casa_read",
    "skynet_agenda_read",
    "skynet_frota_read",
}
READ_TOKENS = {
    "OPENJARVIS_API_KEY",
    "SKYNET_JARVIS_CASA_READ_TOKEN",
    "SKYNET_JARVIS_AGENDA_READ_TOKEN",
    "SKYNET_JARVIS_FROTA_READ_TOKEN",
}


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _tool_names(value: str) -> set[str]:
    return {name.strip() for name in value.split(",") if name.strip()}


def test_shadow_profile_is_local_only_and_read_only() -> None:
    config = _load_toml(CONFIG)
    profile_text = CONFIG.read_text()

    assert config["engine"]["default"] == "ollama"
    assert config["engine"]["ollama"]["host"] == "http://127.0.0.1:11434"
    assert config["server"] == {
        "host": "127.0.0.1",
        "port": 8000,
        "agent": "orchestrator",
        "workers": 1,
        "cors_origins": [
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ],
        "cloud_enabled": False,
    }
    assert _tool_names(config["agent"]["tools"]) == READ_TOOLS
    assert _tool_names(config["tools"]["enabled"]) == READ_TOOLS
    assert config["agent"]["context_from_memory"] is False
    assert config["security"]["mode"] == "block"
    assert config["security"]["enforce_tool_confirmation"] is True
    assert config["security"]["local_engine_bypass"] is False
    assert config["security"]["local_tool_bypass"] is False
    assert config["security"]["capabilities"]["enabled"] is True
    assert config["security"]["capabilities"]["default_deny"] is True

    disabled_sections = (
        ("tools", "storage"),
        ("tools", "mcp"),
        ("learning",),
        ("telemetry",),
        ("analytics",),
        ("traces",),
        ("channel",),
        ("sandbox",),
        ("scheduler",),
        ("workflow",),
        ("sessions",),
        ("a2a",),
        ("operators",),
        ("agent_manager",),
        ("skills",),
        ("digest",),
        ("proactive",),
        ("memory",),
        ("compression",),
    )
    for section_path in disabled_sections:
        section = config
        for name in section_path:
            section = section[name]
        assert section["enabled"] is False, ".".join(section_path)
    assert config["speech"]["backend"] == "disabled"
    assert "identidade já é isolada por request" in profile_text
    assert "não\n# interrompe imediatamente a thread síncrona" in profile_text
    assert "read-only/default-deny e sem qualquer tool mutante" in profile_text


def test_shadow_policy_grants_only_three_read_boundaries() -> None:
    policy = json.loads(POLICY.read_text())
    agents = {agent["agent_id"]: agent for agent in policy["agents"]}
    assert set(agents) == {"orchestrator", "api:l99-desktop"}

    grants = {
        (grant["capability"], grant["pattern"])
        for grant in agents["orchestrator"]["grants"]
    }
    assert grants == {
        ("tool:invoke", "skynet://casa/read/*"),
        ("network:fetch", "skynet://casa/read/*"),
        ("skynet:casa:read", "skynet://casa/read/*"),
        ("tool:invoke", "skynet://agenda/busy"),
        ("network:fetch", "skynet://agenda/busy"),
        ("skynet:agenda:read", "skynet://agenda/busy"),
        ("tool:invoke", "skynet://frota/*"),
        ("network:fetch", "skynet://frota/*"),
        ("skynet:frota:read", "skynet://frota/*"),
    }
    assert not any(
        marker in capability.lower() or marker in pattern.lower()
        for capability, pattern in grants
        for marker in ("write", "action", "approval", "reminder", "schedule")
    )
    assert ("skynet:casa:write") in agents["orchestrator"]["deny"]
    assert ("skynet:casa:approval:*") in agents["orchestrator"]["deny"]
    assert ("skynet:agenda:reminder:*") in agents["orchestrator"]["deny"]

    desktop_grants = {
        (grant["capability"], grant["pattern"])
        for grant in agents["api:l99-desktop"]["grants"]
    }
    assert desktop_grants == grants | {
        ("tool:invoke", "/v1/chat/completions"),
    }

    denied_actions = {
        "skynet:casa:approval:*",
        "skynet:casa:write",
        "skynet:agenda:reminder:*",
    }
    for agent_id in ("orchestrator", "api:l99-desktop"):
        assert denied_actions <= set(agents[agent_id]["deny"])
        assert not any(
            marker in grant["capability"].lower() or marker in grant["pattern"].lower()
            for grant in agents[agent_id]["grants"]
            for marker in ("write", "action", "approval", "reminder", "schedule")
        )


def test_shadow_plist_is_dormant_private_and_contains_no_secret() -> None:
    raw = PLIST.read_bytes()
    plist = plistlib.loads(raw)

    assert plist["Label"] == "com.openjarvis.skynet-l99-shadow"
    assert plist["ProgramArguments"] == [
        "/usr/bin/python3",
        "-I",
        "-S",
        "__OPENJARVIS_SHADOW_HELPER__",
    ]
    assert plist["WorkingDirectory"] == "__OPENJARVIS_SHADOW_ROOT__"
    assert plist["RunAtLoad"] is False
    assert plist["KeepAlive"] is False
    assert plist["ThrottleInterval"] >= 30
    assert plist["Umask"] == 0o77
    assert plist["StandardOutPath"] == "__OPENJARVIS_SHADOW_STDOUT__"
    assert plist["StandardErrorPath"] == "__OPENJARVIS_SHADOW_STDERR__"
    assert "EnvironmentVariables" not in plist
    assert not any(secret.encode() in raw for secret in READ_TOKENS)

    if sys.platform == "darwin" and shutil.which("plutil"):
        result = subprocess.run(
            ["plutil", "-lint", str(PLIST)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS .pth contract")
def test_shadow_bootstrap_and_runtime_disable_dot_pth_execution(
    tmp_path: Path,
) -> None:
    import venv

    managed_venv = tmp_path / "managed-venv"
    venv.EnvBuilder(with_pip=False).create(managed_venv)
    managed_python = managed_venv / "bin" / "python"
    version = subprocess.run(
        [
            str(managed_python),
            "-I",
            "-S",
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2])",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    site_packages = managed_venv / "lib" / f"python{version}" / "site-packages"
    marker = tmp_path / "dot-pth-executed"
    (site_packages / "malicious.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('executed')\n"
    )

    subprocess.run(
        [str(managed_python), "-I", "-c", "pass"],
        check=True,
    )
    assert marker.is_file(), "fixture must reproduce Python's pre-script .pth hook"
    marker.unlink()

    subprocess.run(
        [str(managed_python), "-I", "-S", "-P", "-c", "pass"],
        check=True,
    )
    assert not marker.exists()
    assert plistlib.loads(PLIST.read_bytes())["ProgramArguments"][:3] == [
        "/usr/bin/python3",
        "-I",
        "-S",
    ]
    assert '"-S",\n        "-P",\n        "-m",' in HELPER.read_text()


def test_shadow_helper_uses_exact_keychain_accounts_and_direct_runtime() -> None:
    text = HELPER.read_text()

    assert 'KEYCHAIN_SECURITY_BIN = "/usr/bin/security"' in text
    assert 'LOCAL_KEYCHAIN_SERVICE = "OpenJarvis Cloud Keys"' in text
    assert 'SHADOW_KEYCHAIN_SERVICE = "OpenJarvis L99 Shadow Keys"' in text
    assert text.count('"find-generic-password"') == 1
    for account in READ_TOKENS:
        assert f'"{account}"' in text
    assert "SKYNET_JARVIS_CASA_ACTION_TOKEN" not in text
    assert "SKYNET_JARVIS_REMINDER_PLAN_TOKEN" not in text
    assert "os.environ.clear()" in text
    assert "shell=False" in text
    assert "stderr=subprocess.DEVNULL" in text
    assert "os.execve" in text
    assert "/usr/bin/env -i" not in text
    assert "$HOME/.local/bin/jarvis" not in text
    assert 'API_PRINCIPAL = "api:l99-desktop"' in text
    assert "startup denied by local safety gate" in text
    assert "openjarvis.skynet-l99.install.v1" in text
    assert "openjarvis.skynet-l99.runtime-manifest.v1" in text
    assert "openjarvis.skynet-l99.runtime-tree.v1" in text
    assert "O_NOFOLLOW" in text
    assert "--validate-provenance-only" in text
    assert "--validate-installed-inputs-only" in text
    assert '"-S"' in text
    assert '"-P"' in text
    assert "EXPECTED_CONFIG_SHA256" in text
    assert "EXPECTED_POLICY_SHA256" in text
    assert "tempfile.TemporaryFile" in text


def test_shadow_manager_never_loads_or_starts_the_service() -> None:
    text = MANAGER.read_text()

    assert "launchctl" not in text
    assert "RunAtLoad" in text
    assert "KeepAlive" in text
    assert "install_dormant" in text
    assert "rollback_dormant_files" in text
    assert "no service was loaded or started" in text
    assert '[[ "$HOME" =~ ^/Users/[A-Za-z0-9._-]+$ ]]' in text
    assert "snapshot_versioned_sources" in text
    assert "openjarvis.skynet-l99.source-snapshot.v1" in text
    assert "SNAPSHOT_HELPER" in text
    assert "SNAPSHOT_CONFIG" in text
    assert "SNAPSHOT_POLICY" in text
    assert "__OPENJARVIS_SHADOW_HELPER__" in text
    assert "install <runtime-site> <runtime-receipt> <runtime-manifest>" in text
    assert "validate_runtime_provenance_contract" in text
    assert "validate_owned_directory" in text
    assert "validate_owned_regular_file" in text
    assert "runtime receipt/manifest validation failed" in text
    assert "openjarvis.skynet-l99.shadow-launcher.v3" in text
    assert "receipt-manifest-and-tree-pinned" in text
    assert "VERSIONED_SOURCE_PATHS" in text
    assert 'GIT_CONFIG_GLOBAL="/dev/null"' in text
    assert '"cat-file", "blob"' in text
    assert "openjarvis.skynet-l99.shadow-preinstall.v2" in text
    assert "restore_preinstall_backup" in text
    assert "trap install_error_handler ERR" in text
    assert "apply_staged_install" in text
    assert "directories" in text
    assert "mkdir -p" not in text


def test_tauri_keychain_allowlist_contains_only_l99_read_bindings() -> None:
    text = TAURI_LIB.read_text()

    assert 'const SECURE_KEY_SERVICE: &str = "OpenJarvis Cloud Keys";' in text
    assert (
        'const SHADOW_READ_TOKEN_SERVICE: &str = "OpenJarvis L99 Shadow Keys";'
    ) in text
    for account in READ_TOKENS:
        assert f'"{account}"' in text
    token_block = re.search(
        r"const SKYNET_L99_READ_TOKEN_NAMES: &\[&str\] = &\[(.*?)\];",
        text,
        re.DOTALL,
    )
    assert token_block is not None
    assert set(re.findall(r'"([A-Z0-9_]+)"', token_block.group(1))) == (
        READ_TOKENS - {"OPENJARVIS_API_KEY"}
    )
    managed_block = re.search(
        r"const MANAGED_CLOUD_KEY_NAMES: &\[&str\] = &\[(.*?)\];",
        text,
        re.DOTALL,
    )
    assert managed_block is not None
    for token in READ_TOKENS - {"OPENJARVIS_API_KEY"}:
        assert token not in managed_block.group(1)
    assert "validate_shadow_read_token_name" in text
    assert "shadow_secure_store_get" in text
    assert "shadow_secure_store_set" in text
    assert "save_shadow_read_token" in text


def _manager_python_block(marker: str) -> str:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", MANAGER.read_text(), re.DOTALL)
    matches = [block for block in blocks if marker in block]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Git snapshot contract")
def test_source_snapshot_is_commit_pinned_read_only_and_used_for_staging(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = [
        "frontend/src-tauri/src/lib.rs",
        "deploy/config/skynet-l99-shadow.toml",
        "deploy/config/skynet-l99-shadow-capabilities.json",
        "deploy/launchd/com.openjarvis.skynet-l99-shadow.plist",
        "deploy/launchd/manage-skynet-l99-shadow.sh",
        "deploy/launchd/openjarvis-skynet-l99-shadow.py",
        "tests/deploy/test_launchd_shadow.py",
    ]
    original = {}
    for index, relative in enumerate(paths):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"snapshot-{index}\n".encode()
        path.write_bytes(payload)
        if relative.endswith(("manage-skynet-l99-shadow.sh", ".py")):
            path.chmod(0o755)
        original[relative] = payload
    subprocess.run(["/usr/bin/git", "init", "-q", str(repo)], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(repo), "add", *paths], check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "user.name=OpenJarvis Test",
            "-c",
            "user.email=openjarvis-test@example.invalid",
            "-C",
            str(repo),
            "commit",
            "-qm",
            "snapshot fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(mode=0o700)

    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-",
            str(repo),
            commit,
            str(snapshot),
            *paths,
        ],
        input=_manager_python_block("source-snapshot-builder-v1"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    (repo / paths[0]).write_text("mutated after snapshot\n")
    for relative, payload in original.items():
        snapshotted = snapshot / relative
        assert snapshotted.read_bytes() == payload
        expected_mode = (
            0o500
            if relative.endswith(("manage-skynet-l99-shadow.sh", ".py"))
            else 0o400
        )
        assert stat.S_IMODE(snapshotted.stat().st_mode) == expected_mode
    manifest = json.loads((snapshot / "source-snapshot-manifest.json").read_text())
    assert manifest["source_commit"] == commit
    assert len(manifest["entries"]) == 7

    manager_text = MANAGER.read_text()
    install_body = manager_text.split("install_dormant() {", 1)[1].split(
        "validate_installed() {", 1
    )[0]
    assert "SNAPSHOT_CONFIG" in install_body
    assert "SNAPSHOT_POLICY" in install_body
    assert "SNAPSHOT_HELPER" in install_body
    assert "SNAPSHOT_PLIST" in install_body
    assert "SOURCE_CONFIG" not in install_body
    assert "SOURCE_POLICY" not in install_body
    assert "SOURCE_HELPER" not in install_body
    assert "SOURCE_PLIST" not in install_body

    for directory in sorted(
        (path for path in snapshot.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o700)
    snapshot.chmod(0o700)


@pytest.mark.skipif(
    sys.platform != "darwin" or os.geteuid() == 0,
    reason="macOS transactional permission contract",
)
def test_staged_install_compensates_after_mid_apply_failure(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir(mode=0o700)
    modes = (0o600, 0o600, 0o700, 0o600, 0o600, 0o600, 0o600)
    arguments: list[str] = []
    targets: list[Path] = []
    original: dict[Path, bytes] = {}
    readonly_parent: Path | None = None
    for index, mode in enumerate(modes):
        stage = stage_dir / f"stage-{index}"
        stage.write_bytes(f"new-{index}\n".encode())
        stage.chmod(mode)
        parent = tmp_path / f"target-parent-{index}"
        parent.mkdir(mode=0o700)
        target = parent / f"target-{index}"
        payload = f"old-{index}\n".encode()
        target.write_bytes(payload)
        target.chmod(mode)
        targets.append(target)
        original[target] = payload
        arguments.extend((str(stage), str(target), f"{mode:03o}"))
        if index == 3:
            readonly_parent = parent
    assert readonly_parent is not None
    readonly_parent.chmod(0o500)

    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-", *arguments],
        input=_manager_python_block("openjarvis.skynet-l99.shadow-install-atomic.v1"),
        check=False,
        capture_output=True,
        text=True,
    )
    readonly_parent.chmod(0o700)

    assert result.returncode == 78
    for target in targets:
        assert target.read_bytes() == original[target]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS rollback contract")
@pytest.mark.parametrize("unexpected_file", (False, True))
def test_shadow_rollback_restores_present_and_removes_absent_files(
    tmp_path: Path,
    unexpected_file: bool,
) -> None:
    root = tmp_path.resolve()
    rollback_root = root / "rollbacks"
    backup = rollback_root / "preinstall-20260728T170856Z.ABC123"
    destination = rollback_root / "shadow-20260728T180000Z.DEF456"
    launch_agents = root / "Library" / "LaunchAgents"
    shadow = root / "shadow"
    config_dir = shadow / "config"
    bin_dir = shadow / "bin"
    log_dir = shadow / "logs"
    state_dir = shadow / "state"
    for directory in (
        rollback_root,
        backup,
        destination,
        launch_agents,
        config_dir,
        bin_dir,
        log_dir,
        state_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)

    labels = (
        "plist",
        "helper",
        "config",
        "policy",
        "stdout",
        "stderr",
        "receipt",
    )
    targets = (
        launch_agents / "com.openjarvis.skynet-l99-shadow.plist",
        bin_dir / "openjarvis-skynet-l99-shadow.py",
        config_dir / "skynet-l99-shadow.toml",
        config_dir / "skynet-l99-shadow-capabilities.json",
        log_dir / "shadow.stdout.log",
        log_dir / "shadow.stderr.log",
        state_dir / "install-receipt.json",
    )
    modes = (0o600, 0o700, 0o600, 0o600, 0o600, 0o600, 0o600)
    backup_names = (
        "plist",
        "helper.py",
        "config.toml",
        "policy.json",
        "stdout.log",
        "stderr.log",
        "receipt.json",
    )
    preinstall_payloads = {
        "config": b"old-config\n",
        "stdout": b"old-log\n",
    }
    entries = []
    for label, target, mode, backup_name in zip(
        labels,
        targets,
        modes,
        backup_names,
    ):
        payload = preinstall_payloads.get(label)
        if payload is None:
            entries.append(
                {
                    "label": label,
                    "target": str(target),
                    "backup_name": None,
                    "present": False,
                    "sha256": None,
                    "mode": f"{mode:03o}",
                }
            )
            continue
        backup_file = backup / backup_name
        backup_file.write_bytes(payload)
        backup_file.chmod(mode)
        entries.append(
            {
                "label": label,
                "target": str(target),
                "backup_name": backup_name,
                "present": True,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mode": f"{mode:03o}",
            }
        )
    backup_manifest = backup / "backup-manifest.json"
    backup_manifest.write_text(
        json.dumps(
            {
                "schema": "openjarvis.skynet-l99.shadow-preinstall.v2",
                "backup": str(backup),
                "files": entries,
                "directories": [
                    {
                        "label": "launch_agents",
                        "path": str(launch_agents),
                        "present": True,
                        "mode": "700",
                    },
                    {
                        "label": "shadow",
                        "path": str(shadow),
                        "present": True,
                        "mode": "700",
                    },
                    {
                        "label": "config",
                        "path": str(config_dir),
                        "present": True,
                        "mode": "700",
                    },
                    {
                        "label": "bin",
                        "path": str(bin_dir),
                        "present": False,
                        "mode": None,
                    },
                    {
                        "label": "logs",
                        "path": str(log_dir),
                        "present": True,
                        "mode": "700",
                    },
                    {
                        "label": "state",
                        "path": str(state_dir),
                        "present": False,
                        "mode": None,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    backup_manifest.chmod(0o600)
    backup_manifest_sha256 = hashlib.sha256(backup_manifest.read_bytes()).hexdigest()

    for label, target, mode in zip(labels[:-1], targets[:-1], modes[:-1]):
        target.write_bytes(f"new-{label}\n".encode())
        target.chmod(mode)
    targets[-1].write_text(
        json.dumps(
            {
                "schema": "openjarvis.skynet-l99.shadow-launcher.v3",
                "preinstall": {
                    "backup": str(backup),
                    "manifest": str(backup_manifest),
                    "manifest_sha256": backup_manifest_sha256,
                },
            }
        )
    )
    targets[-1].chmod(0o600)
    installed_payloads = {target: target.read_bytes() for target in targets}
    unexpected = bin_dir / "unexpected"
    if unexpected_file:
        unexpected.write_text("must block exact directory removal\n")
        unexpected.chmod(0o600)

    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-",
            str(targets[-1]),
            str(rollback_root),
            str(destination),
            *(str(path) for path in targets),
            "--directories--",
            str(launch_agents),
            str(shadow),
            str(config_dir),
            str(bin_dir),
            str(log_dir),
            str(state_dir),
        ],
        input=_manager_python_block("openjarvis.skynet-l99.shadow-rollback.v2"),
        check=False,
        capture_output=True,
        text=True,
    )

    if unexpected_file:
        assert result.returncode == 78
        for target, payload in installed_payloads.items():
            assert target.read_bytes() == payload
        assert unexpected.is_file()
        for directory in (
            launch_agents,
            shadow,
            config_dir,
            bin_dir,
            log_dir,
            state_dir,
        ):
            assert directory.is_dir()
        return

    assert result.returncode == 0, result.stderr
    assert targets[2].read_bytes() == preinstall_payloads["config"]
    assert targets[4].read_bytes() == preinstall_payloads["stdout"]
    for index in (0, 1, 3, 5, 6):
        assert not targets[index].exists()
    assert not bin_dir.exists()
    assert not state_dir.exists()
    assert config_dir.is_dir()
    assert log_dir.is_dir()
    for label in labels:
        assert (destination / label).is_file()
    assert (destination / "rollback-manifest.json").is_file()


def _render_test_helper(
    tmp_path: Path,
    *,
    security_script: str,
) -> tuple[Path, Path, Path, Path]:
    home = tmp_path.resolve() / "Users" / "paulobrito"
    runtime_site = (
        home / ".openjarvis" / "skynet-l99" / "runtimes" / ("a" * 64) / "site-packages"
    )
    (runtime_site / "openjarvis").mkdir(parents=True)
    package_marker = runtime_site / "openjarvis" / "__init__.py"
    package_marker.write_text("")
    (runtime_site / "openjarvis" / "cli.py").write_text(
        """import os
import sys
from pathlib import Path

state = Path(os.environ["HOME"]) / ".openjarvis/skynet-l99/shadow/state"
state.mkdir(parents=True, exist_ok=True)
(state / "trusted-runtime").write_text("trusted\\n")
(state / "trusted-sys-path").write_text("\\n".join(sys.path) + "\\n")
"""
    )

    source_commit = "b" * 40
    runtime_archive_sha256 = "c" * 64
    manifest_root = home / ".openjarvis" / "skynet-l99" / "manifests"
    receipt_root = home / ".openjarvis" / "skynet-l99" / "receipts"
    manifest_root.mkdir(parents=True)
    receipt_root.mkdir(parents=True)
    manifest = manifest_root / f"{source_commit[:12]}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "openjarvis.skynet-l99.runtime-manifest.v1",
                "source_commit": source_commit,
                "runtime_version": "test",
                "platform": "darwin-arm64",
                "python_abi": "cp312",
                "openjarvis_rust": {},
                "wheel": {
                    "file": "openjarvis-test.whl",
                    "sha256": "a" * 64,
                    "bytes": 1,
                },
                "runtime_archive": {
                    "file": "openjarvis-test-runtime.tar.zst",
                    "sha256": runtime_archive_sha256,
                    "bytes": 1,
                },
                "dependency_wheels": [],
                "profile_files": {},
                "activation": {"runtime_site": str(runtime_site)},
            },
            sort_keys=True,
        )
    )
    os.chmod(manifest, stat.S_IRUSR | stat.S_IWUSR)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt = receipt_root / "install-receipt-air-20260728T170856Z.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "openjarvis.skynet-l99.install.v1",
                "source_commit": source_commit,
                "wheel_sha256": "a" * 64,
                "runtime_archive_sha256": runtime_archive_sha256,
                "manifest_sha256": manifest_sha256,
                "runtime_site": str(runtime_site),
            },
            sort_keys=True,
        )
    )
    os.chmod(receipt, stat.S_IRUSR | stat.S_IWUSR)
    receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
    runtime_tree_sha256 = runpy.run_path(str(HELPER))["_runtime_tree_sha256"](
        runtime_site
    )

    shadow_config = home / ".openjarvis" / "skynet-l99" / "shadow" / "config"
    shadow_config.mkdir(parents=True)
    shadow_state = shadow_config.parent / "state"
    shadow_state.mkdir(mode=0o700)
    shutil.copyfile(CONFIG, shadow_config / CONFIG.name)
    shutil.copyfile(POLICY, shadow_config / POLICY.name)
    os.chmod(shadow_config / CONFIG.name, stat.S_IRUSR | stat.S_IWUSR)
    os.chmod(shadow_config / POLICY.name, stat.S_IRUSR | stat.S_IWUSR)
    config_sha256 = hashlib.sha256(
        (shadow_config / CONFIG.name).read_bytes()
    ).hexdigest()
    policy_sha256 = hashlib.sha256(
        (shadow_config / POLICY.name).read_bytes()
    ).hexdigest()

    security = tmp_path / "security"
    security.write_text(security_script)
    security.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    python_bin = home / ".openjarvis" / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        f"""#!{sys.executable}
import os
import re
import sys
from pathlib import Path

state = Path(os.environ["HOME"]) / ".openjarvis/skynet-l99/shadow/state"
state.mkdir(parents=True, exist_ok=True)
(state / "observed-config").write_text(
    Path(os.environ["OPENJARVIS_CONFIG"]).read_text()
)
policy_match = re.search(
    r'^policy_path = "([^"]+)"$',
    (state / "observed-config").read_text(),
    re.MULTILINE,
)
if policy_match is None:
    raise SystemExit(91)
(state / "observed-policy").write_text(
    Path(policy_match.group(1)).read_text()
)
(state / "observed-env").write_text("\\n".join(sorted(os.environ)) + "\\n")
(state / "observed-argv").write_text("\\n".join(sys.argv[1:]) + "\\n")
bindings = [
    os.environ["SKYNET_JARVIS_API_BASE_URL"],
    os.environ["SKYNET_JARVIS_CASA_READ_DEVICE_ID"],
    os.environ["SKYNET_JARVIS_CASA_READ_CHANNEL"],
    os.environ["SKYNET_JARVIS_AGENDA_READ_DEVICE_ID"],
    os.environ["SKYNET_JARVIS_AGENDA_READ_CHANNEL"],
    os.environ["SKYNET_JARVIS_FROTA_READ_DEVICE_ID"],
    os.environ["SKYNET_JARVIS_FROTA_READ_CHANNEL"],
]
(state / "observed-bindings").write_text("\\n".join(bindings) + "\\n")
"""
    )
    python_bin.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    text = HELPER.read_text()
    replacements = {
        "__OPENJARVIS_RUNTIME_SITE__": str(runtime_site),
        "__OPENJARVIS_RUNTIME_RECEIPT__": str(receipt),
        "__OPENJARVIS_RUNTIME_MANIFEST__": str(manifest),
        "__OPENJARVIS_RUNTIME_RECEIPT_SHA256__": receipt_sha256,
        "__OPENJARVIS_RUNTIME_MANIFEST_SHA256__": manifest_sha256,
        "__OPENJARVIS_RUNTIME_TREE_SHA256__": runtime_tree_sha256,
        "__OPENJARVIS_SOURCE_COMMIT__": source_commit,
        "__OPENJARVIS_SHADOW_CONFIG_SHA256__": config_sha256,
        "__OPENJARVIS_SHADOW_POLICY_SHA256__": policy_sha256,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    text = text.replace(
        'KEYCHAIN_SECURITY_BIN = "/usr/bin/security"',
        f"KEYCHAIN_SECURITY_BIN = {str(security)!r}",
    )
    text = text.replace(
        'AIR_HOME_PATTERN = r"^/Users/[A-Za-z0-9._-]+$"',
        f"AIR_HOME_PATTERN = {'^' + re.escape(str(home)) + '$'!r}",
    )
    helper = tmp_path / "openjarvis-skynet-l99-shadow.py"
    helper.write_text(text)
    helper.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return helper, home, receipt, manifest


def _shadow_helper_command(helper: Path, *arguments: str) -> list[str]:
    if sys.platform == "darwin":
        return ["/usr/bin/python3", "-I", "-S", str(helper), *arguments]
    return [sys.executable, "-I", "-S", str(helper), *arguments]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS safe-path contract")
def test_shadow_runtime_safe_path_ignores_cwd_decoy_package(
    tmp_path: Path,
) -> None:
    fake_value = "0123456789abcdef0123456789abcdef"
    helper, home, receipt, _ = _render_test_helper(
        tmp_path,
        security_script=f"#!/bin/bash\nprintf '%s' '{fake_value}'\n",
    )
    runtime_site = Path(json.loads(receipt.read_text())["runtime_site"])
    shadow_root = home / ".openjarvis" / "skynet-l99" / "shadow"
    state = shadow_root / "state"
    decoy_package = shadow_root / "openjarvis"
    decoy_package.mkdir()
    (decoy_package / "__init__.py").write_text("")
    (decoy_package / "cli.py").write_text(
        """import os
from pathlib import Path

state = Path(os.environ["HOME"]) / ".openjarvis/skynet-l99/shadow/state"
(state / "decoy-runtime").write_text("decoy\\n")
"""
    )

    baseline_environment = {
        "HOME": str(home),
        "PYTHONPATH": str(runtime_site),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    baseline = subprocess.run(
        [sys.executable, "-S", "-m", "openjarvis.cli"],
        cwd=shadow_root,
        env=baseline_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert baseline.returncode == 0, baseline.stderr
    assert (state / "decoy-runtime").is_file()
    assert not (state / "trusted-runtime").exists()
    (state / "decoy-runtime").unlink()

    python_bin = home / ".openjarvis" / ".venv" / "bin" / "python"
    python_bin.unlink()
    python_bin.symlink_to(Path(sys.executable).resolve())
    result = subprocess.run(
        _shadow_helper_command(helper),
        cwd=shadow_root,
        env={"HOME": str(home)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (state / "trusted-runtime").read_text() == "trusted\n"
    assert not (state / "decoy-runtime").exists()
    runtime_paths = (state / "trusted-sys-path").read_text().splitlines()
    assert str(runtime_site) in runtime_paths
    assert str(shadow_root) not in runtime_paths
    assert "" not in runtime_paths


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
@pytest.mark.parametrize("missing_account", sorted(READ_TOKENS))
def test_shadow_helper_fails_closed_when_keychain_item_is_missing(
    tmp_path: Path,
    missing_account: str,
) -> None:
    fake_value = "0123456789abcdef0123456789abcdef"
    helper, home, _, _ = _render_test_helper(
        tmp_path,
        security_script=f"""#!/bin/bash
set -eu
account=""
service=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-a" ]]; then
        account="$2"
        shift 2
    elif [[ "$1" == "-s" ]]; then
        service="$2"
        shift 2
    else
        shift
    fi
done
[[ "$account" != "{missing_account}" ]] || exit 44
printf '%s' '{fake_value}'
""",
    )
    result = subprocess.run(
        _shadow_helper_command(helper),
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": str(home),
            "OPENAI_API_KEY": "must-not-leak",
            "SKYNET_JARVIS_CASA_ACTION_TOKEN": "must-not-leak",
        },
    )

    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "openjarvis-shadow: startup denied by local safety gate"
    )
    state = home / ".openjarvis" / "skynet-l99" / "shadow" / "state"
    assert not (state / "observed-env").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
@pytest.mark.parametrize("metadata_name", ("receipt", "manifest"))
def test_shadow_helper_rejects_tampered_runtime_metadata(
    tmp_path: Path,
    metadata_name: str,
) -> None:
    helper, home, receipt, manifest = _render_test_helper(
        tmp_path,
        security_script="#!/bin/bash\nexit 99\n",
    )
    metadata = receipt if metadata_name == "receipt" else manifest
    metadata.write_bytes(metadata.read_bytes() + b"\n")

    result = subprocess.run(
        _shadow_helper_command(helper, "--validate-provenance-only"),
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(home)},
    )

    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "openjarvis-shadow: startup denied by local safety gate"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
def test_shadow_helper_rejects_validly_hashed_but_mismatched_receipt(
    tmp_path: Path,
) -> None:
    helper, home, receipt, _ = _render_test_helper(
        tmp_path,
        security_script="#!/bin/bash\nexit 99\n",
    )
    old_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt_data = json.loads(receipt.read_text())
    receipt_data["runtime_site"] = f"{receipt_data['runtime_site']}-other"
    receipt.write_text(json.dumps(receipt_data, sort_keys=True))
    new_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
    helper.write_text(helper.read_text().replace(old_sha256, new_sha256))
    helper.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    result = subprocess.run(
        _shadow_helper_command(helper, "--validate-provenance-only"),
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(home)},
    )

    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "openjarvis-shadow: startup denied by local safety gate"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
def test_shadow_helper_rejects_symlinked_receipt(tmp_path: Path) -> None:
    helper, home, receipt, _ = _render_test_helper(
        tmp_path,
        security_script="#!/bin/bash\nexit 99\n",
    )
    receipt_copy = receipt.with_name("receipt-copy.json")
    receipt_copy.write_bytes(receipt.read_bytes())
    os.chmod(receipt_copy, stat.S_IRUSR | stat.S_IWUSR)
    receipt.unlink()
    receipt.symlink_to(receipt_copy)

    result = subprocess.run(
        _shadow_helper_command(helper, "--validate-provenance-only"),
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(home)},
    )

    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "openjarvis-shadow: startup denied by local safety gate"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
@pytest.mark.parametrize("tamper_kind", ("byte", "symlink"))
def test_shadow_helper_rejects_runtime_tree_tampering(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    helper, home, receipt, _ = _render_test_helper(
        tmp_path,
        security_script="#!/bin/bash\nexit 99\n",
    )
    runtime_site = Path(json.loads(receipt.read_text())["runtime_site"])
    package_marker = runtime_site / "openjarvis" / "__init__.py"
    if tamper_kind == "byte":
        package_marker.write_text("tampered")
    else:
        (runtime_site / "unexpected-link").symlink_to(package_marker)

    result = subprocess.run(
        _shadow_helper_command(helper, "--validate-provenance-only"),
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(home)},
    )

    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "openjarvis-shadow: startup denied by local safety gate"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
@pytest.mark.parametrize("artifact_name", (CONFIG.name, POLICY.name))
@pytest.mark.parametrize("tamper_kind", ("byte", "symlink"))
def test_shadow_helper_rejects_tampered_config_or_policy(
    tmp_path: Path,
    artifact_name: str,
    tamper_kind: str,
) -> None:
    helper, home, _, _ = _render_test_helper(
        tmp_path,
        security_script="#!/bin/bash\nexit 99\n",
    )
    artifact = home / ".openjarvis" / "skynet-l99" / "shadow" / "config" / artifact_name
    if tamper_kind == "byte":
        artifact.write_bytes(artifact.read_bytes() + b"\n")
    else:
        replacement = tmp_path / f"{artifact_name}.replacement"
        replacement.write_bytes(artifact.read_bytes())
        replacement.chmod(0o600)
        artifact.unlink()
        artifact.symlink_to(replacement)

    result = subprocess.run(
        _shadow_helper_command(helper, "--validate-installed-inputs-only"),
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(home)},
    )

    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "openjarvis-shadow: startup denied by local safety gate"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper contract")
def test_shadow_helper_sanitizes_environment_without_logging_secrets(
    tmp_path: Path,
) -> None:
    fake_value = "0123456789abcdef0123456789abcdef"
    helper, home, _, _ = _render_test_helper(
        tmp_path,
        security_script=f"""#!/bin/bash
set -eu
account=""
service=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-a" ]]; then
        account="$2"
        shift 2
    elif [[ "$1" == "-s" ]]; then
        service="$2"
        shift 2
    else
        shift
    fi
done
case "$account" in
    OPENJARVIS_API_KEY|\
SKYNET_JARVIS_CASA_READ_TOKEN|\
SKYNET_JARVIS_AGENDA_READ_TOKEN|\
SKYNET_JARVIS_FROTA_READ_TOKEN)
        state="$HOME/.openjarvis/skynet-l99/shadow/state"
        /bin/mkdir -p "$state"
        printf '%s|%s\\n' "$service" "$account" >>"$state/keychain-accounts"
        if [[ "$account" == "SKYNET_JARVIS_FROTA_READ_TOKEN" ]]; then
            printf '\\n# post-validation-tamper\\n' >>\
"$HOME/.openjarvis/skynet-l99/shadow/config/skynet-l99-shadow.toml"
            printf '\\n ' >>\
"$HOME/.openjarvis/skynet-l99/shadow/config/skynet-l99-shadow-capabilities.json"
        fi
        printf '%s' '{fake_value}'
        ;;
    *)
        exit 44
        ;;
esac
""",
    )
    result = subprocess.run(
        _shadow_helper_command(helper),
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": str(home),
            "OPENAI_API_KEY": "inherited-provider-secret",
            "ARBITRARY_TOKEN": "inherited-arbitrary-secret",
            "SKYNET_JARVIS_CASA_ACTION_TOKEN": "inherited-action-secret",
            "SKYNET_JARVIS_API_BASE_URL": "https://attacker.invalid",
            "SKYNET_JARVIS_CASA_READ_DEVICE_ID": "attacker-device",
            "SKYNET_JARVIS_CASA_READ_CHANNEL": "attacker-channel",
        },
    )

    assert result.returncode == 0, result.stderr
    assert fake_value not in result.stdout
    assert fake_value not in result.stderr

    state = home / ".openjarvis" / "skynet-l99" / "shadow" / "state"
    names = set((state / "observed-env").read_text().splitlines())
    # macOS may inject this CoreFoundation locale variable when it starts the
    # shebang interpreter; it was not inherited from the launcher.
    names.discard("__CF_USER_TEXT_ENCODING")
    assert names == READ_TOKENS | {
        "HOME",
        "LANG",
        "LOGNAME",
        "OPENJARVIS_CONFIG",
        "OPENJARVIS_HOME",
        "OPENJARVIS_NO_UPDATE_CHECK",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TMPDIR",
        "USER",
        "SKYNET_JARVIS_API_BASE_URL",
        "SKYNET_JARVIS_CASA_READ_CHANNEL",
        "SKYNET_JARVIS_CASA_READ_DEVICE_ID",
        "SKYNET_JARVIS_AGENDA_READ_CHANNEL",
        "SKYNET_JARVIS_AGENDA_READ_DEVICE_ID",
        "SKYNET_JARVIS_FROTA_READ_CHANNEL",
        "SKYNET_JARVIS_FROTA_READ_DEVICE_ID",
        "OPENJARVIS_API_PRINCIPAL",
        "OPENJARVIS_API_PRINCIPAL_ALLOWLIST",
    }
    accounts = set((state / "keychain-accounts").read_text().splitlines())
    assert accounts == {
        "OpenJarvis Cloud Keys|OPENJARVIS_API_KEY",
        "OpenJarvis L99 Shadow Keys|SKYNET_JARVIS_CASA_READ_TOKEN",
        "OpenJarvis L99 Shadow Keys|SKYNET_JARVIS_AGENDA_READ_TOKEN",
        "OpenJarvis L99 Shadow Keys|SKYNET_JARVIS_FROTA_READ_TOKEN",
    }
    assert (state / "observed-bindings").read_text().splitlines() == [
        "https://skynet.britos.app",
        "air-openjarvis-l99-shadow",
        "openjarvis-local",
        "air-openjarvis-l99-shadow",
        "openjarvis-local",
        "air-openjarvis-l99-shadow",
        "openjarvis-local",
    ]
    assert 'policy_path = "/dev/fd/' in (state / "observed-config").read_text()
    assert "post-validation-tamper" not in (state / "observed-config").read_text()
    assert (
        "post-validation-tamper"
        in (
            home / ".openjarvis" / "skynet-l99" / "shadow" / "config" / CONFIG.name
        ).read_text()
    )
    assert '"agent_id": "orchestrator"' in (state / "observed-policy").read_text()

    argv = (state / "observed-argv").read_text().splitlines()
    assert argv == [
        "-S",
        "-P",
        "-m",
        "openjarvis.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--agent",
        "orchestrator",
    ]
    assert fake_value not in "\n".join(argv)
