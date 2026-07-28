#!/usr/bin/env python3
"""Fail-closed macOS launcher for the OpenJarvis L99 read-only shadow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

KEYCHAIN_SECURITY_BIN = "/usr/bin/security"
LOCAL_KEYCHAIN_SERVICE = "OpenJarvis Cloud Keys"
SHADOW_KEYCHAIN_SERVICE = "OpenJarvis L99 Shadow Keys"
RUNTIME_SITE = "__OPENJARVIS_RUNTIME_SITE__"
RUNTIME_RECEIPT = "__OPENJARVIS_RUNTIME_RECEIPT__"
RUNTIME_MANIFEST = "__OPENJARVIS_RUNTIME_MANIFEST__"
EXPECTED_RECEIPT_SHA256 = "__OPENJARVIS_RUNTIME_RECEIPT_SHA256__"
EXPECTED_MANIFEST_SHA256 = "__OPENJARVIS_RUNTIME_MANIFEST_SHA256__"
EXPECTED_RUNTIME_TREE_SHA256 = "__OPENJARVIS_RUNTIME_TREE_SHA256__"
EXPECTED_SOURCE_COMMIT = "__OPENJARVIS_SOURCE_COMMIT__"
EXPECTED_CONFIG_SHA256 = "__OPENJARVIS_SHADOW_CONFIG_SHA256__"
EXPECTED_POLICY_SHA256 = "__OPENJARVIS_SHADOW_POLICY_SHA256__"
SKYNET_API_BASE_URL = "https://skynet.britos.app"
DEVICE_ID = "air-openjarvis-l99-shadow"
API_PRINCIPAL = "api:l99-desktop"
AIR_HOME_PATTERN = r"^/Users/[A-Za-z0-9._-]+$"

SECRET_BINDINGS = (
    ("OPENJARVIS_API_KEY", LOCAL_KEYCHAIN_SERVICE),
    ("SKYNET_JARVIS_CASA_READ_TOKEN", SHADOW_KEYCHAIN_SERVICE),
    ("SKYNET_JARVIS_AGENDA_READ_TOKEN", SHADOW_KEYCHAIN_SERVICE),
    ("SKYNET_JARVIS_FROTA_READ_TOKEN", SHADOW_KEYCHAIN_SERVICE),
)
SECRET_ACCOUNTS = tuple(account for account, _service in SECRET_BINDINGS)


class _StartupDenied(RuntimeError):
    """Internal marker whose details are never rendered to logs."""


def _deny() -> None:
    os.write(2, b"openjarvis-shadow: startup denied by local safety gate\n")
    raise SystemExit(78)


def _read_owned_file_no_follow(
    path: Path,
    *,
    exact_mode: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise _StartupDenied
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _StartupDenied from exc
    try:
        before = os.fstat(descriptor)
        lexical = path.lstat()
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or (before.st_dev, before.st_ino) != (lexical.st_dev, lexical.st_ino)
            or (exact_mode is not None and mode != exact_mode)
            or (exact_mode is None and mode & 0o022 != 0)
        ):
            raise _StartupDenied

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise _StartupDenied
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final_lexical = path.lstat()
        metadata_changed = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_changed = (
            after.st_dev,
            after.st_ino,
        ) != (
            final_lexical.st_dev,
            final_lexical.st_ino,
        )
        if metadata_changed or path_changed or total != after.st_size:
            raise _StartupDenied
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _owned_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        not path.is_symlink()
        and path.resolve(strict=True) == path
        and stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o022 == 0
    )


def _validated_home(raw_home: Optional[str]) -> Path:
    if not raw_home or not re.fullmatch(AIR_HOME_PATTERN, raw_home):
        raise _StartupDenied
    home = Path(raw_home)
    if home.name in {"", ".", ".."} or home.resolve(strict=True) != home:
        raise _StartupDenied
    return home


def _validated_runtime_site(home: Path) -> Path:
    runtime_site = Path(RUNTIME_SITE)
    runtime_root = home / ".openjarvis" / "skynet-l99" / "runtimes"
    runtime_digest_root = runtime_site.parent
    package_root = runtime_site / "openjarvis"
    package_marker = runtime_site / "openjarvis" / "__init__.py"
    try:
        relative = runtime_site.relative_to(runtime_root)
    except ValueError as exc:
        raise _StartupDenied from exc
    if (
        len(relative.parts) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", relative.parts[0])
        or relative.parts[1] != "site-packages"
        or not _owned_directory(runtime_root)
        or not _owned_directory(runtime_digest_root)
        or not _owned_directory(runtime_site)
        or not _owned_directory(package_root)
        or runtime_site.resolve(strict=True) != runtime_site
        or package_marker.is_symlink()
        or not package_marker.is_file()
        or package_marker.stat().st_uid != os.getuid()
        or stat.S_IMODE(package_marker.stat().st_mode) & 0o022 != 0
    ):
        raise _StartupDenied
    return runtime_site


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _StartupDenied
        value[key] = item
    return value


def _load_pinned_json(
    path: Path,
    expected_sha256: str,
    *,
    private: bool,
) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise _StartupDenied
    payload = _read_owned_file_no_follow(
        path,
        exact_mode=0o600 if private else None,
        max_bytes=1024 * 1024,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise _StartupDenied
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _StartupDenied from exc
    if not isinstance(parsed, dict):
        raise _StartupDenied
    return parsed


def _tree_record(digest, *fields: bytes) -> None:
    for field in fields:
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)


def _runtime_tree_sha256(runtime_site: Path) -> str:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise _StartupDenied
    directory_flags = (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    digest = hashlib.sha256()
    digest.update(b"openjarvis.skynet-l99.runtime-tree.v1\0")
    counters = {"entries": 0, "bytes": 0}

    def validate_metadata(info: os.stat_result, *, directory: bool) -> None:
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_type(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o022 != 0
        ):
            raise _StartupDenied

    def unchanged(before: os.stat_result, after: os.stat_result) -> bool:
        return (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )

    def walk(directory_fd: int, relative: bytes, depth: int) -> None:
        if depth > 128:
            raise _StartupDenied
        before_directory = os.fstat(directory_fd)
        validate_metadata(before_directory, directory=True)
        names = sorted(os.listdir(directory_fd), key=os.fsencode)
        for name in names:
            counters["entries"] += 1
            if counters["entries"] > 200_000:
                raise _StartupDenied
            name_bytes = os.fsencode(name)
            relative_path = name_bytes if not relative else relative + b"/" + name_bytes
            lexical = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)

            if stat.S_ISDIR(lexical.st_mode):
                validate_metadata(lexical, directory=True)
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    validate_metadata(opened, directory=True)
                    if not unchanged(lexical, opened):
                        raise _StartupDenied
                    _tree_record(
                        digest,
                        b"D",
                        relative_path,
                        f"{stat.S_IMODE(opened.st_mode):04o}".encode(),
                    )
                    walk(child_fd, relative_path, depth + 1)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(lexical.st_mode):
                validate_metadata(lexical, directory=False)
                file_fd = os.open(name, file_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(file_fd)
                    validate_metadata(opened, directory=False)
                    if not unchanged(lexical, opened):
                        raise _StartupDenied
                    file_digest = hashlib.sha256()
                    file_size = 0
                    while True:
                        chunk = os.read(file_fd, 1024 * 1024)
                        if not chunk:
                            break
                        file_size += len(chunk)
                        counters["bytes"] += len(chunk)
                        if counters["bytes"] > 10 * 1024 * 1024 * 1024:
                            raise _StartupDenied
                        file_digest.update(chunk)
                    after = os.fstat(file_fd)
                    if not unchanged(opened, after) or file_size != opened.st_size:
                        raise _StartupDenied
                    _tree_record(
                        digest,
                        b"F",
                        relative_path,
                        f"{stat.S_IMODE(opened.st_mode):04o}".encode(),
                        str(file_size).encode(),
                        file_digest.hexdigest().encode(),
                    )
                finally:
                    os.close(file_fd)
            else:
                raise _StartupDenied
        if not unchanged(before_directory, os.fstat(directory_fd)):
            raise _StartupDenied

    root_fd = os.open(runtime_site, directory_flags)
    try:
        root_info = os.fstat(root_fd)
        root_lexical = runtime_site.lstat()
        validate_metadata(root_info, directory=True)
        if not unchanged(root_lexical, root_info):
            raise _StartupDenied
        _tree_record(
            digest,
            b"D",
            b"",
            f"{stat.S_IMODE(root_info.st_mode):04o}".encode(),
        )
        walk(root_fd, b"", 0)
    finally:
        os.close(root_fd)
    return digest.hexdigest()


def _validated_runtime_provenance(home: Path) -> Path:
    runtime_site = _validated_runtime_site(home)
    receipt = Path(RUNTIME_RECEIPT)
    manifest = Path(RUNTIME_MANIFEST)
    metadata_root = home / ".openjarvis" / "skynet-l99"
    receipt_root = metadata_root / "receipts"
    manifest_root = metadata_root / "manifests"

    if (
        not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_SOURCE_COMMIT)
        or not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_RUNTIME_TREE_SHA256)
        or receipt.parent != receipt_root
        or manifest.parent != manifest_root
        or not _owned_directory(metadata_root)
        or not _owned_directory(receipt_root)
        or not _owned_directory(manifest_root)
        or not re.fullmatch(
            r"install-receipt-air-[0-9]{8}T[0-9]{6}Z[.]json",
            receipt.name,
        )
        or manifest.name != f"{EXPECTED_SOURCE_COMMIT[:12]}.json"
    ):
        raise _StartupDenied

    receipt_data = _load_pinned_json(
        receipt,
        EXPECTED_RECEIPT_SHA256,
        private=True,
    )
    manifest_data = _load_pinned_json(
        manifest,
        EXPECTED_MANIFEST_SHA256,
        private=False,
    )
    wheel = manifest_data.get("wheel")
    runtime_archive = manifest_data.get("runtime_archive")
    activation = manifest_data.get("activation")
    if (
        not isinstance(wheel, dict)
        or not isinstance(runtime_archive, dict)
        or not isinstance(activation, dict)
    ):
        raise _StartupDenied

    runtime_digest = runtime_site.parent.name
    archive_sha256 = runtime_archive.get("sha256")
    manifest_strings = (
        manifest_data.get("runtime_version"),
        manifest_data.get("platform"),
        manifest_data.get("python_abi"),
        wheel.get("file"),
        runtime_archive.get("file"),
    )
    if (
        receipt_data.get("schema") != "openjarvis.skynet-l99.install.v1"
        or manifest_data.get("schema") != "openjarvis.skynet-l99.runtime-manifest.v1"
        or receipt_data.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or manifest_data.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or receipt_data.get("runtime_site") != str(runtime_site)
        or activation.get("runtime_site") != str(runtime_site)
        or receipt_data.get("wheel_sha256") != runtime_digest
        or wheel.get("sha256") != runtime_digest
        or receipt_data.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or not isinstance(archive_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", archive_sha256)
        or receipt_data.get("runtime_archive_sha256") != archive_sha256
        or not all(isinstance(value, str) and bool(value) for value in manifest_strings)
        or "openjarvis_rust" not in manifest_data
        or not isinstance(manifest_data.get("dependency_wheels"), list)
        or not isinstance(manifest_data.get("profile_files"), dict)
        or not isinstance(wheel.get("bytes"), int)
        or wheel.get("bytes", 0) <= 0
        or not isinstance(runtime_archive.get("bytes"), int)
        or runtime_archive.get("bytes", 0) <= 0
        or _runtime_tree_sha256(runtime_site) != EXPECTED_RUNTIME_TREE_SHA256
    ):
        raise _StartupDenied
    return runtime_site


def _keychain_environment(home: Path) -> dict[str, str]:
    username = home.name
    return {
        "HOME": str(home),
        "USER": username,
        "LOGNAME": username,
        "LANG": "en_US.UTF-8",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/tmp",
    }


def _read_keychain_secret(account: str, service: str, home: Path) -> str:
    if (account, service) not in SECRET_BINDINGS:
        raise _StartupDenied
    try:
        result = subprocess.run(
            [
                KEYCHAIN_SECURITY_BIN,
                "find-generic-password",
                "-s",
                service,
                "-a",
                account,
                "-w",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            shell=False,
            close_fds=True,
            env=_keychain_environment(home),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _StartupDenied from exc

    value = result.stdout.rstrip("\r\n")
    if (
        result.returncode != 0
        or not 32 <= len(value) <= 512
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise _StartupDenied
    return value


def _validated_shadow_inputs(home: Path):
    shadow_root = home / ".openjarvis" / "skynet-l99" / "shadow"
    config_dir = shadow_root / "config"
    config_path = config_dir / "skynet-l99-shadow.toml"
    policy_path = config_path.with_name("skynet-l99-shadow-capabilities.json")
    if not _owned_directory(shadow_root) or not _owned_directory(config_dir):
        raise _StartupDenied
    config_payload = _read_owned_file_no_follow(
        config_path,
        exact_mode=0o600,
        max_bytes=1024 * 1024,
    )
    policy_payload = _read_owned_file_no_follow(
        policy_path,
        exact_mode=0o600,
        max_bytes=1024 * 1024,
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_CONFIG_SHA256)
        or not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_POLICY_SHA256)
        or hashlib.sha256(config_payload).hexdigest() != EXPECTED_CONFIG_SHA256
        or hashlib.sha256(policy_payload).hexdigest() != EXPECTED_POLICY_SHA256
    ):
        raise _StartupDenied
    return config_payload, policy_payload


def _sealed_shadow_inputs(home: Path, state_dir: Path):
    config_payload, policy_payload = _validated_shadow_inputs(home)
    policy_handle = tempfile.TemporaryFile(mode="w+b", dir=state_dir)
    try:
        policy_handle.write(policy_payload)
        policy_handle.flush()
        os.fsync(policy_handle.fileno())
        policy_handle.seek(0)
        os.set_inheritable(policy_handle.fileno(), True)

        configured_policy = (
            b"policy_path = "
            b'"~/.openjarvis/skynet-l99/shadow/config/'
            b'skynet-l99-shadow-capabilities.json"'
        )
        fd_policy = (
            'policy_path = "/dev/fd/{}"'.format(policy_handle.fileno())
        ).encode("ascii")
        if config_payload.count(configured_policy) != 1:
            raise _StartupDenied
        launch_config = config_payload.replace(configured_policy, fd_policy)

        config_handle = tempfile.TemporaryFile(mode="w+b", dir=state_dir)
        try:
            config_handle.write(launch_config)
            config_handle.flush()
            os.fsync(config_handle.fileno())
            config_handle.seek(0)
            os.set_inheritable(config_handle.fileno(), True)
            return config_handle, policy_handle
        except Exception:
            config_handle.close()
            raise
    except Exception:
        policy_handle.close()
        raise


def _runtime_environment(
    home: Path,
    runtime_site: Path,
    secrets: dict[str, str],
    config_descriptor: int,
) -> dict[str, str]:
    if set(secrets) != set(SECRET_ACCOUNTS):
        raise _StartupDenied
    environment = _keychain_environment(home)
    environment.update(
        {
            "OPENJARVIS_NO_UPDATE_CHECK": "1",
            "OPENJARVIS_HOME": str(
                home / ".openjarvis" / "skynet-l99" / "shadow" / "state"
            ),
            "OPENJARVIS_CONFIG": "/dev/fd/{}".format(config_descriptor),
            "OPENJARVIS_API_PRINCIPAL": API_PRINCIPAL,
            "OPENJARVIS_API_PRINCIPAL_ALLOWLIST": API_PRINCIPAL,
            "PYTHONPATH": str(runtime_site),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SKYNET_JARVIS_API_BASE_URL": SKYNET_API_BASE_URL,
            "SKYNET_JARVIS_CASA_READ_DEVICE_ID": DEVICE_ID,
            "SKYNET_JARVIS_CASA_READ_CHANNEL": "openjarvis-local",
            "SKYNET_JARVIS_AGENDA_READ_DEVICE_ID": DEVICE_ID,
            "SKYNET_JARVIS_AGENDA_READ_CHANNEL": "openjarvis-local",
            "SKYNET_JARVIS_FROTA_READ_DEVICE_ID": DEVICE_ID,
            "SKYNET_JARVIS_FROTA_READ_CHANNEL": "openjarvis-local",
        }
    )
    environment.update(secrets)
    return environment


def _launch(home: Path) -> None:
    runtime_site = _validated_runtime_provenance(home)
    python_bin = home / ".openjarvis" / ".venv" / "bin" / "python"
    state_dir = home / ".openjarvis" / "skynet-l99" / "shadow" / "state"

    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise _StartupDenied
    if (
        state_dir.is_symlink()
        or state_dir.resolve(strict=True) != state_dir
        or not state_dir.is_dir()
        or state_dir.stat().st_uid != os.getuid()
        or stat.S_IMODE(state_dir.stat().st_mode) != 0o700
    ):
        raise _StartupDenied

    config_handle, _policy_handle = _sealed_shadow_inputs(home, state_dir)
    secrets = {
        account: _read_keychain_secret(account, service, home)
        for account, service in SECRET_BINDINGS
    }
    environment = _runtime_environment(
        home,
        runtime_site,
        secrets,
        config_handle.fileno(),
    )
    argv = [
        str(python_bin),
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
    os.execve(str(python_bin), argv, environment)


def main() -> None:
    raw_home = os.environ.get("HOME")
    arguments = tuple(sys.argv[1:])
    os.environ.clear()
    try:
        home = _validated_home(raw_home)
        if arguments == ("--validate-provenance-only",):
            _validated_runtime_provenance(home)
            return
        if arguments == ("--validate-installed-inputs-only",):
            _validated_runtime_provenance(home)
            _validated_shadow_inputs(home)
            return
        if arguments:
            raise _StartupDenied
        _launch(home)
    except (OSError, TypeError, ValueError, _StartupDenied):
        _deny()


if __name__ == "__main__":
    main()
