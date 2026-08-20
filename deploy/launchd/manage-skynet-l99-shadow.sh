#!/bin/bash
set -Eeuo pipefail
set +x
umask 077

readonly LABEL="com.openjarvis.skynet-l99-shadow"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
readonly CURRENT_UID="$(id -u)"
readonly -a VERSIONED_SOURCE_PATHS=(
    "frontend/src-tauri/src/lib.rs"
    "deploy/config/skynet-l99-shadow.toml"
    "deploy/config/skynet-l99-shadow-capabilities.json"
    "deploy/launchd/com.openjarvis.skynet-l99-shadow.plist"
    "deploy/launchd/manage-skynet-l99-shadow.sh"
    "deploy/launchd/openjarvis-skynet-l99-shadow.py"
    "tests/deploy/test_launchd_shadow.py"
)

die() {
    printf '%s\n' "manage-skynet-l99-shadow: $*" >&2
    if [[ -n "${INSTALL_TRANSACTION_RECEIPT:-}" ]] &&
        [[ "${INSTALL_ROLLBACK_ACTIVE:-0}" == "0" ]] &&
        declare -F install_error_handler >/dev/null; then
        install_error_handler 1
    fi
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  manage-skynet-l99-shadow.sh install <runtime-site> <runtime-receipt> <runtime-manifest>
  manage-skynet-l99-shadow.sh validate
  manage-skynet-l99-shadow.sh rollback

`install` only stages a dormant LaunchAgent. It never loads or starts it.
Runtime provenance is mandatory; discovery by directory name is forbidden.
EOF
}

path_exists() {
    [[ -e "$1" || -L "$1" ]]
}

private_mode() {
    /usr/bin/stat -f '%Lp' "$1"
}

owner_uid() {
    /usr/bin/stat -f '%u' "$1"
}

validate_owned_directory() {
    local path="$1"
    local physical mode permission

    [[ -d "$path" && ! -L "$path" ]] ||
        die "unsafe directory in managed path"
    physical="$(cd "$path" && pwd -P)"
    [[ "$physical" == "$path" ]] ||
        die "managed directory must not traverse symlinks"
    [[ "$(owner_uid "$path")" == "$CURRENT_UID" ]] ||
        die "managed directory owner mismatch"
    mode="$(private_mode "$path")"
    [[ "$mode" =~ ^[0-7]{3,4}$ ]] ||
        die "could not validate managed directory mode"
    permission=$((8#$mode))
    (( (permission & 0022) == 0 )) ||
        die "managed directory is group/world writable"
}

validate_private_directory() {
    validate_owned_directory "$1"
    [[ "$(private_mode "$1")" == "700" ]] ||
        die "private managed directory must be mode 0700"
}

ensure_owned_directory() {
    local path="$1"
    local parent

    if path_exists "$path"; then
        validate_owned_directory "$path"
        return
    fi
    parent="$(dirname "$path")"
    validate_owned_directory "$parent"
    /bin/mkdir "$path"
    /bin/chmod 700 "$path"
    validate_owned_directory "$path"
}

ensure_private_directory() {
    ensure_owned_directory "$1"
    validate_private_directory "$1"
}

validate_owned_regular_file() {
    local path="$1"
    local expected_mode="${2:-}"
    local mode permission

    [[ -f "$path" && ! -L "$path" ]] ||
        die "unsafe regular file in managed path"
    validate_owned_directory "$(dirname "$path")"
    [[ "$(owner_uid "$path")" == "$CURRENT_UID" ]] ||
        die "managed file owner mismatch"
    mode="$(private_mode "$path")"
    [[ "$mode" =~ ^[0-7]{3,4}$ ]] ||
        die "could not validate managed file mode"
    permission=$((8#$mode))
    (( (permission & 0022) == 0 )) ||
        die "managed file is group/world writable"
    if [[ -n "$expected_mode" && "$mode" != "$expected_mode" ]]; then
        die "managed file has an unexpected private mode"
    fi
}

validate_optional_private_file() {
    local path="$1"
    if path_exists "$path"; then
        validate_owned_regular_file "$path" "600"
    fi
}

validate_optional_helper_file() {
    if path_exists "$helper_path"; then
        validate_owned_regular_file "$helper_path" "700"
    fi
}

require_air_compatible_home() {
    [[ "$(uname -s)" == "Darwin" ]] || die "macOS is required"
    [[ "$CURRENT_UID" -ne 0 ]] || die "run as the login user, never as root"
    [[ "$HOME" =~ ^/Users/[A-Za-z0-9._-]+$ ]] || die "unsupported HOME path"
    [[ "${HOME##*/}" != "." && "${HOME##*/}" != ".." ]] ||
        die "unsafe HOME path"
    validate_owned_directory "$HOME"
}

shadow_root="$HOME/.openjarvis/skynet-l99/shadow"
config_dir="$shadow_root/config"
bin_dir="$shadow_root/bin"
log_dir="$shadow_root/logs"
state_dir="$shadow_root/state"
rollback_root="$HOME/.openjarvis/skynet-l99/rollbacks"
transaction_root="$HOME/.openjarvis/skynet-l99/transactions"
runtime_root="$HOME/.openjarvis/skynet-l99/runtimes"
runtime_receipt_root="$HOME/.openjarvis/skynet-l99/receipts"
runtime_manifest_root="$HOME/.openjarvis/skynet-l99/manifests"
helper_path="$bin_dir/openjarvis-skynet-l99-shadow.py"
python_bin="$HOME/.openjarvis/.venv/bin/python"
bootstrap_python="/usr/bin/python3"
plist_path="$HOME/Library/LaunchAgents/$LABEL.plist"
stdout_path="$log_dir/shadow.stdout.log"
stderr_path="$log_dir/shadow.stderr.log"
shadow_receipt_path="$state_dir/install-receipt.json"
PREINSTALL_BACKUP_DIR=""
PREINSTALL_MANIFEST_SHA256=""
SOURCE_SNAPSHOT_MANIFEST=""
SOURCE_SNAPSHOT_MANIFEST_SHA256=""
SNAPSHOT_CONFIG=""
SNAPSHOT_POLICY=""
SNAPSHOT_HELPER=""
SNAPSHOT_PLIST=""
INSTALL_TRANSACTION_DIR=""
INSTALL_TRANSACTION_RECEIPT=""
INSTALL_ROLLBACK_ACTIVE="0"

safe_git() {
    /usr/bin/env -i \
        HOME="$HOME" \
        LANG="en_US.UTF-8" \
        LC_ALL="C" \
        PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
        GIT_CONFIG_GLOBAL="/dev/null" \
        GIT_CONFIG_NOSYSTEM="1" \
        GIT_NO_REPLACE_OBJECTS="1" \
        GIT_OPTIONAL_LOCKS="0" \
        GIT_TERMINAL_PROMPT="0" \
        /usr/bin/git \
        -c core.fsmonitor=false \
        -C "$REPO_ROOT" \
        "$@"
}

safe_python() {
    /usr/bin/env -i \
        HOME="$HOME" \
        USER="${HOME##*/}" \
        LOGNAME="${HOME##*/}" \
        LANG="en_US.UTF-8" \
        PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
        PYTHONDONTWRITEBYTECODE="1" \
        PYTHONNOUSERSITE="1" \
        "$bootstrap_python" -I -S "$@"
}

sha256_file() {
    local digest
    digest="$(/usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}')"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
        die "could not calculate a SHA-256 digest"
    printf '%s\n' "$digest"
}

repo_source_commit() {
    local commit
    commit="$(safe_git rev-parse --verify HEAD^{commit})"
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] ||
        die "could not resolve the source commit"
    printf '%s\n' "$commit"
}

validate_runtime_inputs() {
    local runtime_site="$1"
    local runtime_receipt="$2"
    local runtime_manifest="$3"
    local prefix relative digest receipt_name

    validate_owned_directory "$HOME/.openjarvis"
    validate_owned_directory "$HOME/.openjarvis/skynet-l99"
    validate_owned_directory "$runtime_root"
    validate_owned_directory "$runtime_receipt_root"
    validate_owned_directory "$runtime_manifest_root"

    prefix="$runtime_root/"
    [[ "$runtime_site" == "$prefix"*"/site-packages" ]] ||
        die "runtime must be inside the managed L99 runtime root"
    relative="${runtime_site#"$prefix"}"
    digest="${relative%%/*}"
    [[ "$relative" == "$digest/site-packages" ]] ||
        die "runtime path has unexpected nesting"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
        die "runtime directory must be a lowercase SHA-256"
    validate_owned_directory "$runtime_root/$digest"
    validate_owned_directory "$runtime_site"
    validate_owned_directory "$runtime_site/openjarvis"
    validate_owned_regular_file "$runtime_site/openjarvis/__init__.py"

    [[ "$(dirname "$runtime_receipt")" == "$runtime_receipt_root" ]] ||
        die "runtime receipt must use the canonical receipt directory"
    receipt_name="$(basename "$runtime_receipt")"
    [[ "$receipt_name" =~ ^install-receipt-air-[0-9]{8}T[0-9]{6}Z[.]json$ ]] ||
        die "runtime receipt filename is invalid"
    validate_owned_regular_file "$runtime_receipt" "600"

    [[ "$(dirname "$runtime_manifest")" == "$runtime_manifest_root" ]] ||
        die "runtime manifest must use the canonical manifest directory"
    [[ "$(basename "$runtime_manifest")" =~ ^[0-9a-f]{12}[.]json$ ]] ||
        die "runtime manifest filename is invalid"
    validate_owned_regular_file "$runtime_manifest"

    [[ -x "$python_bin" ]] ||
        die "managed OpenJarvis Python is unavailable"
    [[ -x "$bootstrap_python" ]] ||
        die "immutable system Python bootstrap is unavailable"
}

snapshot_versioned_sources() {
    local expected_commit="$1"
    local snapshot_root="$2"

    [[ ${#VERSIONED_SOURCE_PATHS[@]} -eq 7 ]] ||
        die "versioned source allowlist must contain exactly seven paths"
    /bin/mkdir "$snapshot_root"
    /bin/chmod 700 "$snapshot_root"
    validate_private_directory "$snapshot_root"

    safe_python - \
        "$REPO_ROOT" \
        "$expected_commit" \
        "$snapshot_root" \
        "${VERSIONED_SOURCE_PATHS[@]}" <<'PY'
# source-snapshot-builder-v1
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

try:
    repo = Path(sys.argv[1])
    commit = sys.argv[2]
    snapshot = Path(sys.argv[3])
    relative_paths = sys.argv[4:]
    if (
        len(relative_paths) != 7
        or len(set(relative_paths)) != 7
        or not all(
            path
            and not path.startswith("/")
            and ".." not in Path(path).parts
            for path in relative_paths
        )
    ):
        raise ValueError

    git_environment = {
        "HOME": str(Path.home()),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def git(*arguments):
        return subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(repo),
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=git_environment,
        ).stdout

    def read_owned(path, expected_mode):
        if path.resolve(strict=True) != path:
            raise ValueError
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            lexical = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) & 0o022
                or (before.st_dev, before.st_ino)
                != (lexical.st_dev, lexical.st_ino)
                or (
                    "100755"
                    if stat.S_IMODE(before.st_mode) & 0o111
                    else "100644"
                )
                != expected_mode
            ):
                raise ValueError
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
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
            ):
                raise ValueError
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    if git("rev-parse", "--verify", "HEAD^{commit}").decode().strip() != commit:
        raise ValueError

    entries = []
    validated = {}
    for relative in relative_paths:
        tree = git("ls-tree", "-z", commit, "--", relative)
        match = tree.rstrip(b"\0").split(b"\t", 1)
        if len(match) != 2 or match[1].decode() != relative:
            raise ValueError
        metadata = match[0].decode().split()
        if (
            len(metadata) != 3
            or metadata[0] not in {"100644", "100755"}
            or metadata[1] != "blob"
            or len(metadata[2]) != 40
        ):
            raise ValueError
        mode, _kind, blob_sha1 = metadata
        blob = git("cat-file", "blob", "{}:{}".format(commit, relative))
        git_object = b"blob " + str(len(blob)).encode() + b"\0" + blob
        if hashlib.sha1(git_object).hexdigest() != blob_sha1:
            raise ValueError
        worktree = repo / relative
        if read_owned(worktree, mode) != blob:
            raise ValueError

        destination = snapshot / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o700 if mode == "100755" else 0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.chmod(destination, 0o500 if mode == "100755" else 0o400)
        validated[relative] = (mode, blob)
        entries.append(
            {
                "path": relative,
                "mode": mode,
                "git_blob_sha1": blob_sha1,
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )

    if git("rev-parse", "--verify", "HEAD^{commit}").decode().strip() != commit:
        raise ValueError
    for relative, (mode, blob) in validated.items():
        if read_owned(repo / relative, mode) != blob:
            raise ValueError

    manifest = snapshot / "source-snapshot-manifest.json"
    payload = json.dumps(
        {
            "schema": "openjarvis.skynet-l99.source-snapshot.v1",
            "source_commit": commit,
            "entries": entries,
        },
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    descriptor = os.open(
        manifest,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(manifest, 0o400)
    for directory in sorted(
        (path for path in snapshot.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        os.chmod(directory, 0o500)
    os.chmod(snapshot, 0o500)
except BaseException:
    raise SystemExit(78)
PY

    SOURCE_SNAPSHOT_MANIFEST="$snapshot_root/source-snapshot-manifest.json"
    SNAPSHOT_CONFIG="$snapshot_root/deploy/config/skynet-l99-shadow.toml"
    SNAPSHOT_POLICY="$snapshot_root/deploy/config/skynet-l99-shadow-capabilities.json"
    SNAPSHOT_HELPER="$snapshot_root/deploy/launchd/openjarvis-skynet-l99-shadow.py"
    SNAPSHOT_PLIST="$snapshot_root/deploy/launchd/$LABEL.plist"
    validate_owned_regular_file "$SOURCE_SNAPSHOT_MANIFEST" "400"
    validate_owned_regular_file "$SNAPSHOT_CONFIG" "400"
    validate_owned_regular_file "$SNAPSHOT_POLICY" "400"
    validate_owned_regular_file "$SNAPSHOT_HELPER"
    [[ "$(private_mode "$SNAPSHOT_HELPER")" =~ ^(400|500)$ ]] ||
        die "snapshot helper has an unexpected immutable mode"
    validate_owned_regular_file "$SNAPSHOT_PLIST" "400"
    SOURCE_SNAPSHOT_MANIFEST_SHA256="$(sha256_file "$SOURCE_SNAPSHOT_MANIFEST")"
}

runtime_tree_sha256() {
    local helper_source="$1"
    local runtime_site="$2"
    local digest

    if ! digest="$(
        safe_python - "$helper_source" "$runtime_site" <<'PY'
import importlib.util
import sys
from pathlib import Path

try:
    helper_source, runtime_site = sys.argv[1:]
    spec = importlib.util.spec_from_file_location(
        "openjarvis_shadow_tree",
        helper_source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(module._runtime_tree_sha256(Path(runtime_site)))
except BaseException:
    raise SystemExit(78)
PY
    )"; then
        die "runtime tree validation failed"
    fi
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
        die "runtime tree digest is invalid"
    printf '%s\n' "$digest"
}

validate_runtime_provenance_contract() {
    local helper_source="$1"
    local runtime_site="$2"
    local runtime_receipt="$3"
    local runtime_manifest="$4"
    local receipt_sha256="$5"
    local manifest_sha256="$6"
    local runtime_tree_sha256="$7"
    local source_commit="$8"

    if ! safe_python - \
        "$helper_source" \
        "$HOME" \
        "$runtime_site" \
        "$runtime_receipt" \
        "$runtime_manifest" \
        "$receipt_sha256" \
        "$manifest_sha256" \
        "$runtime_tree_sha256" \
        "$source_commit" <<'PY'
import importlib.util
import sys
from pathlib import Path

try:
    (
        helper_source,
        raw_home,
        runtime_site,
        runtime_receipt,
        runtime_manifest,
        receipt_sha256,
        manifest_sha256,
        runtime_tree_sha256,
        source_commit,
    ) = sys.argv[1:]
    spec = importlib.util.spec_from_file_location(
        "openjarvis_shadow_provenance",
        helper_source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUNTIME_SITE = runtime_site
    module.RUNTIME_RECEIPT = runtime_receipt
    module.RUNTIME_MANIFEST = runtime_manifest
    module.EXPECTED_RECEIPT_SHA256 = receipt_sha256
    module.EXPECTED_MANIFEST_SHA256 = manifest_sha256
    module.EXPECTED_RUNTIME_TREE_SHA256 = runtime_tree_sha256
    module.EXPECTED_SOURCE_COMMIT = source_commit
    module._validated_runtime_provenance(Path(raw_home))
except BaseException:
    raise SystemExit(78)
PY
    then
        die "runtime receipt/manifest validation failed"
    fi
}

ensure_install_directories() {
    ensure_private_directory "$shadow_root"
    ensure_private_directory "$config_dir"
    ensure_private_directory "$bin_dir"
    ensure_private_directory "$log_dir"
    ensure_private_directory "$state_dir"
    ensure_owned_directory "$HOME/Library"
    ensure_owned_directory "$HOME/Library/LaunchAgents"
}

ensure_management_directories() {
    ensure_private_directory "$rollback_root"
    ensure_private_directory "$transaction_root"
}

validate_output_targets() {
    validate_optional_private_file "$config_dir/skynet-l99-shadow.toml"
    validate_optional_private_file \
        "$config_dir/skynet-l99-shadow-capabilities.json"
    validate_optional_helper_file
    validate_optional_private_file "$stdout_path"
    validate_optional_private_file "$stderr_path"
    validate_optional_private_file "$plist_path"
    validate_optional_private_file "$shadow_receipt_path"
}

render_helper() {
    local runtime_site="$1"
    local runtime_receipt="$2"
    local runtime_manifest="$3"
    local receipt_sha256="$4"
    local manifest_sha256="$5"
    local runtime_tree_sha256="$6"
    local source_commit="$7"
    local config_sha256="$8"
    local policy_sha256="$9"
    local helper_source="${10}"
    local output="${11}"

    /usr/bin/sed \
        -e "s|__OPENJARVIS_RUNTIME_SITE__|$runtime_site|g" \
        -e "s|__OPENJARVIS_RUNTIME_RECEIPT__|$runtime_receipt|g" \
        -e "s|__OPENJARVIS_RUNTIME_MANIFEST__|$runtime_manifest|g" \
        -e "s|__OPENJARVIS_RUNTIME_RECEIPT_SHA256__|$receipt_sha256|g" \
        -e "s|__OPENJARVIS_RUNTIME_MANIFEST_SHA256__|$manifest_sha256|g" \
        -e "s|__OPENJARVIS_RUNTIME_TREE_SHA256__|$runtime_tree_sha256|g" \
        -e "s|__OPENJARVIS_SOURCE_COMMIT__|$source_commit|g" \
        -e "s|__OPENJARVIS_SHADOW_CONFIG_SHA256__|$config_sha256|g" \
        -e "s|__OPENJARVIS_SHADOW_POLICY_SHA256__|$policy_sha256|g" \
        "$helper_source" >"$output"
}

render_plist() {
    local plist_source="$1"
    local output="$2"
    /usr/bin/sed \
        -e "s|__OPENJARVIS_SHADOW_HELPER__|$helper_path|g" \
        -e "s|__OPENJARVIS_SHADOW_ROOT__|$shadow_root|g" \
        -e "s|__OPENJARVIS_SHADOW_STDOUT__|$stdout_path|g" \
        -e "s|__OPENJARVIS_SHADOW_STDERR__|$stderr_path|g" \
        "$plist_source" >"$output"
}

backup_owned_file_no_follow() {
    local source="$1"
    local destination="$2"
    local mode="$3"
    safe_python - "$source" "$destination" "$mode" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

try:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    expected_mode = int(sys.argv[3], 8)
    descriptor = os.open(
        source,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        lexical = source.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_uid != os.getuid()
            or (before.st_dev, before.st_ino)
            != (lexical.st_dev, lexical.st_ino)
        ):
            raise ValueError
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
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
        ):
            raise ValueError
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)

    output = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        expected_mode,
    )
    with os.fdopen(output, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, expected_mode)
    print(hashlib.sha256(payload).hexdigest())
except BaseException:
    raise SystemExit(78)
PY
}

backup_existing() {
    local stamp backup manifest index target backup_name digest mode directory
    local -a labels targets backup_names modes manifest_args
    local -a directory_labels directory_paths directory_args

    labels=("plist" "helper" "config" "policy" "stdout" "stderr" "receipt")
    targets=(
        "$plist_path"
        "$helper_path"
        "$config_dir/skynet-l99-shadow.toml"
        "$config_dir/skynet-l99-shadow-capabilities.json"
        "$stdout_path"
        "$stderr_path"
        "$shadow_receipt_path"
    )
    backup_names=(
        "plist"
        "helper.py"
        "config.toml"
        "policy.json"
        "stdout.log"
        "stderr.log"
        "receipt.json"
    )
    modes=("600" "700" "600" "600" "600" "600" "600")
    directory_labels=(
        "launch_agents"
        "shadow"
        "config"
        "bin"
        "logs"
        "state"
    )
    directory_paths=(
        "$HOME/Library/LaunchAgents"
        "$shadow_root"
        "$config_dir"
        "$bin_dir"
        "$log_dir"
        "$state_dir"
    )
    stamp="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
    backup="$(/usr/bin/mktemp -d "$rollback_root/preinstall-$stamp.XXXXXX")"
    /bin/chmod 700 "$backup"
    validate_private_directory "$backup"

    for ((index = 0; index < ${#labels[@]}; index++)); do
        target="${targets[$index]}"
        backup_name="${backup_names[$index]}"
        mode="${modes[$index]}"
        if path_exists "$target"; then
            validate_owned_regular_file "$target" "$mode"
            digest="$(
                backup_owned_file_no_follow \
                    "$target" \
                    "$backup/$backup_name" \
                    "$mode"
            )"
            validate_owned_regular_file "$backup/$backup_name" "$mode"
            [[ "$(sha256_file "$backup/$backup_name")" == "$digest" ]] ||
                die "preinstall backup hash mismatch"
            manifest_args+=(
                "${labels[$index]}"
                "$target"
                "$backup_name"
                "true"
                "$digest"
                "$mode"
            )
        else
            manifest_args+=(
                "${labels[$index]}"
                "$target"
                ""
                "false"
                ""
                "$mode"
            )
        fi
    done

    for ((index = 0; index < ${#directory_labels[@]}; index++)); do
        directory="${directory_paths[$index]}"
        if path_exists "$directory"; then
            validate_owned_directory "$directory"
            directory_args+=(
                "${directory_labels[$index]}"
                "$directory"
                "true"
                "$(private_mode "$directory")"
            )
        else
            directory_args+=(
                "${directory_labels[$index]}"
                "$directory"
                "false"
                ""
            )
        fi
    done

    manifest="$backup/backup-manifest.json"
    safe_python - \
        "$manifest" \
        "$backup" \
        "${manifest_args[@]}" \
        "--directories--" \
        "${directory_args[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
backup = Path(sys.argv[2])
values = sys.argv[3:]
separator = values.index("--directories--")
file_values = values[:separator]
directory_values = values[separator + 1:]
if len(file_values) != 7 * 6 or len(directory_values) != 6 * 4:
    raise SystemExit(78)
files = []
for offset in range(0, len(file_values), 6):
    label, target, backup_name, present, sha256, mode = file_values[
        offset:offset + 6
    ]
    files.append(
        {
            "label": label,
            "target": target,
            "backup_name": backup_name or None,
            "present": present == "true",
            "sha256": sha256 or None,
            "mode": mode,
        }
    )
directories = []
for offset in range(0, len(directory_values), 4):
    label, path, present, mode = directory_values[offset:offset + 4]
    directories.append(
        {
            "label": label,
            "path": path,
            "present": present == "true",
            "mode": mode or None,
        }
    )
payload = {
    "schema": "openjarvis.skynet-l99.shadow-preinstall.v2",
    "backup": str(backup),
    "files": files,
    "directories": directories,
}
descriptor = os.open(
    manifest,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    /bin/chmod 600 "$manifest"
    validate_owned_regular_file "$manifest" "600"
    PREINSTALL_BACKUP_DIR="$backup"
    PREINSTALL_MANIFEST_SHA256="$(sha256_file "$manifest")"
}

prepare_private_log() {
    local path="$1"
    if path_exists "$path"; then
        validate_owned_regular_file "$path" "600"
    else
        /usr/bin/install -m 0600 /dev/null "$path"
    fi
}

validate_shadow_receipt() {
    if ! safe_python - \
        "$shadow_receipt_path" \
        "$config_dir/skynet-l99-shadow.toml" \
        "$config_dir/skynet-l99-shadow-capabilities.json" \
        "$helper_path" \
        "$plist_path" \
        "$rollback_root" <<'PY'
import ast
import hashlib
import json
import sys
from pathlib import Path

try:
    receipt_path = Path(sys.argv[1])
    artifact_paths = list(map(Path, sys.argv[2:6]))
    rollback_root = Path(sys.argv[6])
    pairs_seen = set()

    def unique_object(pairs):
        pairs_seen.clear()
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    data = json.loads(
        receipt_path.read_text(),
        object_pairs_hook=unique_object,
    )
    if (
        data.get("schema") != "openjarvis.skynet-l99.shadow-launcher.v3"
        or data.get("activation") != "dormant"
        or data.get("runtime_integrity")
        != "receipt-manifest-and-tree-pinned"
    ):
        raise ValueError
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError
    source_snapshot = data.get("source_snapshot")
    if (
        not isinstance(source_snapshot, dict)
        or source_snapshot.get("schema")
        != "openjarvis.skynet-l99.source-snapshot.v1"
        or not isinstance(source_snapshot.get("entries"), list)
        or len(source_snapshot["entries"]) != 7
    ):
        raise ValueError
    reconstructed_snapshot = json.dumps(
        {
            "schema": source_snapshot["schema"],
            "source_commit": data.get("source_commit"),
            "entries": source_snapshot["entries"],
        },
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    if (
        hashlib.sha256(reconstructed_snapshot).hexdigest()
        != source_snapshot.get("manifest_sha256")
    ):
        raise ValueError
    snapshot_entries = {
        item.get("path"): item
        for item in source_snapshot["entries"]
        if isinstance(item, dict)
    }
    if set(snapshot_entries) != {
        "frontend/src-tauri/src/lib.rs",
        "deploy/config/skynet-l99-shadow.toml",
        "deploy/config/skynet-l99-shadow-capabilities.json",
        "deploy/launchd/com.openjarvis.skynet-l99-shadow.plist",
        "deploy/launchd/manage-skynet-l99-shadow.sh",
        "deploy/launchd/openjarvis-skynet-l99-shadow.py",
        "tests/deploy/test_launchd_shadow.py",
    }:
        raise ValueError
    for label, path in zip(
        ("config", "policy", "helper", "plist"),
        artifact_paths,
    ):
        item = artifacts.get(label)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            not isinstance(item, dict)
            or item.get("path") != str(path)
            or item.get("sha256") != digest
        ):
            raise ValueError
    helper_path = artifact_paths[2]
    constants = {}
    for node in ast.parse(helper_path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            constants[node.targets[0].id] = node.value.value
    runtime = data.get("runtime")
    expected_runtime = {
        "site": constants.get("RUNTIME_SITE"),
        "wheel_sha256": Path(constants["RUNTIME_SITE"]).parent.name,
        "tree_sha256": constants.get("EXPECTED_RUNTIME_TREE_SHA256"),
        "receipt": constants.get("RUNTIME_RECEIPT"),
        "receipt_sha256": constants.get("EXPECTED_RECEIPT_SHA256"),
        "manifest": constants.get("RUNTIME_MANIFEST"),
        "manifest_sha256": constants.get("EXPECTED_MANIFEST_SHA256"),
    }
    if (
        runtime != expected_runtime
        or data.get("source_commit")
        != constants.get("EXPECTED_SOURCE_COMMIT")
        or artifacts["config"]["sha256"]
        != constants.get("EXPECTED_CONFIG_SHA256")
        or artifacts["policy"]["sha256"]
        != constants.get("EXPECTED_POLICY_SHA256")
        or artifacts["config"]["sha256"]
        != snapshot_entries[
            "deploy/config/skynet-l99-shadow.toml"
        ].get("sha256")
        or artifacts["policy"]["sha256"]
        != snapshot_entries[
            "deploy/config/skynet-l99-shadow-capabilities.json"
        ].get("sha256")
        or hashlib.sha256(
            Path(runtime["receipt"]).read_bytes()
        ).hexdigest()
        != runtime["receipt_sha256"]
        or hashlib.sha256(
            Path(runtime["manifest"]).read_bytes()
        ).hexdigest()
        != runtime["manifest_sha256"]
    ):
        raise ValueError
    preinstall = data.get("preinstall")
    if not isinstance(preinstall, dict):
        raise ValueError
    backup = Path(preinstall["backup"])
    manifest_path = Path(preinstall["manifest"])
    if (
        backup.parent != rollback_root
        or not backup.name.startswith("preinstall-")
        or manifest_path != backup / "backup-manifest.json"
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != preinstall.get("manifest_sha256")
    ):
        raise ValueError
    backup_manifest = json.loads(
        manifest_path.read_text(),
        object_pairs_hook=unique_object,
    )
    if (
        backup_manifest.get("schema")
        != "openjarvis.skynet-l99.shadow-preinstall.v2"
        or backup_manifest.get("backup") != str(backup)
    ):
        raise ValueError
    entries = backup_manifest.get("files")
    expected_labels = {
        "plist",
        "helper",
        "config",
        "policy",
        "stdout",
        "stderr",
        "receipt",
    }
    if (
        not isinstance(entries, list)
        or {item.get("label") for item in entries} != expected_labels
    ):
        raise ValueError
    for item in entries:
        if item.get("present"):
            backup_file = backup / item["backup_name"]
            if (
                hashlib.sha256(backup_file.read_bytes()).hexdigest()
                != item.get("sha256")
            ):
                raise ValueError
        elif (
            item.get("backup_name") is not None
            or item.get("sha256") is not None
        ):
            raise ValueError
    directories = backup_manifest.get("directories")
    if (
        not isinstance(directories, list)
        or {item.get("label") for item in directories} != {
            "launch_agents",
            "shadow",
            "config",
            "bin",
            "logs",
            "state",
        }
        or any(
            not isinstance(item.get("present"), bool)
            or (
                item["present"]
                and not isinstance(item.get("mode"), str)
            )
            or (
                not item["present"]
                and item.get("mode") is not None
            )
            for item in directories
        )
    ):
        raise ValueError
except BaseException:
    raise SystemExit(78)
PY
    then
        die "shadow install receipt validation failed"
    fi
}

cleanup_install_transaction() {
    local transaction="${INSTALL_TRANSACTION_DIR:-}"
    [[ -n "$transaction" ]] || return 0
    safe_python - "$transaction_root" "$transaction" <<'PY'
import os
import re
import shutil
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
transaction = Path(sys.argv[2])
if (
    transaction.parent != root
    or not re.fullmatch(
        r"install-[0-9]{8}T[0-9]{6}Z[.][A-Za-z0-9]{6}",
        transaction.name,
    )
):
    raise SystemExit(78)
try:
    info = transaction.lstat()
except FileNotFoundError:
    raise SystemExit(0)
if (
    transaction.is_symlink()
    or not stat.S_ISDIR(info.st_mode)
    or info.st_uid != os.getuid()
):
    raise SystemExit(78)
for current, directories, _files in os.walk(transaction, topdown=False):
    for name in directories:
        os.chmod(Path(current) / name, 0o700, follow_symlinks=False)
    os.chmod(current, 0o700, follow_symlinks=False)
shutil.rmtree(transaction)
PY
    INSTALL_TRANSACTION_DIR=""
}

write_transaction_receipt() {
    local output="$1"
    safe_python - \
        "$output" \
        "$PREINSTALL_BACKUP_DIR" \
        "$PREINSTALL_MANIFEST_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
backup = Path(sys.argv[2])
manifest_sha256 = sys.argv[3]
payload = {
    "schema": "openjarvis.skynet-l99.shadow-install-transaction.v1",
    "preinstall": {
        "backup": str(backup),
        "manifest": str(backup / "backup-manifest.json"),
        "manifest_sha256": manifest_sha256,
    },
}
descriptor = os.open(
    output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    validate_owned_regular_file "$output" "600"
}

write_shadow_receipt() {
    local output="$1"
    local source_commit="$2"
    local installed_at="$3"
    local runtime_site="$4"
    local runtime_wheel_sha256="$5"
    local runtime_tree_digest="$6"
    local runtime_receipt="$7"
    local receipt_sha256="$8"
    local runtime_manifest="$9"
    local manifest_sha256="${10}"
    local config_sha256="${11}"
    local policy_sha256="${12}"
    local helper_sha256="${13}"
    local plist_sha256="${14}"

    safe_python - \
        "$output" \
        "$SOURCE_SNAPSHOT_MANIFEST" \
        "$SOURCE_SNAPSHOT_MANIFEST_SHA256" \
        "$source_commit" \
        "$installed_at" \
        "$runtime_site" \
        "$runtime_wheel_sha256" \
        "$runtime_tree_digest" \
        "$runtime_receipt" \
        "$receipt_sha256" \
        "$runtime_manifest" \
        "$manifest_sha256" \
        "$PREINSTALL_BACKUP_DIR" \
        "$PREINSTALL_MANIFEST_SHA256" \
        "$config_dir/skynet-l99-shadow.toml" \
        "$config_sha256" \
        "$config_dir/skynet-l99-shadow-capabilities.json" \
        "$policy_sha256" \
        "$helper_path" \
        "$helper_sha256" \
        "$plist_path" \
        "$plist_sha256" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    output,
    snapshot_manifest_path,
    snapshot_manifest_sha256,
    source_commit,
    installed_at,
    runtime_site,
    runtime_wheel_sha256,
    runtime_tree_digest,
    runtime_receipt,
    receipt_sha256,
    runtime_manifest,
    manifest_sha256,
    preinstall_backup,
    preinstall_manifest_sha256,
    config_path,
    config_sha256,
    policy_path,
    policy_sha256,
    helper_path,
    helper_sha256,
    plist_path,
    plist_sha256,
) = sys.argv[1:]
snapshot_payload = Path(snapshot_manifest_path).read_bytes()
import hashlib
if hashlib.sha256(snapshot_payload).hexdigest() != snapshot_manifest_sha256:
    raise SystemExit(78)
snapshot = json.loads(snapshot_payload)
payload = {
    "schema": "openjarvis.skynet-l99.shadow-launcher.v3",
    "source_commit": source_commit,
    "source_snapshot": {
        "schema": snapshot.get("schema"),
        "manifest_sha256": snapshot_manifest_sha256,
        "entries": snapshot.get("entries"),
    },
    "installed_at": installed_at,
    "activation": "dormant",
    "run_at_load": False,
    "keep_alive": False,
    "runtime_integrity": "receipt-manifest-and-tree-pinned",
    "runtime": {
        "site": runtime_site,
        "wheel_sha256": runtime_wheel_sha256,
        "tree_sha256": runtime_tree_digest,
        "receipt": runtime_receipt,
        "receipt_sha256": receipt_sha256,
        "manifest": runtime_manifest,
        "manifest_sha256": manifest_sha256,
    },
    "preinstall": {
        "backup": preinstall_backup,
        "manifest": str(Path(preinstall_backup) / "backup-manifest.json"),
        "manifest_sha256": preinstall_manifest_sha256,
    },
    "artifacts": {
        "config": {"path": config_path, "sha256": config_sha256},
        "policy": {"path": policy_path, "sha256": policy_sha256},
        "helper": {"path": helper_path, "sha256": helper_sha256},
        "plist": {"path": plist_path, "sha256": plist_sha256},
    },
}
descriptor = os.open(
    output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    validate_owned_regular_file "$output" "600"
}

apply_staged_install() {
    safe_python - \
        "$1" "$config_dir/skynet-l99-shadow.toml" "600" \
        "$2" "$config_dir/skynet-l99-shadow-capabilities.json" "600" \
        "$3" "$helper_path" "700" \
        "$4" "$stdout_path" "600" \
        "$5" "$stderr_path" "600" \
        "$6" "$plist_path" "600" \
        "$7" "$shadow_receipt_path" "600" <<'PY'
# openjarvis.skynet-l99.shadow-install-atomic.v1
import os
import stat
import sys
import tempfile
from pathlib import Path

try:
    values = sys.argv[1:]
    if len(values) != 7 * 3:
        raise ValueError
    operations = []

    def safe_directory(path):
        info = path.lstat()
        if (
            path.is_symlink()
            or path.resolve(strict=True) != path
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ValueError

    def read_file(path, mode):
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            lexical = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != mode
                or before.st_uid != os.getuid()
                or (before.st_dev, before.st_ino)
                != (lexical.st_dev, lexical.st_ino)
            ):
                raise ValueError
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def atomic_write(path, payload, mode):
        safe_directory(path.parent)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".openjarvis-install.",
            dir=path.parent,
        )
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = -1
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    for offset in range(0, len(values), 3):
        stage = Path(values[offset])
        target = Path(values[offset + 1])
        mode = int(values[offset + 2], 8)
        safe_directory(stage.parent)
        safe_directory(target.parent)
        staged_payload = read_file(stage, mode)
        try:
            target.lstat()
        except FileNotFoundError:
            current_payload = None
        else:
            current_payload = read_file(target, mode)
        operations.append((target, mode, staged_payload, current_payload))

    applied = []
    try:
        for target, mode, staged_payload, current_payload in operations:
            atomic_write(target, staged_payload, mode)
            applied.append((target, mode, current_payload))
    except BaseException:
        for target, mode, current_payload in reversed(applied):
            if current_payload is None:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            else:
                atomic_write(target, current_payload, mode)
        raise
except BaseException:
    raise SystemExit(78)
PY
}

install_error_handler() {
    local status="${1:-$?}"
    local stamp destination
    trap - ERR
    set +e
    INSTALL_ROLLBACK_ACTIVE="1"
    if [[ -n "${INSTALL_TRANSACTION_RECEIPT:-}" ]] &&
        path_exists "$INSTALL_TRANSACTION_RECEIPT"; then
        stamp="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
        destination="$(
            /usr/bin/mktemp -d "$rollback_root/failed-install-$stamp.XXXXXX"
        )"
        /bin/chmod 700 "$destination"
        if ! restore_preinstall_backup \
            "$INSTALL_TRANSACTION_RECEIPT" \
            "$destination"; then
            printf '%s\n' \
                "manage-skynet-l99-shadow: automatic install rollback failed" >&2
        fi
    fi
    exit "$status"
}

install_dormant() {
    local runtime_site="$1"
    local runtime_receipt="$2"
    local runtime_manifest="$3"
    local receipt_sha256 manifest_sha256 runtime_tree_digest
    local source_commit runtime_wheel_sha256
    local snapshot_root stage_dir helper_tmp plist_tmp receipt_tmp stamp
    local config_tmp policy_tmp stdout_tmp stderr_tmp
    local config_sha256 policy_sha256 helper_sha256 plist_sha256

    validate_runtime_inputs "$runtime_site" "$runtime_receipt" "$runtime_manifest"
    ensure_management_directories
    stamp="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
    INSTALL_TRANSACTION_DIR="$(
        /usr/bin/mktemp -d "$transaction_root/install-$stamp.XXXXXX"
    )"
    /bin/chmod 700 "$INSTALL_TRANSACTION_DIR"
    validate_private_directory "$INSTALL_TRANSACTION_DIR"
    trap cleanup_install_transaction EXIT

    source_commit="$(repo_source_commit)"
    snapshot_root="$INSTALL_TRANSACTION_DIR/sources"
    snapshot_versioned_sources "$source_commit" "$snapshot_root"
    receipt_sha256="$(sha256_file "$runtime_receipt")"
    manifest_sha256="$(sha256_file "$runtime_manifest")"
    runtime_tree_digest="$(runtime_tree_sha256 "$SNAPSHOT_HELPER" "$runtime_site")"
    validate_runtime_provenance_contract \
        "$SNAPSHOT_HELPER" \
        "$runtime_site" \
        "$runtime_receipt" \
        "$runtime_manifest" \
        "$receipt_sha256" \
        "$manifest_sha256" \
        "$runtime_tree_digest" \
        "$source_commit"

    runtime_wheel_sha256="$(basename "$(dirname "$runtime_site")")"
    validate_output_targets
    backup_existing
    [[ -n "$PREINSTALL_BACKUP_DIR" && -n "$PREINSTALL_MANIFEST_SHA256" ]] ||
        die "preinstall backup receipt is incomplete"
    INSTALL_TRANSACTION_RECEIPT="$INSTALL_TRANSACTION_DIR/preinstall-receipt.json"
    write_transaction_receipt "$INSTALL_TRANSACTION_RECEIPT"
    trap install_error_handler ERR

    ensure_install_directories
    validate_output_targets
    stage_dir="$INSTALL_TRANSACTION_DIR/staged"
    /bin/mkdir "$stage_dir"
    /bin/chmod 700 "$stage_dir"
    validate_private_directory "$stage_dir"
    config_tmp="$stage_dir/skynet-l99-shadow.toml"
    policy_tmp="$stage_dir/skynet-l99-shadow-capabilities.json"
    helper_tmp="$stage_dir/openjarvis-skynet-l99-shadow.py"
    stdout_tmp="$stage_dir/shadow.stdout.log"
    stderr_tmp="$stage_dir/shadow.stderr.log"
    plist_tmp="$stage_dir/$LABEL.plist"
    receipt_tmp="$stage_dir/install-receipt.json"

    /usr/bin/install -m 0600 "$SNAPSHOT_CONFIG" "$config_tmp"
    /usr/bin/install -m 0600 "$SNAPSHOT_POLICY" "$policy_tmp"
    config_sha256="$(sha256_file "$config_tmp")"
    policy_sha256="$(sha256_file "$policy_tmp")"
    [[ "$config_sha256" == "$(sha256_file "$SNAPSHOT_CONFIG")" ]] ||
        die "staged config differs from the immutable source snapshot"
    [[ "$policy_sha256" == "$(sha256_file "$SNAPSHOT_POLICY")" ]] ||
        die "staged policy differs from the immutable source snapshot"

    render_helper \
        "$runtime_site" \
        "$runtime_receipt" \
        "$runtime_manifest" \
        "$receipt_sha256" \
        "$manifest_sha256" \
        "$runtime_tree_digest" \
        "$source_commit" \
        "$config_sha256" \
        "$policy_sha256" \
        "$SNAPSHOT_HELPER" \
        "$helper_tmp"
    /bin/chmod 700 "$helper_tmp"
    render_plist "$SNAPSHOT_PLIST" "$plist_tmp"
    /bin/chmod 600 "$plist_tmp"
    if path_exists "$stdout_path"; then
        backup_owned_file_no_follow \
            "$stdout_path" \
            "$stdout_tmp" \
            "600" >/dev/null
    else
        /usr/bin/install -m 0600 /dev/null "$stdout_tmp"
    fi
    if path_exists "$stderr_path"; then
        backup_owned_file_no_follow \
            "$stderr_path" \
            "$stderr_tmp" \
            "600" >/dev/null
    else
        /usr/bin/install -m 0600 /dev/null "$stderr_tmp"
    fi

    safe_python -c \
        'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())' \
        "$helper_tmp"
    safe_python "$helper_tmp" --validate-provenance-only
    /usr/bin/plutil -lint "$plist_tmp" >/dev/null
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :RunAtLoad' "$plist_tmp")" == "false" ]] ||
        die "rendered plist must not run at load"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :KeepAlive' "$plist_tmp")" == "false" ]] ||
        die "rendered plist must not keep the service alive"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$plist_tmp")" == "$bootstrap_python" ]] ||
        die "rendered plist must use the immutable system bootstrap"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:1' "$plist_tmp")" == "-I" ]] &&
        [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:2' "$plist_tmp")" == "-S" ]] ||
        die "rendered plist must disable site and .pth processing"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:3' "$plist_tmp")" == "$helper_path" ]] ||
        die "rendered plist must invoke the shadow helper"
    ! /usr/bin/grep -Eq \
        'OPENJARVIS_API_KEY|SKYNET_JARVIS_[A-Z0-9_]+_TOKEN' \
        "$plist_tmp" || die "rendered plist contains credential material"

    helper_sha256="$(sha256_file "$helper_tmp")"
    plist_sha256="$(sha256_file "$plist_tmp")"
    stamp="$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)"
    write_shadow_receipt \
        "$receipt_tmp" \
        "$source_commit" \
        "$stamp" \
        "$runtime_site" \
        "$runtime_wheel_sha256" \
        "$runtime_tree_digest" \
        "$runtime_receipt" \
        "$receipt_sha256" \
        "$runtime_manifest" \
        "$manifest_sha256" \
        "$config_sha256" \
        "$policy_sha256" \
        "$helper_sha256" \
        "$plist_sha256"

    apply_staged_install \
        "$config_tmp" \
        "$policy_tmp" \
        "$helper_tmp" \
        "$stdout_tmp" \
        "$stderr_tmp" \
        "$plist_tmp" \
        "$receipt_tmp"
    [[ "$(sha256_file "$config_dir/skynet-l99-shadow.toml")" == "$config_sha256" ]] ||
        die "installed config differs from staged snapshot"
    [[ "$(sha256_file "$config_dir/skynet-l99-shadow-capabilities.json")" == "$policy_sha256" ]] ||
        die "installed policy differs from staged snapshot"
    [[ "$(sha256_file "$helper_path")" == "$helper_sha256" ]] ||
        die "installed helper differs from staged artifact"
    [[ "$(sha256_file "$plist_path")" == "$plist_sha256" ]] ||
        die "installed plist differs from staged artifact"
    safe_python "$helper_path" --validate-installed-inputs-only
    validate_shadow_receipt

    trap - ERR
    cleanup_install_transaction
    trap - EXIT
    printf '%s\n' "Shadow launcher staged dormant; no service was loaded or started."
}

validate_installed() {
    local required

    validate_private_directory "$shadow_root"
    validate_private_directory "$config_dir"
    validate_private_directory "$bin_dir"
    validate_private_directory "$log_dir"
    validate_private_directory "$state_dir"
    validate_owned_directory "$HOME/Library/LaunchAgents"
    for required in \
        "$config_dir/skynet-l99-shadow.toml" \
        "$config_dir/skynet-l99-shadow-capabilities.json" \
        "$stdout_path" \
        "$stderr_path" \
        "$plist_path" \
        "$shadow_receipt_path"; do
        validate_owned_regular_file "$required" "600"
    done
    validate_owned_regular_file "$helper_path" "700"

    safe_python -c \
        'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())' \
        "$helper_path"
    safe_python "$helper_path" --validate-installed-inputs-only
    /usr/bin/plutil -lint "$plist_path" >/dev/null
    ! /usr/bin/grep -q '__OPENJARVIS_' "$helper_path" "$plist_path" ||
        die "an install placeholder remains unresolved"
    /usr/bin/grep -q 'os.execve' "$helper_path" ||
        die "helper must invoke the runtime without an intermediate argv"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$plist_path")" == "$bootstrap_python" ]] ||
        die "installed plist must invoke the immutable system bootstrap"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:1' "$plist_path")" == "-I" ]] ||
        die "installed plist must isolate the helper interpreter"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:2' "$plist_path")" == "-S" ]] ||
        die "installed plist must disable site and .pth processing"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:3' "$plist_path")" == "$helper_path" ]] ||
        die "installed plist must invoke the shadow helper"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :RunAtLoad' "$plist_path")" == "false" ]] ||
        die "installed plist must not run at load"
    [[ "$(/usr/libexec/PlistBuddy -c 'Print :KeepAlive' "$plist_path")" == "false" ]] ||
        die "installed plist must not keep the service alive"
    validate_shadow_receipt
    printf '%s\n' "Shadow launcher files validate; activation state was not changed."
}

restore_preinstall_backup() {
    local receipt_path="$1"
    local destination="$2"

    if ! safe_python - \
        "$receipt_path" \
        "$rollback_root" \
        "$destination" \
        "$plist_path" \
        "$helper_path" \
        "$config_dir/skynet-l99-shadow.toml" \
        "$config_dir/skynet-l99-shadow-capabilities.json" \
        "$stdout_path" \
        "$stderr_path" \
        "$shadow_receipt_path" \
        "--directories--" \
        "$HOME/Library/LaunchAgents" \
        "$shadow_root" \
        "$config_dir" \
        "$bin_dir" \
        "$log_dir" \
        "$state_dir" <<'PY'
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

try:
    receipt_path = Path(sys.argv[1])
    rollback_root = Path(sys.argv[2])
    destination = Path(sys.argv[3])
    values = sys.argv[4:]
    separator = values.index("--directories--")
    target_paths = list(map(Path, values[:separator]))
    directory_paths = list(map(Path, values[separator + 1:]))
    labels = (
        "plist",
        "helper",
        "config",
        "policy",
        "stdout",
        "stderr",
        "receipt",
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
    directory_labels = (
        "launch_agents",
        "shadow",
        "config",
        "bin",
        "logs",
        "state",
    )
    if len(target_paths) != len(labels) or len(directory_paths) != len(
        directory_labels
    ):
        raise ValueError
    expected_files = {
        label: {
            "target": target,
            "mode": mode,
            "backup_name": backup_name,
        }
        for label, target, mode, backup_name in zip(
            labels,
            target_paths,
            modes,
            backup_names,
        )
    }
    expected_directories = {
        label: path
        for label, path in zip(directory_labels, directory_paths)
    }

    def safe_directory(path, *, private=False):
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if (
            path.is_symlink()
            or path.resolve(strict=True) != path
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or mode & 0o022 != 0
            or (private and mode != 0o700)
        ):
            raise ValueError

    def read_file(path, mode, maximum=None):
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            lexical = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != mode
                or before.st_uid != os.getuid()
                or (before.st_dev, before.st_ino)
                != (lexical.st_dev, lexical.st_ino)
            ):
                raise ValueError
            chunks = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if maximum is not None and total > maximum:
                    raise ValueError
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError
            value[key] = item
        return value

    def parse_json(payload):
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
        )
        if not isinstance(value, dict):
            raise ValueError
        return value

    def atomic_write(path, payload, mode):
        safe_directory(path.parent)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".openjarvis-rollback.",
            dir=path.parent,
        )
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = -1
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    safe_directory(receipt_path.parent)
    safe_directory(rollback_root, private=True)
    safe_directory(destination, private=True)
    receipt = parse_json(read_file(receipt_path, 0o600, 1024 * 1024))
    if receipt.get("schema") not in {
        "openjarvis.skynet-l99.shadow-launcher.v3",
        "openjarvis.skynet-l99.shadow-install-transaction.v1",
    }:
        raise ValueError
    preinstall = receipt.get("preinstall")
    if not isinstance(preinstall, dict):
        raise ValueError
    backup = Path(preinstall["backup"])
    manifest_path = Path(preinstall["manifest"])
    if (
        backup.parent != rollback_root
        or not re.fullmatch(
            r"preinstall-[0-9]{8}T[0-9]{6}Z[.][A-Za-z0-9]{6}",
            backup.name,
        )
        or manifest_path != backup / "backup-manifest.json"
    ):
        raise ValueError
    safe_directory(backup, private=True)
    manifest_payload = read_file(manifest_path, 0o600, 1024 * 1024)
    if (
        hashlib.sha256(manifest_payload).hexdigest()
        != preinstall.get("manifest_sha256")
    ):
        raise ValueError
    manifest = parse_json(manifest_payload)
    entries = manifest.get("files")
    directory_entries = manifest.get("directories")
    if (
        manifest.get("schema")
        != "openjarvis.skynet-l99.shadow-preinstall.v2"
        or manifest.get("backup") != str(backup)
        or not isinstance(entries, list)
        or len(entries) != len(labels)
        or not isinstance(directory_entries, list)
        or len(directory_entries) != len(directory_labels)
    ):
        raise ValueError

    backup_payloads = {}
    for item in entries:
        if (
            not isinstance(item, dict)
            or item.get("label") not in expected_files
        ):
            raise ValueError
        label = item["label"]
        details = expected_files.pop(label)
        if (
            item.get("target") != str(details["target"])
            or item.get("mode") != f"{details['mode']:03o}"
            or not isinstance(item.get("present"), bool)
        ):
            raise ValueError
        if item["present"]:
            if item.get("backup_name") != details["backup_name"]:
                raise ValueError
            backup_file = backup / details["backup_name"]
            payload = read_file(backup_file, details["mode"])
            if hashlib.sha256(payload).hexdigest() != item.get("sha256"):
                raise ValueError
            backup_payloads[label] = payload
        elif (
            item.get("backup_name") is not None
            or item.get("sha256") is not None
        ):
            raise ValueError
        else:
            backup_payloads[label] = None
    if expected_files:
        raise ValueError

    desired_directories = {}
    for item in directory_entries:
        if (
            not isinstance(item, dict)
            or item.get("label") not in expected_directories
        ):
            raise ValueError
        label = item["label"]
        path = expected_directories.pop(label)
        present = item.get("present")
        mode_text = item.get("mode")
        if (
            item.get("path") != str(path)
            or not isinstance(present, bool)
            or (
                present
                and (
                    not isinstance(mode_text, str)
                    or not re.fullmatch(r"[0-7]{3,4}", mode_text)
                    or int(mode_text, 8) & 0o022
                )
            )
            or (not present and mode_text is not None)
        ):
            raise ValueError
        desired_directories[label] = {
            "path": path,
            "present": present,
            "mode": int(mode_text, 8) if present else None,
        }
    if expected_directories:
        raise ValueError

    current_directories = {}
    for label, path in zip(directory_labels, directory_paths):
        try:
            info = path.lstat()
        except FileNotFoundError:
            current_directories[label] = {
                "path": path,
                "present": False,
                "mode": None,
            }
        else:
            safe_directory(path)
            current_directories[label] = {
                "path": path,
                "present": True,
                "mode": stat.S_IMODE(info.st_mode),
            }

    current_payloads = {}
    for label, path, mode in zip(labels, target_paths, modes):
        try:
            path.lstat()
        except FileNotFoundError:
            current_payloads[label] = None
        else:
            current_payloads[label] = read_file(path, mode)

    archive_entries = []
    for label, mode in zip(labels, modes):
        payload = current_payloads[label]
        if payload is None:
            archive_entries.append(
                {"label": label, "present": False, "sha256": None}
            )
            continue
        archive_path = destination / label
        atomic_write(archive_path, payload, mode)
        archive_entries.append(
            {
                "label": label,
                "present": True,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    archive_manifest = json.dumps(
        {
            "schema": "openjarvis.skynet-l99.shadow-rollback.v2",
            "files": archive_entries,
            "directories": [
                {
                    "label": label,
                    "path": str(item["path"]),
                    "present": item["present"],
                    "mode": (
                        f"{item['mode']:03o}"
                        if item["present"]
                        else None
                    ),
                }
                for label, item in current_directories.items()
            ],
        },
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    atomic_write(
        destination / "rollback-manifest.json",
        archive_manifest,
        0o600,
    )

    def ensure_directory(path, mode):
        try:
            path.lstat()
        except FileNotFoundError:
            safe_directory(path.parent)
            path.mkdir(mode=mode)
        safe_directory(path)
        path.chmod(mode)

    def restore_files(payloads):
        for label, path, mode in zip(labels, target_paths, modes):
            payload = payloads[label]
            if payload is None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            else:
                atomic_write(path, payload, mode)

    try:
        for item in sorted(
            desired_directories.values(),
            key=lambda value: len(value["path"].parts),
        ):
            if item["present"]:
                ensure_directory(item["path"], item["mode"])

        restore_files(backup_payloads)

        for item in sorted(
            desired_directories.values(),
            key=lambda value: len(value["path"].parts),
            reverse=True,
        ):
            if not item["present"]:
                try:
                    item["path"].rmdir()
                except FileNotFoundError:
                    pass
        for item in desired_directories.values():
            if item["present"]:
                item["path"].chmod(item["mode"])
    except BaseException:
        for item in sorted(
            current_directories.values(),
            key=lambda value: len(value["path"].parts),
        ):
            if item["present"]:
                ensure_directory(item["path"], item["mode"])
        restore_files(current_payloads)
        for item in sorted(
            current_directories.values(),
            key=lambda value: len(value["path"].parts),
            reverse=True,
        ):
            if not item["present"]:
                try:
                    item["path"].rmdir()
                except FileNotFoundError:
                    pass
        for item in current_directories.values():
            if item["present"]:
                item["path"].chmod(item["mode"])
        raise
except BaseException:
    raise SystemExit(78)
PY
    then
        die "preinstall rollback validation or restore failed"
    fi
}

rollback_dormant_files() {
    local stamp destination path

    validate_private_directory "$shadow_root"
    validate_private_directory "$rollback_root"
    validate_owned_directory "$HOME/Library/LaunchAgents"
    if /usr/bin/pgrep -f "$helper_path" >/dev/null 2>&1; then
        die "shadow helper is running; stop it explicitly before rollback"
    fi
    for path in "$bin_dir" "$config_dir" "$log_dir" "$state_dir"; do
        if path_exists "$path"; then
            validate_private_directory "$path"
        fi
    done
    validate_output_targets
    validate_shadow_receipt

    stamp="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
    destination="$(/usr/bin/mktemp -d "$rollback_root/shadow-$stamp.XXXXXX")"
    /bin/chmod 700 "$destination"
    validate_private_directory "$destination"

    restore_preinstall_backup "$shadow_receipt_path" "$destination"
    printf '%s\n' "Preinstall shadow state restored; no service action was taken."
}

validate_current_versioned_sources() {
    local stamp source_commit snapshot_root
    ensure_management_directories
    stamp="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
    INSTALL_TRANSACTION_DIR="$(
        /usr/bin/mktemp -d "$transaction_root/install-$stamp.XXXXXX"
    )"
    /bin/chmod 700 "$INSTALL_TRANSACTION_DIR"
    validate_private_directory "$INSTALL_TRANSACTION_DIR"
    trap cleanup_install_transaction EXIT
    source_commit="$(repo_source_commit)"
    snapshot_root="$INSTALL_TRANSACTION_DIR/sources"
    snapshot_versioned_sources "$source_commit" "$snapshot_root"
    cleanup_install_transaction
    trap - EXIT
}

require_air_compatible_home
case "${1:-}" in
    install)
        [[ $# -eq 4 ]] || { usage >&2; exit 2; }
        install_dormant "$2" "$3" "$4"
        ;;
    validate)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        validate_current_versioned_sources
        validate_installed
        ;;
    rollback)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        validate_current_versioned_sources
        rollback_dormant_files
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
