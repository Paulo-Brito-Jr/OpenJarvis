"""Narrow, fail-closed tools for the Skynet home-assistant boundary.

These tools intentionally do not expose a generic URL, header, entity, or
action parameter.  Every network destination and physical target is fixed by
code; capability-specific credentials are loaded from the process environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_MAX_RESPONSE_BYTES = 65_536
_HTTP_TIMEOUT = httpx.Timeout(8.5, connect=3.0, read=8.5, write=5.0, pool=2.0)

_SESSION_ID_RE = re.compile(r"^\d{4}-\d{4}-[a-z0-9]{2,8}$")
_BINDING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")
_ISO_OFFSET_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)

_CASA_ALIASES = (
    "casa",
    "luzes",
    "tomadas",
    "clima",
    "aberturas",
    "energia",
    "entretenimento",
)
_CASA_OPERATIONS = ("resumo", "estado")
_AGENDA_WINDOWS = ("today", "upcoming")
_FLEET_HOSTS = ("all", "pro", "air", "pb7", "pc-casa")
_DESIRED_STATES = ("on", "off")
_APPROVAL_STATUSES = (
    "pending",
    "approved",
    "denied",
    "expired",
    "consumed",
)
_ACTION_POLICY_VERSION = "jarvis-actions.v1"
_REMINDER_POLICY_VERSION = "jarvis-agenda-reminder.v1"
_REMINDER_GATE_REQUIREMENTS = [
    "persistent_approval_lookup",
    "atomic_idempotency_claim",
    "durable_terminal_receipt",
    "verified_cancel_undo",
]

_DROP_OUTPUT_KEYS = {
    "account",
    "account_email",
    "action_digest",
    "actor",
    "actor_id",
    "authorization",
    "channel",
    "cookie",
    "device_id",
    "email",
    "entity_id",
    "executor_host",
    "headers",
    "idempotency_key",
    "internal",
    "request_id",
    "session_id",
    "token",
}
_DROP_OUTPUT_KEY_MARKERS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
)

_PROCESS_SESSION_ID: str | None = None


class _ConfigurationError(RuntimeError):
    """Raised for missing or unsafe trusted boundary configuration."""


class _DuplicateJSONKey(ValueError):
    """Raised when a response object contains an ambiguous duplicate key."""


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _generated_session_id() -> str:
    now = time.localtime()
    hhmm = f"{now.tm_hour:02d}{now.tm_min:02d}"
    return f"{hhmm}-{secrets.randbelow(10_000):04d}-jarvis"


def _session_id(environ: Mapping[str, str]) -> str:
    global _PROCESS_SESSION_ID

    configured = environ.get("SKYNET_SESSION_ID", "")
    if _SESSION_ID_RE.fullmatch(configured):
        return configured
    if _PROCESS_SESSION_ID is None:
        _PROCESS_SESSION_ID = _generated_session_id()
    return _PROCESS_SESSION_ID


def _base_url(environ: Mapping[str, str]) -> str:
    raw = environ.get("SKYNET_JARVIS_API_BASE_URL", "")
    if (
        not raw
        or raw != raw.strip()
        or any(character.isspace() or ord(character) < 32 for character in raw)
    ):
        raise _ConfigurationError

    try:
        parsed = urllib.parse.urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise _ConfigurationError from exc

    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or not _HOST_RE.fullmatch(hostname)
        or ".." in hostname
        or any(
            not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
            for label in hostname.split(".")
        )
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise _ConfigurationError

    return f"https://{parsed.netloc}"


def _binding_headers(
    environ: Mapping[str, str],
    prefix: str,
    *,
    include_content_type: bool,
) -> dict[str, str]:
    token = environ.get(f"{prefix}_TOKEN", "")
    device_id = environ.get(f"{prefix}_DEVICE_ID", "")
    channel = environ.get(f"{prefix}_CHANNEL", "")

    if (
        len(token) < 32
        or len(token) > 512
        or token != token.strip()
        or any(ord(character) < 33 or ord(character) == 127 for character in token)
        or not _BINDING_RE.fullmatch(device_id)
        or channel != "openjarvis-local"
    ):
        raise _ConfigurationError

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {token}",
        "x-skynet-channel": channel,
        "x-skynet-device-id": device_id,
        "x-skynet-session-id": _session_id(environ),
    }
    if include_content_type:
        headers["content-type"] = "application/json"
    return headers


def _sanitize_output(
    value: Any,
    *,
    depth: int = 0,
    redact_values: tuple[str, ...] = (),
) -> Any:
    if depth > 12:
        return None
    if isinstance(value, dict):
        return {
            key: _sanitize_output(
                item,
                depth=depth + 1,
                redact_values=redact_values,
            )
            for key, item in value.items()
            if isinstance(key, str) and not _sensitive_output_key(key)
        }
    if isinstance(value, list):
        return [
            _sanitize_output(
                item,
                depth=depth + 1,
                redact_values=redact_values,
            )
            for item in value[:200]
        ]
    if isinstance(value, str):
        sanitized = value[:4_096]
        for secret_value in redact_values:
            if secret_value:
                sanitized = sanitized.replace(secret_value, "[redacted]")
        return sanitized
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def _sensitive_output_key(key: str) -> bool:
    snake = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = snake.replace("_", "")
    blocked_compact = {item.replace("_", "") for item in _DROP_OUTPUT_KEYS}
    return (
        snake in _DROP_OUTPUT_KEYS
        or compact in blocked_compact
        or any(marker in compact for marker in _DROP_OUTPUT_KEY_MARKERS)
    )


def _valid_uuid(value: Any) -> bool:
    if not isinstance(value, str) or not _UUID_RE.fullmatch(value):
        return False
    try:
        return str(uuid.UUID(value)).lower() == value.lower()
    except ValueError:
        return False


def _valid_idempotency_key(value: Any) -> bool:
    return isinstance(value, str) and _IDEMPOTENCY_RE.fullmatch(value) is not None


def _valid_scheduled_for(value: Any) -> bool:
    if not isinstance(value, str) or _ISO_OFFSET_RE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _valid_reminder_text(value: Any) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 240:
        return False
    normalized = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value.strip())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(normalized) and len(normalized) <= 240


def _normalize_reminder_text(value: str) -> str:
    normalized = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value.strip())
    return re.sub(r"\s+", " ", normalized).strip()


def _canonical_utc_millis(value: Any) -> str | None:
    if not _valid_scheduled_for(value):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _valid_iso_timestamp(value: Any) -> bool:
    return _canonical_utc_millis(value) is not None


def _valid_bound_identifier(value: Any) -> bool:
    return isinstance(value, str) and _BINDING_RE.fullmatch(value) is not None


def _identity_matches(
    value: Any,
    headers: Mapping[str, str],
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"actor", "channel", "device_id", "session_id"}
        and value.get("actor") == "unverified"
        and value.get("channel") == headers.get("x-skynet-channel")
        and value.get("device_id") == headers.get("x-skynet-device-id")
        and value.get("session_id") == headers.get("x-skynet-session-id")
    )


def _valid_approval_payload(
    payload: Mapping[str, Any],
    *,
    approval_id: Any | None = None,
    desired_state: Any | None = None,
) -> bool:
    if (
        not _valid_uuid(payload.get("approval_id"))
        or payload.get("action") != "casa.definir_estado"
        or payload.get("target_alias") != "tomada_cozinha_backlight"
        or payload.get("desired_state") not in _DESIRED_STATES
        or payload.get("status") not in _APPROVAL_STATUSES
        or not _valid_iso_timestamp(payload.get("expires_at"))
        or payload.get("policy_version") != _ACTION_POLICY_VERSION
        or not isinstance(payload.get("replayed"), bool)
    ):
        return False
    if approval_id is not None and payload.get("approval_id") != approval_id:
        return False
    return desired_state is None or payload.get("desired_state") == desired_state


def _only_keys(params: Mapping[str, Any], allowed: frozenset[str]) -> bool:
    return set(params).issubset(allowed)


def _invalid(tool_name: str) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        content="Invalid bounded Skynet tool arguments.",
        success=False,
        metadata={"error_code": "invalid_arguments"},
    )


class _SkynetBoundaryTool(BaseTool):
    """Shared exact-origin HTTP client for bounded Skynet tools."""

    is_local = False
    _credential_prefix: str
    _path: str
    _schemas: frozenset[str]

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._transport = transport

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        return True

    def _request(
        self,
        *,
        method: str,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
        outcome_unknown_on_failure: bool = False,
    ) -> ToolResult:
        try:
            url = f"{_base_url(self._environ)}{self._path}"
            headers = _binding_headers(
                self._environ,
                self._credential_prefix,
                include_content_type=body is not None,
            )
        except _ConfigurationError:
            return ToolResult(
                tool_name=self.spec.name,
                content="Skynet boundary is not safely configured.",
                success=False,
                metadata={"error_code": "unsafe_configuration"},
            )

        failure_metadata: dict[str, Any] = {"error_code": "request_failed"}
        if outcome_unknown_on_failure:
            failure_metadata.update({"outcome": "unknown", "reconcile_required": True})

        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=_HTTP_TIMEOUT,
                transport=self._transport,
                trust_env=False,
            ) as client:
                with client.stream(
                    method,
                    url,
                    params=query,
                    json=body,
                    headers=headers,
                ) as response:
                    if response.is_redirect:
                        failure_metadata["error_code"] = "redirect_denied"
                        return ToolResult(
                            tool_name=self.spec.name,
                            content="Skynet request failed safely.",
                            success=False,
                            metadata=failure_metadata,
                        )

                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_RESPONSE_BYTES:
                            failure_metadata["error_code"] = "response_too_large"
                            return ToolResult(
                                tool_name=self.spec.name,
                                content="Skynet request failed safely.",
                                success=False,
                                metadata=failure_metadata,
                            )
                        chunks.append(chunk)

                    failure_metadata["status_code"] = response.status_code
                    if response.status_code < 200 or response.status_code >= 300:
                        failure_metadata["error_code"] = "remote_rejected"
                        return ToolResult(
                            tool_name=self.spec.name,
                            content="Skynet request failed safely.",
                            success=False,
                            metadata=failure_metadata,
                        )

                    content_type = response.headers.get("content-type", "")
                    media_type = content_type.split(";", 1)[0].strip().lower()
                    if media_type != "application/json" and not media_type.endswith(
                        "+json"
                    ):
                        failure_metadata["error_code"] = "non_json_response"
                        return ToolResult(
                            tool_name=self.spec.name,
                            content="Skynet request failed safely.",
                            success=False,
                            metadata=failure_metadata,
                        )

            payload = json.loads(
                b"".join(chunks).decode("utf-8"),
                object_pairs_hook=_json_object,
            )
            if (
                not isinstance(payload, dict)
                or payload.get("ok") is not True
                or payload.get("schema") not in self._schemas
                or not self._valid_response_payload(
                    payload,
                    query=query or {},
                    body=body or {},
                    headers=headers,
                )
            ):
                failure_metadata["error_code"] = "invalid_response_contract"
                return ToolResult(
                    tool_name=self.spec.name,
                    content="Skynet request failed safely.",
                    success=False,
                    metadata=failure_metadata,
                )
        except (
            httpx.HTTPError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJSONKey,
        ):
            return ToolResult(
                tool_name=self.spec.name,
                content="Skynet request failed safely.",
                success=False,
                metadata=failure_metadata,
            )

        sanitized = _sanitize_output(
            payload,
            redact_values=(
                headers["authorization"],
                headers["authorization"].removeprefix("Bearer "),
                headers["x-skynet-device-id"],
                headers["x-skynet-session-id"],
            ),
        )
        return ToolResult(
            tool_name=self.spec.name,
            content=json.dumps(
                sanitized,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            success=True,
            metadata={
                "schema": payload["schema"],
                "status_code": failure_metadata["status_code"],
            },
        )


@ToolRegistry.register("skynet_casa_read")
class SkynetCasaReadTool(_SkynetBoundaryTool):
    """Read a curated, redacted Home Assistant snapshot."""

    tool_id = "skynet_casa_read"
    _credential_prefix = "SKYNET_JARVIS_CASA_READ"
    _path = "/api/agent/jarvis/casa"
    _schemas = frozenset({"jarvis.casa.snapshot.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Read a curated, redacted state snapshot from the home.",
            parameters={
                "type": "object",
                "properties": {
                    "alias": {"type": "string", "enum": list(_CASA_ALIASES)},
                    "operation": {
                        "type": "string",
                        "enum": list(_CASA_OPERATIONS),
                    },
                },
                "required": ["alias", "operation"],
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=["network:fetch", "skynet:casa:read"],
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        alias = params.get("alias")
        if alias not in _CASA_ALIASES:
            alias = "invalid"
        return f"skynet://casa/read/{alias}"

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del body
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            return False
        entities = snapshot.get("entities")
        health = snapshot.get("health")
        source = snapshot.get("source")
        return (
            payload.get("alias") == query.get("alias")
            and payload.get("operation") == query.get("operation")
            and payload.get("audit_status") == "not_persisted"
            and _valid_iso_timestamp(payload.get("captured_at"))
            and _valid_bound_identifier(payload.get("receipt_id"))
            and _identity_matches(payload.get("identity"), headers)
            and isinstance(entities, list)
            and len(entities) <= 200
            and snapshot.get("total") == len(entities)
            and isinstance(snapshot.get("truncated"), bool)
            and isinstance(health, dict)
            and isinstance(health.get("fresh"), bool)
            and health.get("reachable") is True
            and isinstance(source, dict)
            and source.get("instance") == "ha-casa-proxmox"
        )

    def execute(self, **params: Any) -> ToolResult:
        alias = params.get("alias")
        operation = params.get("operation")
        if (
            not _only_keys(params, frozenset({"alias", "operation"}))
            or alias not in _CASA_ALIASES
            or operation not in _CASA_OPERATIONS
        ):
            return _invalid(self.tool_id)
        return self._request(
            method="GET",
            query={"alias": alias, "operation": operation},
        )


@ToolRegistry.register("skynet_agenda_read")
class SkynetAgendaReadTool(_SkynetBoundaryTool):
    """Read only redacted busy windows from the trusted calendar mirror."""

    tool_id = "skynet_agenda_read"
    _credential_prefix = "SKYNET_JARVIS_AGENDA_READ"
    _path = "/api/agent/jarvis/agenda"
    _schemas = frozenset({"jarvis.agenda.snapshot.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Read redacted busy windows without event titles or identities."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window": {
                        "type": "string",
                        "enum": list(_AGENDA_WINDOWS),
                        "default": "today",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 7,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                    },
                },
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=["network:fetch", "skynet:agenda:read"],
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        return "skynet://agenda/busy"

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del body, headers
        window = payload.get("window")
        agenda = payload.get("agenda")
        if not isinstance(window, dict) or not isinstance(agenda, dict):
            return False
        events = agenda.get("events")
        policy = agenda.get("content_policy")
        if not isinstance(events, list) or not isinstance(policy, dict):
            return False

        expected_days = 1 if query.get("window") == "today" else int(query["days"])
        starts_at = _canonical_utc_millis(window.get("starts_at"))
        ends_at = _canonical_utc_millis(window.get("ends_at"))
        if (
            starts_at is None
            or ends_at is None
            or datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
            >= datetime.fromisoformat(ends_at.replace("Z", "+00:00"))
        ):
            return False

        for event in events:
            if not isinstance(event, dict) or set(event) != {
                "availability",
                "starts_at",
                "ends_at",
                "all_day",
            }:
                return False
            event_start = _canonical_utc_millis(event.get("starts_at"))
            event_end = (
                _canonical_utc_millis(event.get("ends_at"))
                if event.get("ends_at") is not None
                else None
            )
            if (
                event.get("availability") != "busy"
                or event_start is None
                or not isinstance(event.get("all_day"), bool)
                or event_start < starts_at
                or event_start >= ends_at
                or (event.get("ends_at") is not None and event_end is None)
                or (event_end is not None and event_end < event_start)
            ):
                return False

        return (
            payload.get("audit_status") == "not_persisted"
            and _valid_iso_timestamp(payload.get("captured_at"))
            and _valid_bound_identifier(payload.get("receipt_id"))
            and window.get("kind") == query.get("window")
            and window.get("days") == expected_days
            and window.get("timezone") == "America/Sao_Paulo"
            and agenda.get("source") == "apple_calendar_mirror"
            and agenda.get("reachable") is True
            and agenda.get("freshness") == "not_observed"
            and policy.get("event_titles_exposed") is False
            and policy.get("executable") is False
            and agenda.get("total") == len(events)
            and len(events) <= int(query["limit"])
            and isinstance(agenda.get("truncated"), bool)
        )

    def execute(self, **params: Any) -> ToolResult:
        window = params.get("window", "today")
        days = params.get("days", 7)
        limit = params.get("limit", 10)
        if (
            not _only_keys(params, frozenset({"window", "days", "limit"}))
            or window not in _AGENDA_WINDOWS
            or isinstance(days, bool)
            or not isinstance(days, int)
            or not 1 <= days <= 7
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 20
        ):
            return _invalid(self.tool_id)
        return self._request(
            method="GET",
            query={"window": window, "days": str(days), "limit": str(limit)},
        )


@ToolRegistry.register("skynet_agenda_reminder_plan")
class SkynetAgendaReminderPlanTool(_SkynetBoundaryTool):
    """Prepare a reminder plan without persisting or executing a reminder."""

    tool_id = "skynet_agenda_reminder_plan"
    _credential_prefix = "SKYNET_JARVIS_REMINDER_PLAN"
    _path = "/api/agent/jarvis/agenda/reminder"
    _schemas = frozenset({"jarvis.agenda.reminder.plan.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Validate and return a dry-run plan for Paulo's reminder. "
                "This never persists, schedules, or executes a reminder."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "approval_id": {"type": "string", "format": "uuid"},
                    "scheduled_for": {"type": "string", "format": "date-time"},
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 240,
                    },
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 16,
                        "maxLength": 128,
                        "pattern": _IDEMPOTENCY_RE.pattern,
                    },
                },
                "required": [
                    "approval_id",
                    "scheduled_for",
                    "text",
                    "idempotency_key",
                ],
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=[
                "network:fetch",
                "skynet:agenda:reminder:plan",
            ],
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        return "skynet://agenda/reminder/plan"

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del query
        undo = payload.get("undo")
        plan = payload.get("plan")
        gate = payload.get("gate")
        identity = payload.get("identity")
        if (
            not isinstance(plan, dict)
            or not isinstance(undo, dict)
            or not isinstance(gate, dict)
            or not _identity_matches(identity, headers)
        ):
            return False

        scheduled_for = _canonical_utc_millis(body.get("scheduled_for"))
        text = body.get("text")
        if scheduled_for is None or not isinstance(text, str):
            return False
        normalized_text = _normalize_reminder_text(text)
        text_sha256 = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        digest_input = {
            "action": "agenda.reminder.create",
            "actor": "unverified",
            "approval_id": body.get("approval_id"),
            "channel": headers.get("x-skynet-channel"),
            "device_id": headers.get("x-skynet-device-id"),
            "idempotency_key": body.get("idempotency_key"),
            "policy_version": _REMINDER_POLICY_VERSION,
            "scheduled_for": scheduled_for,
            "session_id": headers.get("x-skynet-session-id"),
            "target_alias": "paulo_self",
            "text_sha256": text_sha256,
        }
        action_digest = hashlib.sha256(
            json.dumps(
                digest_input,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        plan_text = plan.get("text")
        idempotency = plan.get("idempotency")
        return (
            payload.get("dry_run") is True
            and payload.get("mutation_executed") is False
            and payload.get("audit_status") == "not_persisted"
            and payload.get("plan_id") == f"plan_{action_digest[:32]}"
            and plan.get("action") == "agenda.reminder.create"
            and plan.get("target_alias") == "paulo_self"
            and plan.get("scheduled_for") == scheduled_for
            and isinstance(plan_text, dict)
            and plan_text.get("length") == len(normalized_text)
            and plan_text.get("sha256") == text_sha256
            and plan_text.get("trust") == "untrusted_user_data"
            and plan_text.get("echoed") is False
            and plan.get("action_digest") == action_digest
            and plan.get("policy_version") == _REMINDER_POLICY_VERSION
            and plan.get("approval_binding") == "required_but_not_verified_in_dry_run"
            and isinstance(idempotency, dict)
            and idempotency.get("key_bound_to_digest") is True
            and idempotency.get("atomic_claim") == "not_persisted"
            and undo.get("action") == "agenda.reminder.cancel"
            and undo.get("availability") == "blocked_until_durable_backend"
            and gate.get("error_if_executed") == "reminder_backend_not_ready"
            and gate.get("requirements") == _REMINDER_GATE_REQUIREMENTS
        )

    def execute(self, **params: Any) -> ToolResult:
        approval_id = params.get("approval_id")
        scheduled_for = params.get("scheduled_for")
        text = params.get("text")
        idempotency_key = params.get("idempotency_key")
        if (
            not _only_keys(
                params,
                frozenset(
                    {
                        "approval_id",
                        "scheduled_for",
                        "text",
                        "idempotency_key",
                    }
                ),
            )
            or not _valid_uuid(approval_id)
            or not _valid_scheduled_for(scheduled_for)
            or not _valid_reminder_text(text)
            or not _valid_idempotency_key(idempotency_key)
        ):
            return _invalid(self.tool_id)

        return self._request(
            method="POST",
            body={
                "action": "agenda.reminder.create",
                "approval_id": approval_id,
                "dry_run": True,
                "idempotency_key": idempotency_key,
                "scheduled_for": scheduled_for,
                "target_alias": "paulo_self",
                "text": text,
            },
        )


@ToolRegistry.register("skynet_frota_read")
class SkynetFrotaReadTool(_SkynetBoundaryTool):
    """Read the canonical health projection for one or all fleet hosts."""

    tool_id = "skynet_frota_read"
    _credential_prefix = "SKYNET_JARVIS_FROTA_READ"
    _path = "/api/agent/jarvis/frota"
    _schemas = frozenset({"jarvis.frota.snapshot.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Read conservative canonical health for the four fleet hosts.",
            parameters={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "enum": list(_FLEET_HOSTS),
                        "default": "all",
                    }
                },
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=["network:fetch", "skynet:frota:read"],
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        host = params.get("host", "all")
        if host not in _FLEET_HOSTS:
            host = "invalid"
        return f"skynet://frota/{host}"

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del body, headers
        fleet = payload.get("fleet")
        if not isinstance(fleet, dict):
            return False
        machines = fleet.get("machines")
        if not isinstance(machines, list):
            return False
        expected_hosts = (
            set(_FLEET_HOSTS[1:])
            if query.get("host") == "all"
            else {str(query.get("host"))}
        )
        seen: set[str] = set()
        observed_hosts = 0
        stale_hosts = 0
        labels = {
            "pro": "Pro",
            "air": "Air",
            "pb7": "PB7",
            "pc-casa": "PC-Casa",
        }
        for machine in machines:
            if not isinstance(machine, dict):
                return False
            host = machine.get("host")
            observed_at = machine.get("observed_at")
            age_seconds = machine.get("age_seconds")
            health = machine.get("health")
            online = machine.get("online")
            stale = machine.get("stale")
            if (
                host not in expected_hosts
                or host in seen
                or machine.get("label") != labels.get(host)
                or health not in {"healthy", "degraded", "offline", "unknown"}
                or online not in {True, False, None}
                or not isinstance(stale, bool)
                or (observed_at is not None and not _valid_iso_timestamp(observed_at))
                or (
                    age_seconds is not None
                    and (
                        isinstance(age_seconds, bool)
                        or not isinstance(age_seconds, int)
                        or age_seconds < 0
                    )
                )
                or ((observed_at is None) != (age_seconds is None))
                or (health == "healthy" and online is not True)
                or (health == "offline" and online is not False)
                or (health in {"degraded", "unknown"} and online is not None)
                or not _valid_bound_identifier(machine.get("reason"))
            ):
                return False
            seen.add(str(host))
            observed_hosts += observed_at is not None
            stale_hosts += stale

        return (
            payload.get("audit_status") == "not_persisted"
            and _valid_iso_timestamp(payload.get("captured_at"))
            and _valid_bound_identifier(payload.get("receipt_id"))
            and seen == expected_hosts
            and fleet.get("expected_hosts") == len(machines)
            and fleet.get("observed_hosts") == observed_hosts
            and fleet.get("stale_hosts") == stale_hosts
        )

    def execute(self, **params: Any) -> ToolResult:
        host = params.get("host", "all")
        if not _only_keys(params, frozenset({"host"})) or host not in _FLEET_HOSTS:
            return _invalid(self.tool_id)
        return self._request(method="GET", query={"host": host})


class _SkynetCasaActionTool(_SkynetBoundaryTool):
    _credential_prefix = "SKYNET_JARVIS_CASA_ACTION"

    @staticmethod
    def _action_body(desired_state: str, idempotency_key: str) -> dict[str, Any]:
        return {
            "action": "casa.definir_estado",
            "args": {"desired_state": desired_state},
            "idempotency_key": idempotency_key,
            "target_alias": "tomada_cozinha_backlight",
        }

    def authorization_resource(self, params: dict[str, Any]) -> str:
        return "skynet://casa/tomada_cozinha_backlight"


@ToolRegistry.register("skynet_casa_request_action")
class SkynetCasaRequestActionTool(_SkynetCasaActionTool):
    """Create a pending request; this tool cannot approve or execute it."""

    tool_id = "skynet_casa_request_action"
    _path = "/api/agent/jarvis/casa/action/approval"
    _schemas = frozenset({"jarvis.casa.approval.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Request human approval for the fixed kitchen outlet indicator. "
                "This does not execute the physical action."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "desired_state": {
                        "type": "string",
                        "enum": list(_DESIRED_STATES),
                    },
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 16,
                        "maxLength": 128,
                        "pattern": _IDEMPOTENCY_RE.pattern,
                    },
                },
                "required": ["desired_state", "idempotency_key"],
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=[
                "network:fetch",
                "skynet:casa:approval:request",
            ],
        )

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del query, headers
        args = body.get("args")
        return isinstance(args, dict) and _valid_approval_payload(
            payload,
            desired_state=args.get("desired_state"),
        )

    def execute(self, **params: Any) -> ToolResult:
        desired_state = params.get("desired_state")
        idempotency_key = params.get("idempotency_key")
        if (
            not _only_keys(
                params,
                frozenset({"desired_state", "idempotency_key"}),
            )
            or desired_state not in _DESIRED_STATES
            or not _valid_idempotency_key(idempotency_key)
        ):
            return _invalid(self.tool_id)
        return self._request(
            method="POST",
            body=self._action_body(desired_state, idempotency_key),
        )


@ToolRegistry.register("skynet_casa_action_status")
class SkynetCasaActionStatusTool(_SkynetCasaActionTool):
    """Poll one exact approval without broad approval-list access."""

    tool_id = "skynet_casa_action_status"
    _path = "/api/agent/jarvis/casa/action/approval"
    _schemas = frozenset({"jarvis.casa.approval.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Read the status of one exact physical-action approval.",
            parameters={
                "type": "object",
                "properties": {"approval_id": {"type": "string", "format": "uuid"}},
                "required": ["approval_id"],
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=[
                "network:fetch",
                "skynet:casa:approval:status",
            ],
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        approval_id = params.get("approval_id")
        if not _valid_uuid(approval_id):
            approval_id = "invalid"
        return f"skynet://casa/approval/{approval_id}"

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del body, headers
        return _valid_approval_payload(
            payload,
            approval_id=query.get("approval_id"),
        )

    def execute(self, **params: Any) -> ToolResult:
        approval_id = params.get("approval_id")
        if not _only_keys(params, frozenset({"approval_id"})) or not _valid_uuid(
            approval_id
        ):
            return _invalid(self.tool_id)
        return self._request(
            method="GET",
            query={"approval_id": approval_id},
        )


@ToolRegistry.register("skynet_casa_execute_action")
class SkynetCasaExecuteActionTool(_SkynetCasaActionTool):
    """Execute only the fixed action tied to an already-approved request."""

    tool_id = "skynet_casa_execute_action"
    _path = "/api/agent/jarvis/casa/action"
    _schemas = frozenset({"jarvis.casa.action.receipt.v1"})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Execute the fixed kitchen outlet indicator action only after "
                "the server validates a matching, unexpired human approval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "approval_id": {"type": "string", "format": "uuid"},
                    "desired_state": {
                        "type": "string",
                        "enum": list(_DESIRED_STATES),
                    },
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 16,
                        "maxLength": 128,
                        "pattern": _IDEMPOTENCY_RE.pattern,
                    },
                },
                "required": [
                    "approval_id",
                    "desired_state",
                    "idempotency_key",
                ],
                "additionalProperties": False,
            },
            category="skynet",
            timeout_seconds=10.0,
            required_capabilities=["network:fetch", "skynet:casa:write"],
        )

    def _valid_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        query: Mapping[str, str],
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> bool:
        del query, headers
        args = body.get("args")
        rollback = payload.get("rollback")
        if not isinstance(args, dict) or not isinstance(rollback, dict):
            return False
        desired_state = args.get("desired_state")
        status = payload.get("status")
        result_code = payload.get("result_code")
        return (
            body.get("dry_run") is False
            and payload.get("mode") == "execute"
            and payload.get("action") == "casa.definir_estado"
            and payload.get("target_alias") == "tomada_cozinha_backlight"
            and payload.get("desired_state") == desired_state
            and payload.get("observed_state") == desired_state
            and status in {"noop", "succeeded"}
            and (
                (status == "noop" and result_code == "already_in_state")
                or (
                    status == "succeeded"
                    and result_code in {"state_changed", "state_changed_recovered"}
                )
            )
            and _valid_bound_identifier(payload.get("receipt_id"))
            and isinstance(payload.get("replayed"), bool)
            and rollback.get("attempted") is False
            and rollback.get("succeeded") is None
        )

    def execute(self, **params: Any) -> ToolResult:
        approval_id = params.get("approval_id")
        desired_state = params.get("desired_state")
        idempotency_key = params.get("idempotency_key")
        if (
            not _only_keys(
                params,
                frozenset({"approval_id", "desired_state", "idempotency_key"}),
            )
            or not _valid_uuid(approval_id)
            or desired_state not in _DESIRED_STATES
            or not _valid_idempotency_key(idempotency_key)
        ):
            return _invalid(self.tool_id)

        body = self._action_body(desired_state, idempotency_key)
        body.update({"approval_id": approval_id, "dry_run": False})
        return self._request(
            method="POST",
            body=body,
            outcome_unknown_on_failure=True,
        )


__all__ = [
    "SkynetAgendaReadTool",
    "SkynetAgendaReminderPlanTool",
    "SkynetCasaActionStatusTool",
    "SkynetCasaExecuteActionTool",
    "SkynetCasaReadTool",
    "SkynetCasaRequestActionTool",
    "SkynetFrotaReadTool",
]
