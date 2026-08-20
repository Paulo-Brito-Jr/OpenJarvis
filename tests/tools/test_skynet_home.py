"""Security and contract tests for the bounded Skynet tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import httpx
import pytest

from openjarvis.tools.skynet_home import (
    SkynetAgendaReadTool,
    SkynetAgendaReminderPlanTool,
    SkynetCasaActionStatusTool,
    SkynetCasaExecuteActionTool,
    SkynetCasaReadTool,
    SkynetCasaRequestActionTool,
    SkynetFrotaReadTool,
)

TOKEN = "test-token-" + ("a" * 40)
SESSION_ID = "1234-5678-jarvis"
APPROVAL_ID = "5d1846d7-0c2b-4d8f-9b24-7d637ba71b0f"
IDEMPOTENCY_KEY = "jarvis-test-action-0001"
REMINDER_TEXT = "Lembrar da reunião"
SCHEDULED_FOR = "2026-07-29T15:00:00.000-03:00"
CAPTURED_AT = "2026-07-28T15:00:00.000Z"
POLICY_VERSION = "jarvis-actions.v1"


def _environment(prefix: str) -> dict[str, str]:
    return {
        "SKYNET_JARVIS_API_BASE_URL": "https://skynet.example.test",
        "SKYNET_SESSION_ID": SESSION_ID,
        f"{prefix}_TOKEN": TOKEN,
        f"{prefix}_DEVICE_ID": "air-openjarvis",
        f"{prefix}_CHANNEL": "openjarvis-local",
    }


def _transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _success(schema: str, **payload: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={"ok": True, "schema": schema, **payload},
    )


def _identity() -> dict[str, str]:
    return {
        "actor": "unverified",
        "channel": "openjarvis-local",
        "device_id": "air-openjarvis",
        "session_id": SESSION_ID,
    }


def _casa_snapshot(
    *, alias: str = "luzes", operation: str = "estado"
) -> dict[str, Any]:
    return {
        "alias": alias,
        "audit_status": "not_persisted",
        "captured_at": CAPTURED_AT,
        "identity": _identity(),
        "operation": operation,
        "receipt_id": "receipt-casa-read-0001",
        "snapshot": {
            "entities": [],
            "health": {"fresh": True, "reachable": True},
            "source": {"instance": "ha-casa-proxmox"},
            "total": 0,
            "truncated": False,
        },
    }


def _agenda_snapshot(*, window: str = "upcoming", days: int = 3) -> dict[str, Any]:
    return {
        "audit_status": "not_persisted",
        "captured_at": CAPTURED_AT,
        "receipt_id": "receipt-agenda-read-0001",
        "window": {
            "kind": window,
            "days": days,
            "starts_at": "2026-07-28T15:00:00.000Z",
            "ends_at": "2026-07-31T15:00:00.000Z",
            "timezone": "America/Sao_Paulo",
        },
        "agenda": {
            "source": "apple_calendar_mirror",
            "reachable": True,
            "freshness": "not_observed",
            "content_policy": {
                "event_titles_exposed": False,
                "executable": False,
            },
            "total": 0,
            "truncated": False,
            "events": [],
        },
    }


def _fleet_snapshot(host: str = "pc-casa") -> dict[str, Any]:
    return {
        "audit_status": "not_persisted",
        "captured_at": CAPTURED_AT,
        "receipt_id": "receipt-frota-read-0001",
        "fleet": {
            "expected_hosts": 1,
            "observed_hosts": 0,
            "stale_hosts": 1,
            "machines": [
                {
                    "host": host,
                    "label": "PC-Casa",
                    "health": "unknown",
                    "online": None,
                    "stale": True,
                    "observed_at": None,
                    "age_seconds": None,
                    "reason": "no_telemetry",
                }
            ],
        },
    }


def _approval_payload(
    *,
    approval_id: str = APPROVAL_ID,
    desired_state: str = "off",
    status: str = "pending",
) -> dict[str, Any]:
    return {
        "approval_id": approval_id,
        "action": "casa.definir_estado",
        "target_alias": "tomada_cozinha_backlight",
        "desired_state": desired_state,
        "status": status,
        "expires_at": "2026-07-28T15:10:00.000Z",
        "policy_version": POLICY_VERSION,
        "replayed": False,
    }


def _action_receipt(
    *, desired_state: str = "on", observed_state: str = "on"
) -> dict[str, Any]:
    return {
        "receipt_id": "receipt-action-0001",
        "mode": "execute",
        "action": "casa.definir_estado",
        "target_alias": "tomada_cozinha_backlight",
        "desired_state": desired_state,
        "observed_state": observed_state,
        "status": "succeeded",
        "result_code": "state_changed",
        "replayed": False,
        "rollback": {"attempted": False, "succeeded": None},
    }


def _reminder_plan() -> dict[str, Any]:
    normalized_text = REMINDER_TEXT
    scheduled_for = (
        datetime.fromisoformat(SCHEDULED_FOR)
        .astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    text_sha = sha256(normalized_text.encode()).hexdigest()
    digest_input = {
        "action": "agenda.reminder.create",
        "actor": "unverified",
        "approval_id": APPROVAL_ID,
        "channel": "openjarvis-local",
        "device_id": "air-openjarvis",
        "idempotency_key": IDEMPOTENCY_KEY,
        "policy_version": "jarvis-agenda-reminder.v1",
        "scheduled_for": scheduled_for,
        "session_id": SESSION_ID,
        "target_alias": "paulo_self",
        "text_sha256": text_sha,
    }
    action_digest = sha256(
        json.dumps(digest_input, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "request_id": "request-reminder-0001",
        "plan_id": f"plan_{action_digest[:32]}",
        "audit_status": "not_persisted",
        "dry_run": True,
        "mutation_executed": False,
        "identity": _identity(),
        "plan": {
            "action": "agenda.reminder.create",
            "target_alias": "paulo_self",
            "scheduled_for": scheduled_for,
            "text": {
                "length": len(normalized_text),
                "sha256": text_sha,
                "trust": "untrusted_user_data",
                "echoed": False,
            },
            "action_digest": action_digest,
            "policy_version": "jarvis-agenda-reminder.v1",
            "approval_binding": "required_but_not_verified_in_dry_run",
            "idempotency": {
                "key_bound_to_digest": True,
                "atomic_claim": "not_persisted",
            },
        },
        "undo": {
            "action": "agenda.reminder.cancel",
            "availability": "blocked_until_durable_backend",
        },
        "gate": {
            "error_if_executed": "reminder_backend_not_ready",
            "requirements": [
                "persistent_approval_lookup",
                "atomic_idempotency_claim",
                "durable_terminal_receipt",
                "verified_cancel_undo",
            ],
        },
    }


def test_casa_read_uses_exact_origin_path_query_and_binding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == (
            "https://skynet.example.test/api/agent/jarvis/casa"
            "?alias=luzes&operation=estado"
        )
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.headers["x-skynet-device-id"] == "air-openjarvis"
        assert request.headers["x-skynet-channel"] == "openjarvis-local"
        assert request.headers["x-skynet-session-id"] == SESSION_ID
        assert "x-skynet-request-id" not in request.headers
        return _success(
            "jarvis.casa.snapshot.v1",
            **_casa_snapshot(),
        )

    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(handler),
    )
    result = tool.execute(alias="luzes", operation="estado")

    assert result.success is True
    assert json.loads(result.content)["alias"] == "luzes"


def test_agenda_read_uses_only_bounded_window_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/agent/jarvis/agenda"
        assert dict(request.url.params) == {
            "window": "upcoming",
            "days": "3",
            "limit": "4",
        }
        return _success(
            "jarvis.agenda.snapshot.v1",
            **_agenda_snapshot(),
        )

    tool = SkynetAgendaReadTool(
        environ=_environment("SKYNET_JARVIS_AGENDA_READ"),
        transport=_transport(handler),
    )
    result = tool.execute(window="upcoming", days=3, limit=4)

    assert result.success is True


def test_agenda_reminder_plan_is_fixed_dry_run_with_dedicated_binding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == (
            "https://skynet.example.test/api/agent/jarvis/agenda/reminder"
        )
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert json.loads(request.content) == {
            "action": "agenda.reminder.create",
            "approval_id": APPROVAL_ID,
            "dry_run": True,
            "idempotency_key": IDEMPOTENCY_KEY,
            "scheduled_for": SCHEDULED_FOR,
            "target_alias": "paulo_self",
            "text": REMINDER_TEXT,
        }
        return _success(
            "jarvis.agenda.reminder.plan.v1",
            **_reminder_plan(),
        )

    tool = SkynetAgendaReminderPlanTool(
        environ=_environment("SKYNET_JARVIS_REMINDER_PLAN"),
        transport=_transport(handler),
    )
    result = tool.execute(
        approval_id=APPROVAL_ID,
        scheduled_for=SCHEDULED_FOR,
        text=REMINDER_TEXT,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result.success is True
    assert result.metadata["schema"] == "jarvis.agenda.reminder.plan.v1"
    assert tool.authorization_resource({}) == "skynet://agenda/reminder/plan"
    assert tool.spec.required_capabilities == [
        "network:fetch",
        "skynet:agenda:reminder:plan",
    ]


@pytest.mark.parametrize(
    "unsafe_response",
    [
        {"mutation_executed": True},
        {"audit_status": "persisted"},
        {"dry_run": False},
        {
            "undo": {
                "action": "agenda.reminder.cancel",
                "availability": "available",
            }
        },
    ],
)
def test_agenda_reminder_plan_response_fails_closed(
    unsafe_response: dict[str, Any],
) -> None:
    payload = _reminder_plan()
    payload.update(unsafe_response)
    tool = SkynetAgendaReminderPlanTool(
        environ=_environment("SKYNET_JARVIS_REMINDER_PLAN"),
        transport=_transport(
            lambda request: _success(
                "jarvis.agenda.reminder.plan.v1",
                **payload,
            )
        ),
    )

    result = tool.execute(
        approval_id=APPROVAL_ID,
        scheduled_for=SCHEDULED_FOR,
        text=REMINDER_TEXT,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result.success is False
    assert result.metadata["error_code"] == "invalid_response_contract"


@pytest.mark.parametrize(
    "params",
    [
        {
            "approval_id": "not-a-uuid",
            "scheduled_for": SCHEDULED_FOR,
            "text": REMINDER_TEXT,
            "idempotency_key": IDEMPOTENCY_KEY,
        },
        {
            "approval_id": APPROVAL_ID,
            "scheduled_for": "2026-07-29T15:00:00",
            "text": REMINDER_TEXT,
            "idempotency_key": IDEMPOTENCY_KEY,
        },
        {
            "approval_id": APPROVAL_ID,
            "scheduled_for": SCHEDULED_FOR,
            "text": "\x00\t",
            "idempotency_key": IDEMPOTENCY_KEY,
        },
        {
            "approval_id": APPROVAL_ID,
            "scheduled_for": SCHEDULED_FOR,
            "text": REMINDER_TEXT,
            "idempotency_key": "short",
        },
        {
            "approval_id": APPROVAL_ID,
            "scheduled_for": SCHEDULED_FOR,
            "text": REMINDER_TEXT,
            "idempotency_key": IDEMPOTENCY_KEY,
            "dry_run": False,
        },
    ],
)
def test_agenda_reminder_plan_rejects_unbounded_or_invalid_arguments(
    params: dict[str, Any],
) -> None:
    tool = SkynetAgendaReminderPlanTool(
        environ=_environment("SKYNET_JARVIS_REMINDER_PLAN"),
        transport=_transport(
            lambda request: pytest.fail(f"network called: {request.url}")
        ),
    )

    result = tool.execute(**params)

    assert result.success is False
    assert result.metadata["error_code"] == "invalid_arguments"


def test_agenda_reminder_plan_requires_its_dedicated_credential() -> None:
    environ = _environment("SKYNET_JARVIS_CASA_ACTION")
    tool = SkynetAgendaReminderPlanTool(
        environ=environ,
        transport=_transport(
            lambda request: pytest.fail(f"network called: {request.url}")
        ),
    )

    result = tool.execute(
        approval_id=APPROVAL_ID,
        scheduled_for=SCHEDULED_FOR,
        text=REMINDER_TEXT,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result.success is False
    assert result.metadata["error_code"] == "unsafe_configuration"


def test_fleet_read_allows_only_the_canonical_four_host_projection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/agent/jarvis/frota"
        assert request.url.params["host"] == "pc-casa"
        return _success(
            "jarvis.frota.snapshot.v1",
            **_fleet_snapshot(),
        )

    tool = SkynetFrotaReadTool(
        environ=_environment("SKYNET_JARVIS_FROTA_READ"),
        transport=_transport(handler),
    )
    result = tool.execute(host="pc-casa")

    assert result.success is True
    assert json.loads(result.content)["fleet"]["machines"][0]["host"] == "pc-casa"


def test_request_action_can_only_create_pending_fixed_target_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/agent/jarvis/casa/action/approval"
        assert json.loads(request.content) == {
            "action": "casa.definir_estado",
            "args": {"desired_state": "off"},
            "idempotency_key": IDEMPOTENCY_KEY,
            "target_alias": "tomada_cozinha_backlight",
        }
        return _success(
            "jarvis.casa.approval.v1",
            **_approval_payload(),
        )

    tool = SkynetCasaRequestActionTool(
        environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
        transport=_transport(handler),
    )
    result = tool.execute(
        desired_state="off",
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result.success is True
    assert json.loads(result.content)["status"] == "pending"
    assert tool.spec.requires_confirmation is False


def test_action_status_polls_one_exact_uuid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/agent/jarvis/casa/action/approval"
        assert dict(request.url.params) == {"approval_id": APPROVAL_ID}
        return _success(
            "jarvis.casa.approval.v1",
            **_approval_payload(desired_state="on", status="approved"),
        )

    tool = SkynetCasaActionStatusTool(
        environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
        transport=_transport(handler),
    )
    result = tool.execute(approval_id=APPROVAL_ID)

    assert result.success is True
    assert json.loads(result.content)["status"] == "approved"


def test_execute_action_sends_matching_approval_and_never_dry_run() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/agent/jarvis/casa/action"
        assert json.loads(request.content) == {
            "action": "casa.definir_estado",
            "approval_id": APPROVAL_ID,
            "args": {"desired_state": "on"},
            "dry_run": False,
            "idempotency_key": IDEMPOTENCY_KEY,
            "target_alias": "tomada_cozinha_backlight",
        }
        return _success(
            "jarvis.casa.action.receipt.v1",
            **_action_receipt(),
        )

    tool = SkynetCasaExecuteActionTool(
        environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
        transport=_transport(handler),
    )
    result = tool.execute(
        approval_id=APPROVAL_ID,
        desired_state="on",
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result.success is True
    assert json.loads(result.content)["observed_state"] == "on"
    assert tool.spec.requires_confirmation is False
    assert tool.spec.required_capabilities == [
        "network:fetch",
        "skynet:casa:write",
    ]


@pytest.mark.parametrize(
    ("tool", "params", "schema", "payload"),
    [
        (
            SkynetCasaReadTool(
                environ=_environment("SKYNET_JARVIS_CASA_READ"),
            ),
            {"alias": "luzes", "operation": "estado"},
            "jarvis.casa.snapshot.v1",
            _casa_snapshot(alias="tomadas"),
        ),
        (
            SkynetAgendaReadTool(
                environ=_environment("SKYNET_JARVIS_AGENDA_READ"),
            ),
            {"window": "upcoming", "days": 3, "limit": 4},
            "jarvis.agenda.snapshot.v1",
            _agenda_snapshot(window="today", days=1),
        ),
        (
            SkynetAgendaReminderPlanTool(
                environ=_environment("SKYNET_JARVIS_REMINDER_PLAN"),
            ),
            {
                "approval_id": APPROVAL_ID,
                "scheduled_for": SCHEDULED_FOR,
                "text": REMINDER_TEXT,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
            "jarvis.agenda.reminder.plan.v1",
            {
                **_reminder_plan(),
                "plan": {
                    **_reminder_plan()["plan"],
                    "scheduled_for": "2026-07-29T20:00:00.000Z",
                },
            },
        ),
        (
            SkynetFrotaReadTool(
                environ=_environment("SKYNET_JARVIS_FROTA_READ"),
            ),
            {"host": "pc-casa"},
            "jarvis.frota.snapshot.v1",
            _fleet_snapshot(host="pb7"),
        ),
        (
            SkynetCasaRequestActionTool(
                environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
            ),
            {
                "desired_state": "off",
                "idempotency_key": IDEMPOTENCY_KEY,
            },
            "jarvis.casa.approval.v1",
            {
                **_approval_payload(),
                "target_alias": "outra_tomada",
            },
        ),
        (
            SkynetCasaActionStatusTool(
                environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
            ),
            {"approval_id": APPROVAL_ID},
            "jarvis.casa.approval.v1",
            _approval_payload(
                approval_id="11111111-1111-4111-8111-111111111111",
            ),
        ),
        (
            SkynetCasaExecuteActionTool(
                environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
            ),
            {
                "approval_id": APPROVAL_ID,
                "desired_state": "on",
                "idempotency_key": IDEMPOTENCY_KEY,
            },
            "jarvis.casa.action.receipt.v1",
            _action_receipt(observed_state="off"),
        ),
    ],
)
def test_semantically_mismatched_success_responses_fail_closed(
    tool: Any,
    params: dict[str, Any],
    schema: str,
    payload: dict[str, Any],
) -> None:
    tool._transport = _transport(
        lambda request: _success(schema, **payload),
    )

    result = tool.execute(**params)

    assert result.success is False
    assert result.metadata["error_code"] == "invalid_response_contract"


@pytest.mark.parametrize(
    ("tool", "valid_params"),
    [
        (
            SkynetCasaReadTool(environ=_environment("SKYNET_JARVIS_CASA_READ")),
            {"alias": "casa", "operation": "resumo"},
        ),
        (
            SkynetAgendaReadTool(environ=_environment("SKYNET_JARVIS_AGENDA_READ")),
            {},
        ),
        (
            SkynetAgendaReminderPlanTool(
                environ=_environment("SKYNET_JARVIS_REMINDER_PLAN")
            ),
            {
                "approval_id": APPROVAL_ID,
                "scheduled_for": SCHEDULED_FOR,
                "text": REMINDER_TEXT,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        ),
        (
            SkynetFrotaReadTool(environ=_environment("SKYNET_JARVIS_FROTA_READ")),
            {"host": "all"},
        ),
        (
            SkynetCasaRequestActionTool(
                environ=_environment("SKYNET_JARVIS_CASA_ACTION")
            ),
            {
                "desired_state": "on",
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        ),
        (
            SkynetCasaActionStatusTool(
                environ=_environment("SKYNET_JARVIS_CASA_ACTION")
            ),
            {"approval_id": APPROVAL_ID},
        ),
        (
            SkynetCasaExecuteActionTool(
                environ=_environment("SKYNET_JARVIS_CASA_ACTION")
            ),
            {
                "approval_id": APPROVAL_ID,
                "desired_state": "on",
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        ),
    ],
)
@pytest.mark.parametrize("forbidden", ["url", "headers", "entity_id", "action"])
def test_arbitrary_boundary_parameters_are_rejected(
    tool: Any,
    valid_params: dict[str, Any],
    forbidden: str,
) -> None:
    result = tool.execute(**valid_params, **{forbidden: "https://attacker.test"})

    assert result.success is False
    assert result.metadata["error_code"] == "invalid_arguments"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: env.pop("SKYNET_JARVIS_CASA_READ_TOKEN"),
        lambda env: env.update(
            {"SKYNET_JARVIS_API_BASE_URL": "http://skynet.example.test"}
        ),
        lambda env: env.update(
            {"SKYNET_JARVIS_API_BASE_URL": ("https://skynet.example.test/path")}
        ),
        lambda env: env.update(
            {"SKYNET_JARVIS_API_BASE_URL": ("https://user@skynet.example.test")}
        ),
        lambda env: env.update({"SKYNET_JARVIS_CASA_READ_CHANNEL": "skynet"}),
    ],
)
def test_unsafe_configuration_fails_closed(
    mutate: Callable[[dict[str, str]], Any],
) -> None:
    environ = _environment("SKYNET_JARVIS_CASA_READ")
    mutate(environ)
    tool = SkynetCasaReadTool(
        environ=environ,
        transport=_transport(
            lambda request: pytest.fail(f"network called: {request.url}")
        ),
    )

    result = tool.execute(alias="casa", operation="resumo")

    assert result.success is False
    assert result.metadata["error_code"] == "unsafe_configuration"
    assert TOKEN not in result.content


def test_redirect_is_denied_without_following_location() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            307,
            headers={"location": "https://attacker.test/collect"},
        )

    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(handler),
    )
    result = tool.execute(alias="casa", operation="resumo")

    assert result.success is False
    assert result.metadata["error_code"] == "redirect_denied"
    assert calls == [
        (
            "https://skynet.example.test/api/agent/jarvis/casa"
            "?alias=casa&operation=resumo"
        )
    ]


def test_oversized_response_fails_instead_of_returning_partial_json() -> None:
    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"{" + (b"x" * 65_536) + b"}",
            )
        ),
    )

    result = tool.execute(alias="casa", operation="resumo")

    assert result.success is False
    assert result.metadata["error_code"] == "response_too_large"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="not json",
        ),
        httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"ok":true,"schema":"jarvis.casa.snapshot.v1","schema":"wrong"}',
        ),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(
            200,
            json={"ok": True, "schema": "untrusted.schema.v1"},
        ),
    ],
)
def test_invalid_response_contract_fails_closed(response: httpx.Response) -> None:
    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(lambda request: response),
    )

    result = tool.execute(alias="casa", operation="resumo")

    assert result.success is False
    assert result.content == "Skynet request failed safely."


def test_remote_error_body_and_secrets_are_never_reflected() -> None:
    leaked = f"remote debug {TOKEN} air-openjarvis {SESSION_ID}"
    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(
            lambda request: httpx.Response(
                500,
                json={"ok": False, "error": leaked},
            )
        ),
    )

    result = tool.execute(alias="casa", operation="resumo")

    assert result.success is False
    assert leaked not in result.content
    assert TOKEN not in result.content
    assert "air-openjarvis" not in result.content
    assert SESSION_ID not in result.content


def test_success_payload_is_recursively_stripped_and_redacted() -> None:
    payload = {
        "ok": True,
        "schema": "jarvis.casa.snapshot.v1",
        **_casa_snapshot(alias="casa", operation="resumo"),
        "requestId": "server-request",
        "nested": {
            "device_id": "air-openjarvis",
            "authorization": f"Bearer {TOKEN}",
            "safe": f"value reflected {TOKEN} {SESSION_ID}",
        },
    }
    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(lambda request: httpx.Response(200, json=payload)),
    )

    result = tool.execute(alias="casa", operation="resumo")

    assert result.success is True
    content = json.loads(result.content)
    assert "requestId" not in content
    assert "device_id" not in content["nested"]
    assert "authorization" not in content["nested"]
    assert TOKEN not in result.content
    assert SESSION_ID not in result.content
    assert content["nested"]["safe"] == "value reflected [redacted] [redacted]"


def test_execute_network_failure_marks_outcome_unknown_for_reconciliation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late response", request=request)

    tool = SkynetCasaExecuteActionTool(
        environ=_environment("SKYNET_JARVIS_CASA_ACTION"),
        transport=_transport(handler),
    )
    result = tool.execute(
        approval_id=APPROVAL_ID,
        desired_state="on",
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result.success is False
    assert result.metadata["outcome"] == "unknown"
    assert result.metadata["reconcile_required"] is True
    assert TOKEN not in result.content


def test_specs_are_closed_and_capability_scoped() -> None:
    tools = [
        SkynetCasaReadTool(),
        SkynetAgendaReadTool(),
        SkynetAgendaReminderPlanTool(),
        SkynetFrotaReadTool(),
        SkynetCasaRequestActionTool(),
        SkynetCasaActionStatusTool(),
        SkynetCasaExecuteActionTool(),
    ]

    for tool in tools:
        properties = tool.spec.parameters["properties"]
        assert tool.spec.parameters["additionalProperties"] is False
        assert not {"url", "headers", "entity_id", "action"} & set(properties)
        assert tool.is_local is False
        assert "network:fetch" in tool.spec.required_capabilities
        assert len(tool.spec.required_capabilities) == 2
