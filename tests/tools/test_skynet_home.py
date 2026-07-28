"""Security and contract tests for the bounded Skynet tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from openjarvis.tools.skynet_home import (
    SkynetAgendaReadTool,
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
            casa={"alias": "luzes", "reachable": True},
        )

    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(handler),
    )
    result = tool.execute(alias="luzes", operation="estado")

    assert result.success is True
    assert json.loads(result.content)["casa"]["alias"] == "luzes"


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
            agenda={"events": [], "total": 0},
        )

    tool = SkynetAgendaReadTool(
        environ=_environment("SKYNET_JARVIS_AGENDA_READ"),
        transport=_transport(handler),
    )
    result = tool.execute(window="upcoming", days=3, limit=4)

    assert result.success is True


def test_fleet_read_allows_only_the_canonical_four_host_projection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/agent/jarvis/frota"
        assert request.url.params["host"] == "pc-casa"
        return _success(
            "jarvis.frota.snapshot.v1",
            fleet={"machines": [{"host": "pc-casa", "health": "unknown"}]},
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
            approval_id=APPROVAL_ID,
            status="pending",
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
            approval_id=APPROVAL_ID,
            status="approved",
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
            receipt_id="receipt-1",
            status="succeeded",
            observed_state="on",
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
    ("tool", "valid_params"),
    [
        (
            SkynetCasaReadTool(
                environ=_environment("SKYNET_JARVIS_CASA_READ")
            ),
            {"alias": "casa", "operation": "resumo"},
        ),
        (
            SkynetAgendaReadTool(
                environ=_environment("SKYNET_JARVIS_AGENDA_READ")
            ),
            {},
        ),
        (
            SkynetFrotaReadTool(
                environ=_environment("SKYNET_JARVIS_FROTA_READ")
            ),
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
            {
                "SKYNET_JARVIS_API_BASE_URL": (
                    "https://skynet.example.test/path"
                )
            }
        ),
        lambda env: env.update(
            {
                "SKYNET_JARVIS_API_BASE_URL": (
                    "https://user@skynet.example.test"
                )
            }
        ),
        lambda env: env.update(
            {"SKYNET_JARVIS_CASA_READ_CHANNEL": "skynet"}
        ),
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
            content=b'{"ok":true,"schema":"jarvis.casa.snapshot.v1",'
            b'"schema":"wrong"}',
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
        "requestId": "server-request",
        "nested": {
            "device_id": "air-openjarvis",
            "authorization": f"Bearer {TOKEN}",
            "safe": f"value reflected {TOKEN} {SESSION_ID}",
        },
    }
    tool = SkynetCasaReadTool(
        environ=_environment("SKYNET_JARVIS_CASA_READ"),
        transport=_transport(
            lambda request: httpx.Response(200, json=payload)
        ),
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
