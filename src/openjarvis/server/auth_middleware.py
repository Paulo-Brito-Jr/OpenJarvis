"""API key authentication middleware for the OpenJarvis server."""

from __future__ import annotations

import fnmatch
import logging
import os
import secrets
from collections.abc import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Validates ``Authorization: Bearer <key>`` on ``/v1/*`` and ``/api/*`` routes.

    Webhook routes and health checks are exempt — they use
    per-channel signature verification instead.
    """

    def __init__(
        self,
        app,  # noqa: ANN001
        api_key: str = "",
        *,
        principal: str = "",
        allowed_principals: Iterable[str] = (),
        capability_policy=None,  # noqa: ANN001
    ) -> None:
        super().__init__(app)
        self._api_key = api_key or os.environ.get("OPENJARVIS_API_KEY", "")
        self._principal = principal.strip()
        self._allowed_principals = frozenset(
            value.strip() for value in allowed_principals if value.strip()
        )
        self._capability_policy = capability_policy
        if self._api_key and not self._principal:
            raise RuntimeError(
                "An explicit API principal is required when API authentication "
                "is enabled."
            )
        if self._api_key and (
            not self._allowed_principals
            or self._principal not in self._allowed_principals
        ):
            raise RuntimeError(
                "The API principal must be present in the explicit API "
                "principal allowlist."
            )

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        if self._requires_auth(request.url.path):
            if not self._api_key:
                return JSONResponse(
                    {"detail": "API authentication is not configured"},
                    status_code=503,
                )
            auth = request.headers.get("Authorization", "")
            if not auth:
                return JSONResponse(
                    {"detail": "Missing Authorization header"},
                    status_code=401,
                )
            scheme, _, token = auth.partition(" ")
            # Constant-time comparison to avoid leaking the key via timing.
            if scheme.lower() != "bearer" or not secrets.compare_digest(
                token, self._api_key
            ):
                return JSONResponse(
                    {"detail": "Invalid API key"},
                    status_code=401,
                )
            # Downstream execution must use the authenticated principal rather
            # than a route-level constant or user-controlled request field.
            request.state.api_principal = self._principal

            capability = self._required_capability(
                request.method,
                request.url.path,
            )
            if capability:
                if self._capability_policy is None:
                    return JSONResponse(
                        {"detail": "Capability policy is not configured"},
                        status_code=503,
                    )
                allowed = explicitly_authorized(
                    self._capability_policy,
                    self._principal,
                    capability,
                    request.url.path,
                )
                if not allowed:
                    return JSONResponse(
                        {"detail": "API principal is not authorized"},
                        status_code=403,
                    )
        return await call_next(request)

    @staticmethod
    def _requires_auth(path: str) -> bool:
        """Protect API routes and operational metrics; leave the UI/health open.

        ``/metrics`` exposes request/token counters that should not be readable
        by unauthenticated clients, so it is gated alongside ``/v1`` and
        ``/api``. ``/health`` stays open for liveness probes.
        """
        if (
            path.startswith("/v1/connectors/")
            and path.endswith("/oauth/callback")
        ):
            # OAuth providers cannot attach the API bearer token.  This exact
            # callback route authenticates a one-time state value instead.
            return False
        return (
            path.startswith("/v1/")
            or path.startswith("/api/")
            or path == "/metrics"
            or path.startswith("/metrics/")
        )

    @staticmethod
    def _required_capability(method: str, path: str) -> str:
        """Return the explicit capability required by a protected API route.

        Every mutating ``/v1`` or ``/api`` route gets a capability.  Unknown
        mutations fall back to ``system:admin`` so adding a new route cannot
        silently bypass authorization.  Sensitive reads are also classified.
        """
        normalized_method = method.upper()
        if normalized_method in {"GET", "HEAD"}:
            sensitive_reads = (
                ("/v1/memory", "memory:read"),
                ("/v1/traces", "memory:read"),
                ("/v1/sessions", "memory:read"),
                ("/v1/approvals", "approval:decide"),
                ("/v1/managed-agents", "system:admin"),
                ("/v1/agents", "system:admin"),
                ("/v1/connectors", "system:admin"),
            )
            for prefix, capability in sensitive_reads:
                if path == prefix or path.startswith(f"{prefix}/"):
                    return capability
            return ""

        if normalized_method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return ""

        if path == "/v1/memory/search":
            return "memory:read"
        if path.startswith("/v1/memory/"):
            return "memory:write"
        if path.startswith("/v1/approvals/"):
            return "approval:decide"
        if path == "/v1/channels/send":
            return "channel:send"
        if path.endswith("/message") or path.endswith("/messages"):
            return "message:send"
        if path == "/api/digest/schedule":
            return "schedule:create"
        if path in {
            "/v1/chat/completions",
            "/api/digest/generate",
            "/api/research",
            "/v1/speech/transcribe",
        }:
            return "tool:invoke"

        # Fail closed for every remaining mutation, including model lifecycle,
        # connectors, uploads, managed agents/templates/tools, telemetry,
        # feedback, optimization and future routes not yet classified.
        return "system:admin"


def generate_api_key() -> str:
    """Generate a new API key with ``oj_sk_`` prefix."""
    return f"oj_sk_{secrets.token_urlsafe(32)}"


def check_bind_safety(host: str, *, api_key: str) -> None:
    """Refuse to bind non-loopback without an API key.

    Raises ``SystemExit`` if *host* is not a loopback address and
    *api_key* is empty.
    """
    import ipaddress
    import sys

    try:
        is_loop = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loop = host in ("localhost", "")

    if not is_loop and not api_key:
        logger.error(
            "Binding to %s requires OPENJARVIS_API_KEY to be set. "
            "Run: jarvis auth generate-key",
            host,
        )
        sys.exit(1)


def websocket_authorized(
    websocket,  # noqa: ANN001
    expected_key: str | None,
    *,
    principal: str = "",
    allowed_principals: Iterable[str] = (),
) -> bool:
    """Return ``True`` if a WebSocket connection presents the expected key.

    ``AuthMiddleware`` is a ``BaseHTTPMiddleware`` and never sees WebSocket
    upgrade requests, so streaming endpoints must check the token themselves
    in the handshake before calling ``websocket.accept()``.

    ``None`` means an explicitly embedded/trusted application.  An empty
    string means production authentication was requested but not configured
    and therefore denies the handshake. The token may be supplied either as a
    ``?token=`` query
    parameter — browsers cannot set headers on a WebSocket handshake — or via
    an ``Authorization: Bearer <key>`` header for programmatic clients.
    """
    if expected_key is None:
        return True
    if not expected_key or not principal:
        return False
    allowlist = frozenset(
        value.strip() for value in allowed_principals if value.strip()
    )
    if principal not in allowlist:
        return False
    token = websocket.query_params.get("token", "")
    if not token:
        auth = websocket.headers.get("authorization", "")
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer":
            token = value
    if not token:
        return False
    if not secrets.compare_digest(token, expected_key):
        return False
    websocket.state.api_principal = principal
    return True


def websocket_capability_authorized(
    websocket,  # noqa: ANN001
    capability: str,
    resource: str,
) -> bool:
    """Authorize a WebSocket against the authenticated API principal.

    Token authentication alone is not an execution grant. Every WebSocket
    endpoint must call this helper with its explicit capability and resource
    before accepting the connection.
    """
    if not capability or not resource:
        return False
    websocket_state = getattr(websocket, "state", None)
    candidate = getattr(websocket_state, "api_principal", "")
    if not isinstance(candidate, str) or not candidate.strip():
        app = getattr(websocket, "app", None)
        app_state = getattr(app, "state", None)
        candidate = getattr(app_state, "api_principal", "")
    principal = candidate.strip() if isinstance(candidate, str) else ""
    app = getattr(websocket, "app", None)
    app_state = getattr(app, "state", None)
    capability_policy = getattr(app_state, "capability_policy", None)
    if not principal or capability_policy is None:
        return False
    if explicitly_authorized(
        capability_policy,
        principal,
        capability,
        resource,
    ):
        return True
    return False


def explicitly_authorized(
    capability_policy,  # noqa: ANN001
    principal: str,
    capability: str,
    resource: str,
) -> bool:
    """Require a real matching grant plus a successful policy decision.

    A policy configured with ``default_deny=False`` is not an authorization
    grant for an external request.  Server boundaries call this helper so a
    permissive fallback can never silently turn authentication into authority.
    """
    if (
        capability_policy is None
        or not isinstance(principal, str)
        or not principal.strip()
        or not capability
        or not resource
    ):
        return False
    try:
        grants = capability_policy.list_grants(principal.strip())
        explicitly_granted = any(
            fnmatch.fnmatchcase(capability, grant.capability)
            and (
                grant.pattern == "*"
                or fnmatch.fnmatchcase(resource, grant.pattern)
            )
            for grant in grants
        )
        if not explicitly_granted:
            return False
        return bool(
            capability_policy.check(
                principal.strip(),
                capability,
                resource,
            )
        )
    except Exception:
        logger.warning("Capability check failed; request denied")
        return False
