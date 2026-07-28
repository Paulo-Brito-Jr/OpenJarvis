"""FastAPI application factory for the OpenJarvis API server."""

from __future__ import annotations

import logging
import pathlib
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from openjarvis.server.analytics_routes import router as analytics_router
from openjarvis.server.api_routes import include_all_routes
from openjarvis.server.comparison import comparison_router
from openjarvis.server.connectors_router import create_connectors_router
from openjarvis.server.dashboard import dashboard_router
from openjarvis.server.digest_routes import create_digest_router
from openjarvis.server.research_router import router as research_router
from openjarvis.server.routes import router
from openjarvis.server.upload_router import router as upload_router

logger = logging.getLogger(__name__)


def _restore_sendblue_bindings(app: FastAPI) -> None:
    """Keep legacy managed-channel bindings quarantined at startup.

    Old rows may contain plaintext credentials and lack an immutable operator
    identity.  Do not even enumerate them here: merely reading and reconnecting
    those rows would silently re-enable the confused-deputy path on restart.
    """
    from openjarvis.server.managed_channel_security import (
        quarantine_legacy_sendblue_bindings,
    )

    quarantine_legacy_sendblue_bindings(app)


# No-cache headers applied to static file responses
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles subclass that adds no-cache headers to every response."""

    async def __call__(self, scope, receive, send):
        async def _send_with_headers(message):
            if message["type"] == "http.response.start":
                extra = [(k.encode(), v.encode()) for k, v in _NO_CACHE_HEADERS.items()]
                # Remove etag and last-modified
                existing = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"etag", b"last-modified")
                ]
                message = {**message, "headers": existing + extra}
            await send(message)

        await super().__call__(scope, receive, _send_with_headers)


def create_app(
    engine,
    model: str,
    *,
    agent=None,
    request_agent_factory=None,
    bus=None,
    engine_name: str = "",
    agent_name: str = "",
    channel_bridge=None,
    config=None,
    memory_backend=None,
    memory_service=None,
    speech_backend=None,
    agent_manager=None,
    agent_scheduler=None,
    capability_policy=None,
    audit_logger=None,
    boundary_guard=None,
    api_key: str | None = None,
    api_principal: str = "",
    api_principal_allowlist: set[str] | None = None,
    webhook_config: dict | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    engine:
        The inference engine to use for completions.
    model:
        Default model name.
    agent:
        Optional agent instance for agent-mode completions.
    request_agent_factory:
        Callable that returns a fresh ``RequestAgentScope`` for an
        authenticated API principal. External requests never reuse ``agent``.
    bus:
        Optional event bus for telemetry.
    channel_bridge:
        Optional channel bridge for multi-platform messaging.
    config:
        Optional JarvisConfig for other settings.
    """
    app = FastAPI(
        title="OpenJarvis API",
        description="OpenAI-compatible API server for OpenJarvis",
        version="0.1.0",
    )

    from fastapi.middleware.cors import CORSMiddleware

    _origins = (
        cors_origins
        if cors_origins is not None
        else [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            # Tauri 2 production webview origins:
            #   macOS / Linux / iOS  -> tauri://localhost
            #   Windows / Android    -> http://tauri.localhost (default),
            #                           https://tauri.localhost when
            #                           windows.useHttpsScheme is enabled
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    from openjarvis.server.middleware import RequestBodyLimitMiddleware

    app.add_middleware(RequestBodyLimitMiddleware)

    # Store dependencies in app state
    app.state.engine = engine
    app.state.model = model
    app.state.agent = agent
    app.state.request_agent_factory = request_agent_factory
    app.state.bus = bus
    app.state.engine_name = engine_name
    # Do not promote a class-level agent label into an authorization
    # principal.  Callers must supply the runtime identity explicitly.
    app.state.agent_name = agent_name or ""
    app.state.channel_bridge = channel_bridge
    app.state.config = config
    app.state.memory_backend = memory_backend
    app.state.memory_service = memory_service
    app.state.speech_backend = speech_backend
    app.state.agent_manager = agent_manager
    app.state.agent_scheduler = agent_scheduler
    app.state.capability_policy = capability_policy
    app.state.audit_logger = audit_logger
    app.state.boundary_guard = boundary_guard
    app.state.session_start = time.time()
    # Exposed so WebSocket handlers can authenticate the handshake (the HTTP
    # AuthMiddleware never sees WS upgrade requests). ``None`` is reserved for
    # explicitly embedded/test applications; an empty production key denies.
    app.state.api_key = api_key
    app.state.api_principal = api_principal.strip()
    app.state.api_principal_allowlist = frozenset(
        value.strip() for value in (api_principal_allowlist or set()) if value.strip()
    )

    needs_security = agent is not None and bool(
        getattr(agent, "accepts_tools", False)
        or getattr(agent, "requires_security_context", False)
    )
    if needs_security:
        bind_security = getattr(agent, "bind_security", None)
        if not callable(bind_security):
            raise RuntimeError(
                "Security-bound server agent does not expose bind_security()"
            )
        bind_security(
            capability_policy,
            app.state.agent_name or "",
            boundary_guard,
        )

    # Wire up trace store if traces are enabled.
    #
    # We deliberately do NOT subscribe the trace store to the bus. The chat
    # endpoints persist through a TraceCollector that calls store.save()
    # directly (mirroring system/orchestrator.py), and the collector ALSO
    # publishes TRACE_COMPLETE. A store subscribed to that same bus would
    # therefore save every agent trace twice — the second INSERT hitting the
    # UNIQUE constraint on trace_id (a 500 on every completion). Keeping the
    # collector the single writer is what makes the dual code path safe; only
    # the telemetry store is bus-subscribed (see system/builder.py).
    app.state.trace_store = None
    try:
        from openjarvis.core.config import load_config
        from openjarvis.traces.store import TraceStore

        cfg = config if config is not None else load_config()
        if cfg.traces.enabled:
            app.state.trace_store = TraceStore(db_path=cfg.traces.db_path)
    except Exception:
        pass  # traces are optional; don't block server startup

    # Wire up external analytics if enabled (PostHog) — never block startup.
    # Note: we do NOT fire app_opened here. The frontend owns that event
    # because "server started" (this code path) is not the same as "user
    # opened the app" — the server can run headless via cron, daemons,
    # or test suites.
    app.state.analytics_client = None
    app.state.analytics_bridge = None
    try:
        from openjarvis.analytics import (
            AnalyticsClient,
            EventBridge,
            is_analytics_enabled,
        )
        from openjarvis.core.config import load_config

        _cfg = config if config is not None else load_config()
        if is_analytics_enabled(_cfg.analytics):
            _client = AnalyticsClient(_cfg.analytics)
            app.state.analytics_client = _client
            _bus_ref = getattr(app.state, "bus", None)
            if _bus_ref is not None:
                _bridge = EventBridge(_bus_ref, _client)
                _bridge.start()
                app.state.analytics_bridge = _bridge

            @app.on_event("shutdown")
            async def _shutdown_analytics() -> None:
                bridge = getattr(app.state, "analytics_bridge", None)
                if bridge is not None:
                    try:
                        bridge.stop()
                    except Exception:
                        pass
                client = getattr(app.state, "analytics_client", None)
                if client is not None:
                    try:
                        client.shutdown()
                    except Exception:
                        pass
    except Exception as _exc:
        logger.debug("Analytics init skipped: %s", _exc)

    # Stop the background memory service cleanly when the server shuts down.
    if memory_service is not None:

        @app.on_event("shutdown")
        async def _shutdown_memory_service() -> None:
            svc = getattr(app.state, "memory_service", None)
            if svc is not None:
                try:
                    svc.stop()
                except Exception:
                    pass

    app.include_router(router)
    app.include_router(dashboard_router)
    app.include_router(comparison_router)
    app.include_router(create_connectors_router())
    app.include_router(create_digest_router())
    app.include_router(upload_router)
    app.include_router(research_router)
    app.include_router(analytics_router)
    include_all_routes(app)

    # Restore SendBlue channel bindings from database on startup
    _restore_sendblue_bindings(app)

    # Add security headers middleware
    try:
        from openjarvis.server.middleware import create_security_middleware

        middleware_cls = create_security_middleware()
        if middleware_cls is not None:
            app.add_middleware(middleware_cls)
    except Exception as exc:
        logger.debug("Security middleware init skipped: %s", exc)

    # API key authentication middleware. ``api_key is not None`` means the
    # caller selected server mode, so an empty key must deny rather than
    # silently disabling authentication.
    if api_key is not None:
        from openjarvis.server.auth_middleware import AuthMiddleware

        app.add_middleware(
            AuthMiddleware,
            api_key=api_key,
            principal=app.state.api_principal,
            allowed_principals=app.state.api_principal_allowlist,
            capability_policy=capability_policy,
        )

    # Mount webhook routes (always — SendBlue may be configured dynamically)
    if webhook_config:
        try:
            from openjarvis.server.webhook_routes import (
                create_webhook_router,
            )

            webhook_router = create_webhook_router(
                bridge=channel_bridge,
                twilio_auth_token=webhook_config.get("twilio_auth_token", ""),
                bluebubbles_password=webhook_config.get("bluebubbles_password", ""),
                whatsapp_verify_token=webhook_config.get("whatsapp_verify_token", ""),
                whatsapp_app_secret=webhook_config.get("whatsapp_app_secret", ""),
            )
            app.include_router(webhook_router)
        except Exception as exc:
            logger.debug("Webhook routes init skipped: %s", exc)

    # Serve static frontend assets if the static/ directory exists
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.is_dir():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                _NoCacheStaticFiles(directory=assets_dir),
                name="static-assets",
            )

        @app.get("/{full_path:path}")
        async def spa_catch_all(full_path: str):
            """Serve static files directly, fall back to index.html for SPA routes."""
            if full_path:
                candidate = (static_dir / full_path).resolve()
                # Path traversal prevention
                resolved_root = static_dir.resolve()
                if candidate.is_relative_to(resolved_root) and candidate.is_file():
                    return FileResponse(candidate, headers=_NO_CACHE_HEADERS)
            return FileResponse(
                static_dir / "index.html",
                headers=_NO_CACHE_HEADERS,
            )

    return app


__all__ = ["create_app"]
