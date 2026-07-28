"""Tests for setup_security() helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openjarvis.core.config import CapabilitiesConfig, JarvisConfig, SecurityConfig
from openjarvis.core.events import EventBus
from openjarvis.security import SecurityContext, setup_security


def _make_mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {"content": "ok"}
    engine.list_models.return_value = ["m"]
    engine.health.return_value = True
    return engine


def _make_config(*, enabled: bool = True, caps_enabled: bool = False) -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.security = SecurityConfig(
        enabled=enabled,
        secret_scanner=True,
        pii_scanner=True,
        mode="warn",
        capabilities=CapabilitiesConfig(enabled=caps_enabled),
    )
    return cfg


def _has_rust() -> bool:
    try:
        import openjarvis_rust  # noqa: F401

        return True
    except ImportError:
        return False


class TestSetupSecurityEnabled:
    @pytest.mark.skipif(not _has_rust(), reason="Rust extension not compiled")
    def test_returns_wrapped_engine(self) -> None:
        from openjarvis.security.guardrails import GuardrailsEngine

        engine = _make_mock_engine()
        bus = EventBus()
        sec = setup_security(_make_config(), engine, bus)

        assert isinstance(sec.engine, GuardrailsEngine)
        assert sec.audit_logger is not None

    def test_returns_security_context(self) -> None:
        engine = _make_mock_engine()
        bus = EventBus()
        sec = setup_security(_make_config(), engine, bus)

        assert isinstance(sec, SecurityContext)
        assert sec.capability_policy is not None
        assert sec.capability_policy.enabled is True
        assert sec.capability_policy._default_deny is True
        # Audit logger should always work (no Rust dependency)
        assert sec.audit_logger is not None
        assert sec.boundary_guard is not None
        assert sec.scanners

    def test_graceful_without_rust(self) -> None:
        """Rust absence uses a real scanner rather than a no-op boundary."""
        engine = _make_mock_engine()
        bus = EventBus()
        sec = setup_security(_make_config(), engine, bus)

        assert isinstance(sec, SecurityContext)
        assert sec.scanners
        redacted = sec.boundary_guard.scan_outbound(
            "key sk-abc123def456ghi789jkl012",
            "external:test",
        )
        assert "sk-abc123" not in redacted

    def test_enabled_security_rejects_all_scanners_disabled(self) -> None:
        cfg = _make_config()
        cfg.security.secret_scanner = False
        cfg.security.pii_scanner = False
        with pytest.raises(RuntimeError, match="every scanner is disabled"):
            setup_security(cfg, _make_mock_engine(), EventBus())


class TestSetupSecurityDisabled:
    def test_returns_original_engine(self) -> None:
        engine = _make_mock_engine()
        sec = setup_security(_make_config(enabled=False), engine)

        assert sec.engine is engine
        assert sec.capability_policy is not None
        assert sec.capability_policy.enabled is False
        assert not sec.capability_policy.check("", "tool:invoke", "anything")
        assert sec.audit_logger is None
        assert sec.boundary_guard is None
