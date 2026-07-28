"""Security guardrails — scanners, engine wrapper, audit, SSRF."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from openjarvis.core.events import EventBus
from openjarvis.security._stubs import BaseScanner
from openjarvis.security.audit import AuditLogger
from openjarvis.security.file_policy import (
    DEFAULT_SENSITIVE_PATTERNS,
    filter_sensitive_paths,
    is_sensitive_file,
)
from openjarvis.security.guardrails import GuardrailsEngine, SecurityBlockError
from openjarvis.security.scanner import PIIScanner, SecretScanner
from openjarvis.security.ssrf import check_ssrf, is_private_ip
from openjarvis.security.types import (
    RedactionMode,
    ScanFinding,
    ScanResult,
    SecurityEvent,
    SecurityEventType,
    ThreatLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityContext:
    """Result of setup_security() — wrapped engine, policy, audit."""

    engine: Any
    capability_policy: Any = None
    audit_logger: Any = None
    boundary_guard: Any = None
    scanners: tuple[BaseScanner, ...] = ()


class _RegexFallbackScanner(BaseScanner):
    """Pure-Python scanner used when the optional Rust module is unavailable."""

    def __init__(
        self,
        scanner_id: str,
        patterns: Iterable[tuple[str, tuple[str, ThreatLevel, str]]],
    ) -> None:
        self.scanner_id = scanner_id
        self._patterns = [
            (name, re.compile(pattern, re.IGNORECASE), threat, description)
            for name, (pattern, threat, description) in patterns
        ]

    def scan(self, text: str) -> ScanResult:
        findings: list[ScanFinding] = []
        for name, pattern, threat, description in self._patterns:
            for match in pattern.finditer(text):
                findings.append(
                    ScanFinding(
                        pattern_name=name,
                        # Never retain the matched secret/PII in scanner output;
                        # audit logs only need the location and classification.
                        matched_text="[REDACTED]",
                        threat_level=threat,
                        start=match.start(),
                        end=match.end(),
                        description=description,
                    )
                )
        return ScanResult(findings=findings)

    def redact(self, text: str) -> str:
        redacted = text
        for name, pattern, _threat, _description in self._patterns:
            redacted = pattern.sub(f"[REDACTED:{name}]", redacted)
        return redacted


def _build_scanners(config: Any) -> list[BaseScanner]:
    scanners: list[BaseScanner] = []
    requested: list[tuple[type[BaseScanner], str]] = []
    if config.security.secret_scanner:
        requested.append((SecretScanner, "secrets"))
    if config.security.pii_scanner:
        requested.append((PIIScanner, "pii"))
    if not requested:
        raise RuntimeError(
            "Security is enabled but every scanner is disabled; refusing "
            "to create an unaudited execution boundary."
        )

    for scanner_cls, scanner_id in requested:
        try:
            scanners.append(scanner_cls())
        except Exception as exc:
            pattern_source = (
                SecretScanner.PATTERNS
                if scanner_cls is SecretScanner
                else PIIScanner.PATTERNS
            )
            logger.warning(
                "%s scanner acceleration unavailable (%s); using mandatory "
                "pure-Python fallback",
                scanner_id,
                type(exc).__name__,
            )
            scanners.append(
                _RegexFallbackScanner(
                    scanner_id,
                    pattern_source.items(),
                )
            )
    return scanners


def setup_capability_policy(config: Any) -> Any:
    """Build the capability policy used by every execution boundary.

    A policy object is returned even when the broader security subsystem is
    explicitly disabled.  Callers can therefore distinguish an intentional
    disabled policy from missing security wiring; an explicit agent identity
    is still required by ``ToolExecutor`` in both modes.
    """
    from openjarvis.security.capabilities import CapabilityPolicy

    if not config.security.enabled:
        return CapabilityPolicy(
            default_deny=False,
            enabled=False,
            enforce_tool_confirmation=False,
        )

    # Capability enforcement is mandatory whenever security is enabled.  The
    # old subsystem-level switch is retained for config compatibility, but can
    # no longer create a policy=None fail-open state.
    if not config.security.capabilities.enabled:
        logger.warning(
            "security.capabilities.enabled=false is ignored while security is "
            "enabled; set security.enabled=false to explicitly disable security"
        )
    if not config.security.capabilities.default_deny:
        logger.warning(
            "security.capabilities.default_deny=false is ignored while security "
            "is enabled"
        )
    return CapabilityPolicy(
        policy_path=config.security.capabilities.policy_path or None,
        default_deny=True,
        enabled=True,
        enforce_tool_confirmation=config.security.enforce_tool_confirmation,
    )


def setup_security(
    config: Any,
    engine: Any,
    bus: Optional[EventBus] = None,
) -> SecurityContext:
    """Apply security guardrails to an engine based on config.

    Returns a SecurityContext. Scanner/audit setup no-ops when
    ``config.security.enabled`` is false, while an explicitly disabled
    capability policy is still returned so downstream executors can
    distinguish that mode from a broken or missing policy.
    """
    cap_policy = setup_capability_policy(config)
    if not config.security.enabled:
        return SecurityContext(
            engine=engine,
            capability_policy=cap_policy,
        )

    scanners = _build_scanners(config)
    mode = RedactionMode(config.security.mode)
    engine = GuardrailsEngine(
        engine,
        scanners=scanners,
        mode=mode,
        scan_input=config.security.scan_input,
        scan_output=config.security.scan_output,
        bus=bus,
    )

    # Audit is a mandatory part of the security boundary.  Starting a
    # mutation-capable runtime without receipts is worse than refusing to
    # start, so setup errors propagate as an explicit fail-closed condition.
    try:
        audit = AuditLogger(
            db_path=config.security.audit_log_path,
            bus=bus,
        )
    except Exception as exc:
        raise RuntimeError(
            "Security audit logger unavailable; refusing to start."
        ) from exc

    from openjarvis.security.boundary import BoundaryGuard

    boundary_mode = config.security.mode
    if boundary_mode == "warn":
        logger.warning(
            "BoundaryGuard cannot use observational warn mode; outbound "
            "content will be redacted."
        )
        boundary_mode = "redact"
    boundary_guard = BoundaryGuard(
        mode=boundary_mode,
        enabled=True,
        bus=bus,
        scanners=scanners,
    )

    return SecurityContext(
        engine=engine,
        capability_policy=cap_policy,
        audit_logger=audit,
        boundary_guard=boundary_guard,
        scanners=tuple(scanners),
    )


__all__ = [
    "AuditLogger",
    "BaseScanner",
    "DEFAULT_SENSITIVE_PATTERNS",
    "GuardrailsEngine",
    "PIIScanner",
    "RedactionMode",
    "ScanFinding",
    "ScanResult",
    "SecretScanner",
    "SecurityBlockError",
    "SecurityContext",
    "SecurityEvent",
    "SecurityEventType",
    "ThreatLevel",
    "check_ssrf",
    "filter_sensitive_paths",
    "is_private_ip",
    "is_sensitive_file",
    "setup_capability_policy",
    "setup_security",
]
