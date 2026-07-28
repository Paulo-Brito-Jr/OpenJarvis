"""Fail-closed projections for security-sensitive server responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openjarvis.traces.redaction import redact_sensitive_text

_SAFE_AGENT_CONFIG_KEYS = frozenset(
    {
        "learning_enabled",
        "learning_schedule",
        "max_stall_retries",
        "max_tokens",
        "max_turns",
        "model",
        "schedule_type",
        "schedule_value",
        "temperature",
        "timeout_seconds",
        "tools",
    }
)
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_sensitive_text(value)[:512]
    return None


def _safe_tool_names(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        redact_sensitive_text(item)[:128]
        for item in value
        if isinstance(item, str) and item.strip()
    ][:128]


def project_agent_config(config: Any) -> dict[str, Any]:
    """Return only operational, non-secret managed-agent configuration."""
    if not isinstance(config, Mapping):
        return {}

    projected: dict[str, Any] = {}
    for raw_key, value in config.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        normalized = key.lower().replace("-", "_")
        if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
            continue
        if normalized == "system_prompt":
            projected["system_prompt_configured"] = bool(value)
            continue
        if normalized not in _SAFE_AGENT_CONFIG_KEYS:
            continue
        if normalized == "tools":
            projected[key] = _safe_tool_names(value)
            continue
        safe_value = _safe_scalar(value)
        if safe_value is not None:
            projected[key] = safe_value
    return projected


def project_managed_agent(agent: Any) -> dict[str, Any]:
    """Project a managed-agent row without returning arbitrary config data."""
    if not isinstance(agent, Mapping):
        return {}
    projected = dict(agent)
    projected["config"] = project_agent_config(agent.get("config"))
    for key in ("summary_memory", "current_activity", "name"):
        value = projected.get(key)
        if isinstance(value, str):
            projected[key] = redact_sensitive_text(value)[:512]
    return projected


def project_channel_binding(binding: Any) -> dict[str, Any]:
    """Project a channel binding without credentials, senders or session IDs."""
    if not isinstance(binding, Mapping):
        return {}
    config = binding.get("config")
    config_map = config if isinstance(config, Mapping) else {}
    allowed_senders = config_map.get("allowed_senders")
    sender_count = (
        len(allowed_senders)
        if isinstance(allowed_senders, (list, tuple, set, frozenset))
        else 0
    )
    has_credentials = any(
        isinstance(key, str)
        and any(
            fragment in key.lower().replace("-", "_")
            for fragment in _SECRET_KEY_FRAGMENTS
        )
        and bool(value)
        for key, value in config_map.items()
    )
    return {
        "id": binding.get("id", ""),
        "agent_id": binding.get("agent_id", ""),
        "channel_type": binding.get("channel_type", ""),
        "routing_mode": binding.get("routing_mode", "auto"),
        "config": {
            "configured": bool(config_map),
            "credentials_configured": has_credentials,
            "allowed_senders_configured": sender_count > 0,
            "allowed_sender_count": sender_count,
        },
    }
