"""Defensive trace redaction and payload summarisation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from openjarvis.core.types import StepType, Trace, TraceStep
from openjarvis.security.taint import redact_sensitive_text

_MAX_TRACE_TEXT = 4_000
_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "set-cookie",
    "token",
}


def _safe_text(value: str) -> str:
    redacted = redact_sensitive_text(value)
    if len(redacted) > _MAX_TRACE_TEXT:
        return redacted[:_MAX_TRACE_TEXT] + "[TRUNCATED]"
    return redacted


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    for raw_part in _SENSITIVE_KEY_PARTS:
        part = raw_part.replace("-", "_")
        if (
            normalized == part
            or normalized.startswith(f"{part}_")
            or normalized.endswith(f"_{part}")
        ):
            return True
    return False


def sanitize_value(value: Any, *, key_hint: str = "") -> Any:
    """Return a JSON-safe value with secret-bearing fields redacted."""
    if _is_sensitive_key(key_hint):
        return "[REDACTED]"
    if isinstance(value, str):
        return _safe_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if key_hint.lower() in {"header", "headers"}:
            return {
                "header_names": sorted(
                    str(key).lower() for key in value if isinstance(key, str)
                ),
                "values": "[REDACTED]",
            }
        return {
            str(key): sanitize_value(item, key_hint=str(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [sanitize_value(item, key_hint=key_hint) for item in value[:100]]
    return _safe_text(str(value))


def summarize_payload(value: Any) -> dict[str, Any]:
    """Describe an argument/result payload without retaining its raw value."""
    if (
        isinstance(value, Mapping)
        and value.get("retained") is False
        and isinstance(value.get("sha256"), str)
        and isinstance(value.get("size_bytes"), int)
    ):
        return dict(value)
    sanitized = sanitize_value(value)
    encoded = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode()
    summary: dict[str, Any] = {
        "retained": False,
        "type": type(value).__name__,
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest()[:16],
    }
    if isinstance(value, Mapping):
        summary["keys"] = sorted(str(key) for key in value)[:50]
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        summary["items"] = len(value)
    return summary


def _sanitize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for call in tool_calls[:100]:
        if not isinstance(call, Mapping):
            sanitized.append({"payload": summarize_payload(call)})
            continue
        function = call.get("function")
        name = call.get("name", "")
        arguments = call.get("arguments")
        if isinstance(function, Mapping):
            name = function.get("name", name)
            arguments = function.get("arguments", arguments)
        sanitized.append(
            {
                "id": _safe_text(str(call.get("id", ""))),
                "name": _safe_text(str(name or "")),
                "arguments": summarize_payload(arguments),
            }
        )
    return sanitized


def sanitize_step(step: TraceStep) -> TraceStep:
    input_data = sanitize_value(step.input)
    output_data = sanitize_value(step.output)
    if step.step_type == StepType.TOOL_CALL:
        input_data = {
            "tool": _safe_text(str(step.input.get("tool", ""))),
            "arguments": summarize_payload(step.input.get("arguments")),
        }
        output_data = {
            "success": bool(step.output.get("success", False)),
            "result": summarize_payload(step.output.get("result")),
        }
    elif step.step_type == StepType.GENERATE:
        output_data = {
            key: sanitize_value(value, key_hint=key)
            for key, value in step.output.items()
            if key not in {"tool_calls", "tool_results", "content_blocks"}
        }
        output_data["tool_calls"] = _sanitize_tool_calls(
            step.output.get("tool_calls", [])
        )
        if step.output.get("tool_results") is not None:
            output_data["tool_results"] = summarize_payload(
                step.output.get("tool_results")
            )
        if step.output.get("content_blocks") is not None:
            output_data["content_blocks"] = summarize_payload(
                step.output.get("content_blocks")
            )
    return TraceStep(
        step_type=step.step_type,
        timestamp=step.timestamp,
        duration_seconds=step.duration_seconds,
        input=input_data if isinstance(input_data, dict) else {},
        output=output_data if isinstance(output_data, dict) else {},
        metadata=sanitize_value(step.metadata),
    )


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_messages: list[dict[str, Any]] = []
    for message in messages[:200]:
        if not isinstance(message, Mapping):
            continue
        role = _safe_text(str(message.get("role", "")))
        content = message.get("content", "")
        safe: dict[str, Any] = {"role": role}
        if role == "tool":
            safe["content"] = summarize_payload(content)
        else:
            safe["content"] = sanitize_value(content)
        if message.get("name"):
            safe["name"] = _safe_text(str(message["name"]))
        if message.get("tool_calls") is not None:
            safe["tool_calls"] = _sanitize_tool_calls(message["tool_calls"])
        safe_messages.append(safe)
    return safe_messages


def sanitize_trace(trace: Trace) -> Trace:
    """Return a sanitized trace suitable for memory and SQLite persistence."""
    return Trace(
        trace_id=trace.trace_id,
        query=_safe_text(trace.query),
        agent=_safe_text(trace.agent),
        model=_safe_text(trace.model),
        engine=_safe_text(trace.engine),
        steps=[sanitize_step(step) for step in trace.steps],
        result=_safe_text(trace.result),
        outcome=trace.outcome,
        feedback=trace.feedback,
        started_at=trace.started_at,
        ended_at=trace.ended_at,
        total_tokens=trace.total_tokens,
        total_latency_seconds=trace.total_latency_seconds,
        metadata=sanitize_value(trace.metadata),
        messages=_sanitize_messages(trace.messages),
    )


__all__ = [
    "sanitize_step",
    "sanitize_trace",
    "sanitize_value",
    "summarize_payload",
]
