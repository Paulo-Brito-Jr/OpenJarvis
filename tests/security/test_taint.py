"""Tests for taint tracking system (Phase 14.5)."""

from __future__ import annotations

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.security.taint import (
    SINK_POLICY,
    TaintLabel,
    TaintSet,
    auto_detect_taint,
    check_taint,
    declassify,
    external_taint,
    propagate_taint,
    redact_sensitive_text,
)
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


class TestTaintSet:
    def test_empty_taint(self):
        ts = TaintSet()
        assert not ts
        assert not ts.labels

    def test_from_labels(self):
        ts = TaintSet.from_labels(TaintLabel.PII, TaintLabel.SECRET)
        assert ts.has(TaintLabel.PII)
        assert ts.has(TaintLabel.SECRET)
        assert not ts.has(TaintLabel.EXTERNAL)

    def test_union(self):
        a = TaintSet.from_labels(TaintLabel.PII)
        b = TaintSet.from_labels(TaintLabel.SECRET)
        merged = a.union(b)
        assert merged.has(TaintLabel.PII)
        assert merged.has(TaintLabel.SECRET)

    def test_frozen(self):
        ts = TaintSet.from_labels(TaintLabel.PII)
        # TaintSet is frozen dataclass
        with pytest.raises(AttributeError):
            ts.labels = frozenset()

    def test_bool_true_when_has_labels(self):
        ts = TaintSet.from_labels(TaintLabel.PII)
        assert bool(ts)

    def test_bool_false_when_empty(self):
        ts = TaintSet()
        assert not bool(ts)

    def test_json_roundtrip_is_stable(self):
        ts = TaintSet.from_labels(TaintLabel.SECRET, TaintLabel.EXTERNAL)
        encoded = ts.to_json()
        assert encoded == {"labels": ["external", "secret"]}
        assert TaintSet.from_json(encoded) == ts

    @pytest.mark.parametrize(
        "invalid",
        [
            "external",
            {"labels": "external"},
            {"labels": ["unknown"]},
            {"labels": ["external"], "trusted": True},
            {"other": []},
        ],
    )
    def test_json_parser_rejects_ambiguous_or_unknown_payloads(self, invalid):
        with pytest.raises(ValueError):
            TaintSet.from_json(invalid)


class TestCheckTaint:
    def test_clean_data_passes(self):
        ts = TaintSet()
        assert check_taint("web_search", ts) is None

    def test_pii_blocked_for_web_search(self):
        ts = TaintSet.from_labels(TaintLabel.PII)
        result = check_taint("web_search", ts)
        assert result is not None
        assert "pii" in result.lower()

    def test_secret_blocked_for_web_search(self):
        ts = TaintSet.from_labels(TaintLabel.SECRET)
        result = check_taint("web_search", ts)
        assert result is not None
        assert "secret" in result.lower()

    def test_secret_blocked_for_channel_send(self):
        ts = TaintSet.from_labels(TaintLabel.SECRET)
        result = check_taint("channel_send", ts)
        assert result is not None

    def test_external_allowed_for_web_search(self):
        ts = TaintSet.from_labels(TaintLabel.EXTERNAL)
        assert check_taint("web_search", ts) is None

    def test_unknown_tool_allowed(self):
        ts = TaintSet.from_labels(TaintLabel.PII, TaintLabel.SECRET)
        assert check_taint("calculator", ts) is None

    def test_sink_policy_has_expected_tools(self):
        assert "web_search" in SINK_POLICY
        assert "channel_send" in SINK_POLICY
        assert "code_interpreter" in SINK_POLICY


class TestDeclassify:
    def test_remove_label(self):
        ts = TaintSet.from_labels(TaintLabel.PII, TaintLabel.SECRET)
        result = declassify(ts, TaintLabel.PII, "User consent given")
        assert not result.has(TaintLabel.PII)
        assert result.has(TaintLabel.SECRET)

    def test_remove_nonexistent_label(self):
        ts = TaintSet.from_labels(TaintLabel.PII)
        result = declassify(ts, TaintLabel.SECRET, "Not present")
        assert result.has(TaintLabel.PII)


class TestAutoDetect:
    def test_detect_email(self):
        ts = auto_detect_taint("Contact: user@example.com")
        assert ts.has(TaintLabel.PII)

    def test_detect_ssn(self):
        ts = auto_detect_taint("SSN: 123-45-6789")
        assert ts.has(TaintLabel.PII)

    def test_detect_api_key(self):
        ts = auto_detect_taint("Key: sk-abc123def456ghi789jkl012mno")
        assert ts.has(TaintLabel.SECRET)

    def test_detect_github_token(self):
        ts = auto_detect_taint("Token: ghp_abcdefghijklmnopqrstuvwxyz0123456789")
        assert ts.has(TaintLabel.SECRET)

    def test_clean_text(self):
        ts = auto_detect_taint("Hello, this is a normal message.")
        assert not ts

    def test_detect_private_key(self):
        ts = auto_detect_taint("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        assert ts.has(TaintLabel.SECRET)


class TestPropagate:
    def test_propagate_input_taint(self):
        input_taint = TaintSet.from_labels(TaintLabel.EXTERNAL)
        result = propagate_taint(input_taint, "Normal output")
        assert result.has(TaintLabel.EXTERNAL)

    def test_propagate_detects_new_taint(self):
        input_taint = TaintSet()
        result = propagate_taint(input_taint, "Found: user@example.com")
        assert result.has(TaintLabel.PII)

    def test_propagate_merges(self):
        input_taint = TaintSet.from_labels(TaintLabel.EXTERNAL)
        result = propagate_taint(input_taint, "Key: sk-abc123def456ghi789jkl012mno")
        assert result.has(TaintLabel.EXTERNAL)
        assert result.has(TaintLabel.SECRET)


class TestExternalBoundaryMetadata:
    def test_external_taint_is_json_safe_and_source_scoped(self):
        metadata = external_taint("mcp:home-assistant")
        assert metadata["_taint"] == {"labels": ["external"]}
        assert metadata["provenance"] == {
            "trust": "external",
            "source": "mcp:home-assistant",
        }

    def test_invalid_source_is_not_reflected(self):
        metadata = external_taint("bad source\nAuthorization: Bearer secret")
        assert metadata["provenance"]["source"] == "external"

    def test_redact_sensitive_text_covers_tokens_and_pii(self):
        value = (
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
            "email=user@example.com card=4111-1111-1111-1111"
        )
        redacted = redact_sensitive_text(value)
        assert "abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "user@example.com" not in redacted
        assert "4111-1111-1111-1111" not in redacted


class _TaintProbeTool(BaseTool):
    tool_id = "web_search"

    def __init__(self):
        self.calls = []

    @property
    def spec(self):
        return ToolSpec(
            name="web_search",
            description="Local taint sink probe.",
        )

    def execute(self, **params):
        self.calls.append(params)
        return ToolResult(
            tool_name="web_search",
            content=str(params.get("query", "")),
            success=True,
        )


class _ExternalOutputProbeTool(BaseTool):
    tool_id = "external_probe"

    def __init__(self, taint_payload=None):
        self.taint_payload = taint_payload or {"labels": ["external"]}
        self.calls = 0

    @property
    def spec(self):
        return ToolSpec(
            name=self.tool_id,
            description="Return output with explicit provenance.",
        )

    def execute(self, **params):
        del params
        self.calls += 1
        return ToolResult(
            tool_name=self.tool_id,
            content="normal external output",
            success=True,
            metadata={"_taint": self.taint_payload},
        )


def _taint_executor(tool, *, bus=None):
    policy = CapabilityPolicy()
    policy.grant("taint-agent", "tool:invoke")
    policy.grant("taint-agent", "network:fetch")
    return ToolExecutor(
        [tool],
        bus=bus,
        capability_policy=policy,
        agent_id="taint-agent",
    )


class TestToolExecutorTaintSerialization:
    def test_json_taint_payload_blocks_sink_before_execution(self):
        tool = _TaintProbeTool()
        result = _taint_executor(tool).execute(
            ToolCall(
                id="1",
                name="web_search",
                arguments=('{"query":"do not send","_taint":{"labels":["secret"]}}'),
            )
        )

        assert result.success is False
        assert "Taint violation" in result.content
        assert tool.calls == []

    def test_existing_external_output_taint_is_preserved(self):
        tool = _ExternalOutputProbeTool()

        result = _taint_executor(tool).execute(
            ToolCall(
                id="1",
                name=tool.tool_id,
                arguments="{}",
            )
        )

        assert result.success is True
        assert result.metadata["_taint"] == {"labels": ["external"]}
        assert tool.calls == 1

    def test_malformed_output_taint_withholds_completed_result(self):
        tool = _ExternalOutputProbeTool({"labels": ["unknown"]})

        result = _taint_executor(tool).execute(
            ToolCall(
                id="1",
                name=tool.tool_id,
                arguments="{}",
            )
        )

        assert result.success is False
        assert result.metadata["outcome"] == "completed"
        assert result.metadata["result_withheld"] is True
        assert "_taint" not in result.metadata
        assert tool.calls == 1

    def test_invalid_json_taint_payload_fails_closed(self):
        tool = _TaintProbeTool()
        result = _taint_executor(tool).execute(
            ToolCall(
                id="1",
                name="web_search",
                arguments=('{"query":"do not send","_taint":{"labels":["unknown"]}}'),
            )
        )

        assert result.success is False
        assert "Invalid taint metadata" in result.content
        assert tool.calls == []

    def test_detected_output_taint_is_json_safe_in_result_and_event(self):
        tool = _TaintProbeTool()
        bus = EventBus(record_history=True)
        result = _taint_executor(tool, bus=bus).execute(
            ToolCall(
                id="1",
                name="web_search",
                arguments='{"query":"user@example.com"}',
            )
        )

        assert result.success is True
        assert result.metadata["_taint"] == {"labels": ["pii"]}
        end_event = next(
            event
            for event in bus.history
            if event.event_type == EventType.TOOL_CALL_END
        )
        assert end_event.data["metadata"]["_taint"] == {"labels": ["pii"]}

    def test_allowed_input_taint_propagates_to_output_metadata(self):
        tool = _TaintProbeTool()
        result = _taint_executor(tool).execute(
            ToolCall(
                id="1",
                name="web_search",
                arguments=(
                    '{"query":"public result","_taint":{"labels":["external"]}}'
                ),
            )
        )

        assert result.success is True
        assert result.metadata["_taint"] == {"labels": ["external"]}
        assert tool.calls == [{"query": "public result"}]
