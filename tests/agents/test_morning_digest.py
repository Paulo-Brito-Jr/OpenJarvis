"""Tests for MorningDigestAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openjarvis.agents._stubs import AgentResult
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult


def test_morning_digest_registered():
    from openjarvis.agents.morning_digest import MorningDigestAgent

    AgentRegistry.register_value("morning_digest", MorningDigestAgent)
    assert AgentRegistry.contains("morning_digest")


def test_morning_digest_run(tmp_path):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    mock_engine = MagicMock()
    mock_engine.generate.return_value = {
        "content": "Good morning sir. You have 3 emails and 2 meetings today.",
        "finish_reason": "stop",
        "usage": {},
    }

    # Mock collect result
    mock_collect_result = ToolResult(
        tool_name="digest_collect",
        content='=== MESSAGES ===\n[gmail] From: alice@co.com — "Budget" (1h ago)\n',
        success=True,
        metadata={"total_items": 2},
    )

    # Mock TTS result
    mock_tts_result = ToolResult(
        tool_name="text_to_speech",
        content=str(tmp_path / "digest.mp3"),
        success=True,
        metadata={"audio_path": str(tmp_path / "digest.mp3")},
    )

    agent = MorningDigestAgent(
        mock_engine,
        "test-model",
        tools=[],
        persona="neutral",
        digest_store_path=str(tmp_path / "digest.db"),
    )

    with (
        patch.object(agent._executor, "authorize", return_value=None),
        patch.object(
            agent._executor,
            "execute",
            side_effect=[mock_collect_result, mock_tts_result],
        ),
    ):
        result = agent.run("Generate morning digest")

    assert isinstance(result, AgentResult)
    assert "Good morning" in result.content
    assert result.turns == 1
    assert len(result.tool_results) == 2


def test_morning_digest_stops_before_inference_and_storage_when_collect_denied(
    tmp_path,
):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    engine = MagicMock()
    failed_collect = ToolResult(
        tool_name="digest_collect",
        content="Capability denied.",
        success=False,
    )
    agent = MorningDigestAgent(
        engine,
        "test-model",
        tools=[],
        digest_store_path=str(tmp_path / "must-not-exist.db"),
    )

    with (
        patch.object(agent._executor, "authorize", return_value=None),
        patch.object(agent._executor, "execute", return_value=failed_collect),
        patch("openjarvis.agents.morning_digest.DigestStore") as digest_store,
    ):
        result = agent.run("Generate morning digest")

    assert result.turns == 0
    assert result.metadata["collection_failed"] is True
    engine.generate.assert_not_called()
    digest_store.assert_not_called()
    assert not (tmp_path / "must-not-exist.db").exists()


def test_morning_digest_denies_store_before_collect_or_inference(tmp_path):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    engine = MagicMock()
    agent = MorningDigestAgent(
        engine,
        "test-model",
        tools=[],
        digest_store_path=str(tmp_path / "must-not-exist.db"),
    )

    with (
        patch.object(agent._executor, "execute") as execute,
        patch("openjarvis.agents.morning_digest.DigestStore") as digest_store,
    ):
        result = agent.run("Generate morning digest")

    assert result.turns == 0
    assert result.metadata["security_denied"] is True
    execute.assert_not_called()
    engine.generate.assert_not_called()
    digest_store.assert_not_called()
    assert not (tmp_path / "must-not-exist.db").exists()


def test_load_persona():
    from openjarvis.agents.morning_digest import _load_persona

    # Nonexistent persona returns empty string
    result = _load_persona("nonexistent_persona_xyz")
    assert result == ""
