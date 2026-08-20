"""Fail-closed tests for the legacy managed Slack daemon."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openjarvis.channels.slack_daemon import (
    run_slack_daemon,
    start_slack_daemon,
)


def test_start_never_places_tokens_in_process_arguments() -> None:
    with patch("subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="disabled"):
            start_slack_daemon(
                bot_token="xoxb-private",
                app_token="xapp-private",
                agent_id="service-agent",
                allowed_sender_ids=["sender-1"],
            )

    popen.assert_not_called()


def test_direct_daemon_execution_is_disabled_before_optional_imports() -> None:
    with pytest.raises(RuntimeError, match="sender principal"):
        run_slack_daemon(
            bot_token="xoxb-private",
            app_token="xapp-private",
            agent_id="service-agent",
            allowed_sender_ids=["sender-1"],
        )
