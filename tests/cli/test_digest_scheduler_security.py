"""Security gates for digest scheduling call sites."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from openjarvis.cli.digest_cmd import (
    _cancel_scheduler_tasks,
    _create_scheduler_task,
    digest,
)


def test_digest_schedule_helpers_require_authenticated_context() -> None:
    with pytest.raises(PermissionError, match="authenticated operator identity"):
        _create_scheduler_task("0 7 * * *")
    with pytest.raises(PermissionError, match="authenticated operator identity"):
        _cancel_scheduler_tasks()


def test_digest_schedule_cli_does_not_persist_before_authorization() -> None:
    config = MagicMock()
    config.digest.enabled = False
    config.digest.schedule = "0 6 * * *"
    config.digest.timezone = "UTC"
    runner = CliRunner()

    with (
        patch("openjarvis.cli.digest_cmd.load_config", return_value=config),
        patch("openjarvis.cli.digest_cmd._save_digest_schedule") as save,
    ):
        result = runner.invoke(digest, ["--schedule", "0 7 * * *"])

    assert result.exit_code == 1
    assert "durable recurring consent" in result.output
    save.assert_not_called()


def test_digest_helper_accepts_complete_caller_supplied_context() -> None:
    scheduler = MagicMock()
    scheduler.list_tasks.return_value = []
    scheduler.create_task.return_value = MagicMock(id="digest-task")
    consent = {
        "scope": "recurring",
        "allow_replay": True,
        "granted_at": "2026-07-28T12:00:00+00:00",
        "expires_at": "2026-07-29T12:00:00+00:00",
    }

    task_id = _create_scheduler_task(
        "0 7 * * *",
        scheduler=scheduler,
        operator_id="authenticated-user",
        capabilities=["schedule:create"],
        consent=consent,
    )

    assert task_id == "digest-task"
    scheduler.create_task.assert_called_once()
    assert scheduler.create_task.call_args.kwargs["operator_id"] == (
        "authenticated-user"
    )
    assert scheduler.create_task.call_args.kwargs["consent"] == consent
