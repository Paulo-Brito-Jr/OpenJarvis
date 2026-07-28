"""Fail-closed coverage for unauthenticated scheduler CLI call sites."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.cli.scheduler_cmd import scheduler


def test_scheduler_create_rejects_before_store_mutation() -> None:
    store = MagicMock()
    runner = CliRunner()

    with patch(
        "openjarvis.cli.scheduler_cmd._get_store",
        return_value=store,
    ):
        result = runner.invoke(
            scheduler,
            [
                "create",
                "private task",
                "--type",
                "once",
                "--value",
                "2099-01-01T00:00:00+00:00",
            ],
        )

    assert result.exit_code == 1
    assert "authenticated operator identity" in result.output
    store.save_task.assert_not_called()
    store.close.assert_called_once_with()


def test_scheduler_logs_rejects_unauthenticated_reads() -> None:
    store = MagicMock()
    runner = CliRunner()

    with patch(
        "openjarvis.cli.scheduler_cmd._get_store",
        return_value=store,
    ):
        result = runner.invoke(scheduler, ["logs", "task-id"])

    assert result.exit_code == 1
    assert "authenticated operator identity" in result.output
    store.get_run_logs.assert_not_called()


def test_scheduler_start_rejects_before_starting_thread() -> None:
    store = MagicMock()
    runner = CliRunner()

    with patch(
        "openjarvis.cli.scheduler_cmd._get_store",
        return_value=store,
    ):
        result = runner.invoke(scheduler, ["start"])

    assert result.exit_code == 1
    assert "authenticated operator identity" in result.output
