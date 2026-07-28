"""Tests for TaskScheduler — scheduling logic, lifecycle, and execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from openjarvis.scheduler.scheduler import ScheduledTask, TaskScheduler
from openjarvis.scheduler.store import SchedulerStore
from openjarvis.security.capabilities import CapabilityPolicy

_TEST_OPERATOR = "scheduler-test"


def _policy() -> CapabilityPolicy:
    policy = CapabilityPolicy()
    policy.grant(_TEST_OPERATOR, "schedule:create", "schedule:*")
    return policy


def _make_scheduler(
    store: SchedulerStore,
    system=None,
    *,
    poll_interval: int = 1,
    bus=None,
    policy: CapabilityPolicy | None = None,
) -> TaskScheduler:
    return TaskScheduler(
        store,
        system=system,
        poll_interval=poll_interval,
        bus=bus,
        capability_policy=policy or _policy(),
        default_operator_id=_TEST_OPERATOR,
        default_capabilities=["schedule:create"],
    )


def _consent(schedule_type: str) -> dict:
    if schedule_type == "once":
        return {
            "scope": "once",
            "granted_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "scope": "recurring",
        "allow_replay": True,
        "granted_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    }


def _create_task(
    scheduler: TaskScheduler,
    prompt: str,
    schedule_type: str,
    schedule_value: str,
    **kwargs,
) -> ScheduledTask:
    return scheduler.create_task(
        prompt,
        schedule_type,
        schedule_value,
        consent=_consent(schedule_type),
        **kwargs,
    )


@pytest.fixture()
def store(tmp_path):
    s = SchedulerStore(tmp_path / "scheduler_test.db")
    yield s
    s.close()


@pytest.fixture()
def scheduler(store):
    sched = _make_scheduler(store)
    yield sched
    sched.stop()


# -- ScheduledTask dataclass -------------------------------------------------


class TestScheduledTask:
    def test_round_trip(self):
        task = ScheduledTask(
            id="abc123",
            prompt="hello",
            schedule_type="interval",
            schedule_value="60",
            agent="orchestrator",
            tools="calculator,think",
            metadata={"key": "value"},
        )
        d = task.to_dict()
        restored = ScheduledTask.from_dict(d)
        assert restored.id == "abc123"
        assert restored.prompt == "hello"
        assert restored.schedule_type == "interval"
        assert restored.agent == "orchestrator"
        assert restored.tools == "calculator,think"
        assert restored.metadata == {"key": "value"}

    def test_defaults(self):
        task = ScheduledTask(
            id="x",
            prompt="p",
            schedule_type="once",
            schedule_value="2026-01-01T00:00:00",
        )
        assert task.context_mode == "isolated"
        assert task.status == "active"
        assert task.agent == "simple"
        assert task.tools == ""
        assert task.metadata == {}


# -- TaskScheduler create/list -----------------------------------------------


class TestCreateAndList:
    def test_create_task(self, scheduler):
        task = _create_task(
            scheduler,
            prompt="hello world",
            schedule_type="interval",
            schedule_value="3600",
        )
        assert task.id
        assert task.prompt == "hello world"
        assert task.schedule_type == "interval"
        assert task.next_run is not None
        assert task.status == "active"

    def test_create_task_with_agent_and_tools(self, scheduler):
        task = _create_task(
            scheduler,
            prompt="hello",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
            agent="orchestrator",
            tools="calculator,think",
        )
        assert task.agent == "orchestrator"
        assert task.tools == "calculator,think"

    def test_list_tasks_empty(self, scheduler):
        assert scheduler.list_tasks() == []

    def test_list_tasks(self, scheduler):
        _create_task(scheduler, "a", "interval", "60")
        _create_task(scheduler, "b", "interval", "120")
        assert len(scheduler.list_tasks()) == 2

    def test_list_tasks_filter_status(self, scheduler):
        t1 = _create_task(scheduler, "a", "interval", "60")
        _create_task(scheduler, "b", "interval", "120")
        scheduler.pause_task(t1.id)
        active = scheduler.list_tasks(status="active")
        paused = scheduler.list_tasks(status="paused")
        assert len(active) == 1
        assert len(paused) == 1

    def test_create_requires_explicit_operator_identity(self, store):
        sched = TaskScheduler(store, capability_policy=_policy())

        with pytest.raises(PermissionError, match="operator identity"):
            sched.create_task(
                "hello",
                "once",
                "2099-01-01T00:00:00+00:00",
                consent=_consent("once"),
            )

    @pytest.mark.parametrize("poll_interval", [0, -1, float("nan"), float("inf")])
    def test_scheduler_rejects_invalid_poll_interval(
        self,
        store,
        poll_interval,
    ):
        with pytest.raises(ValueError, match="poll interval"):
            _make_scheduler(store, poll_interval=poll_interval)

    @pytest.mark.parametrize("schedule_value", ["0", "-1", "nan", "inf"])
    def test_create_rejects_invalid_interval(self, scheduler, schedule_value):
        with pytest.raises(ValueError, match="positive number"):
            _create_task(
                scheduler,
                "hello",
                "interval",
                schedule_value,
            )

    @pytest.mark.parametrize(
        "schedule_value",
        ["not-a-cron", "61 2 * * *", "30 24 * * *"],
    )
    def test_create_rejects_invalid_cron(self, scheduler, schedule_value):
        with pytest.raises(ValueError, match="Cron"):
            _create_task(
                scheduler,
                "hello",
                "cron",
                schedule_value,
            )

    @pytest.mark.parametrize(
        "schedule_value",
        ["not-a-date", "2099-01-01T00:00:00"],
    )
    def test_create_rejects_ambiguous_once_datetime(
        self,
        scheduler,
        schedule_value,
    ):
        with pytest.raises(ValueError, match="schedule"):
            _create_task(
                scheduler,
                "hello",
                "once",
                schedule_value,
            )

    def test_create_requires_explicit_consent(self, scheduler):
        with pytest.raises(PermissionError, match="grant timestamp"):
            scheduler.create_task(
                "hello",
                "once",
                "2099-01-01T00:00:00+00:00",
            )

    def test_create_rejects_expired_recurring_consent(self, scheduler):
        expired = _consent("interval")
        expired["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()

        with pytest.raises(PermissionError, match="within 31 days"):
            scheduler.create_task(
                "hello",
                "interval",
                "60",
                consent=expired,
            )

    def test_create_supports_valid_caller_supplied_task_id(self, scheduler):
        task = _create_task(
            scheduler,
            "hello",
            "interval",
            "60",
            task_id="operator:daily_digest",
        )

        assert task.id == "operator:daily_digest"

    def test_create_rejects_unsafe_caller_supplied_task_id(self, scheduler):
        with pytest.raises(ValueError, match="task ID"):
            _create_task(
                scheduler,
                "hello",
                "interval",
                "60",
                task_id="../../outside",
            )

    def test_other_operator_cannot_overwrite_caller_supplied_task_id(
        self,
        scheduler,
        store,
    ):
        task_id = "operator:daily_digest"
        _create_task(
            scheduler,
            "original prompt",
            "interval",
            "60",
            task_id=task_id,
        )
        policy = _policy()
        policy.grant("attacker", "schedule:create", "schedule:*")
        attacker = TaskScheduler(
            store,
            capability_policy=policy,
            default_operator_id="attacker",
            default_capabilities=["schedule:create"],
        )

        with pytest.raises(PermissionError, match="another operator"):
            _create_task(
                attacker,
                "replacement prompt",
                "interval",
                "60",
                task_id=task_id,
            )

        stored = store.get_task(task_id)
        assert stored["operator_id"] == _TEST_OPERATOR
        assert stored["prompt"] == "original prompt"

    def test_same_operator_cannot_overwrite_caller_supplied_task_id(
        self,
        scheduler,
        store,
    ):
        task_id = "operator:stable-id"
        _create_task(
            scheduler,
            "original prompt",
            "interval",
            "60",
            task_id=task_id,
        )

        with pytest.raises(ValueError, match="already exists"):
            _create_task(
                scheduler,
                "replacement prompt",
                "interval",
                "120",
                task_id=task_id,
            )

        stored = store.get_task(task_id)
        assert stored["prompt"] == "original prompt"
        assert stored["schedule_value"] == "60"

    def test_list_requires_live_capability(self, scheduler):
        with pytest.raises(PermissionError, match="capability denied"):
            scheduler.list_tasks(operator_id="attacker")


# -- Pause / resume / cancel -------------------------------------------------


class TestPauseResumeCancel:
    def test_pause_task(self, scheduler):
        task = _create_task(scheduler, "test", "interval", "60")
        scheduler.pause_task(task.id)
        tasks = scheduler.list_tasks(status="paused")
        assert len(tasks) == 1
        assert tasks[0].status == "paused"

    def test_resume_task(self, scheduler):
        task = _create_task(scheduler, "test", "interval", "60")
        scheduler.pause_task(task.id)
        scheduler.resume_task(task.id)
        tasks = scheduler.list_tasks(status="active")
        assert len(tasks) == 1

    def test_resume_revalidates_persisted_schedule(self, scheduler, store):
        task = _create_task(scheduler, "test", "interval", "60")
        scheduler.pause_task(task.id)
        stored = store.get_task(task.id)
        stored["schedule_value"] = "0"
        store.update_task(stored)

        with pytest.raises(ValueError, match="positive number"):
            scheduler.resume_task(task.id)

    def test_cancel_task(self, scheduler):
        task = _create_task(scheduler, "test", "interval", "60")
        scheduler.cancel_task(task.id)
        tasks = scheduler.list_tasks(status="cancelled")
        assert len(tasks) == 1
        assert tasks[0].next_run is None

    def test_pause_nonexistent(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.pause_task("nonexistent")

    def test_resume_nonexistent(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.resume_task("nonexistent")

    def test_cancel_nonexistent(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.cancel_task("nonexistent")

    def test_other_operator_cannot_control_task(self, scheduler):
        task = _create_task(scheduler, "test", "interval", "60")

        with pytest.raises(PermissionError, match="another operator"):
            scheduler.pause_task(task.id, operator_id="attacker")

    @pytest.mark.parametrize("operation", ["pause", "resume", "cancel"])
    def test_control_cannot_clobber_running_claim(
        self,
        scheduler,
        store,
        operation,
    ):
        task = _create_task(scheduler, "test", "interval", "60")
        claimed = store.claim_task(
            task.id,
            "worker:claim",
            datetime.now(timezone.utc).isoformat(),
        )
        assert claimed is not None

        method = getattr(scheduler, f"{operation}_task")
        with pytest.raises(RuntimeError, match="currently running"):
            method(task.id)

        stored = store.get_task(task.id)
        assert stored["status"] == "running"
        assert stored["claim_token"] == "worker:claim"


# -- _compute_next_run -------------------------------------------------------


class TestComputeNextRun:
    def test_interval(self, scheduler):
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="interval", schedule_value="300"
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is not None
        # Should be roughly 300 seconds from now
        parsed = datetime.fromisoformat(next_run)
        diff = (parsed - datetime.now(timezone.utc)).total_seconds()
        assert 295 <= diff <= 310

    def test_once_not_yet_run(self, scheduler):
        target = "2099-06-15T12:00:00+00:00"
        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="once",
            schedule_value=target,
            last_run=None,
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run == target

    def test_once_already_run(self, scheduler):
        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="once",
            schedule_value="2099-06-15T12:00:00+00:00",
            last_run="2099-06-15T12:01:00+00:00",
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is None

    def test_cron_fallback(self, scheduler):
        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="cron",
            schedule_value="30 2 * * *",
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is not None

    def test_unknown_type(self, scheduler):
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="unknown", schedule_value="x"
        )
        assert scheduler._compute_next_run(task) is None


# -- _execute_task -----------------------------------------------------------


class TestExecuteTask:
    def test_execute_with_system(self, store):
        mock_system = MagicMock()
        mock_system.ask.return_value = "result text"
        sched = _make_scheduler(store, system=mock_system)

        task = _create_task(
            sched,
            "what is 2+2?",
            "once",
            "2026-01-01T00:00:00+00:00",
        )
        sched._execute_task(task)

        mock_system.ask.assert_called_once()
        call_args = mock_system.ask.call_args
        assert call_args[0][0] == "what is 2+2?"

        # Check that a run log was recorded
        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert logs[0]["result"] == "completed"

    def test_execute_without_system(self, store):
        sched = _make_scheduler(store)
        task = _create_task(
            sched,
            "dry run",
            "once",
            "2026-01-01T00:00:00+00:00",
        )
        sched._execute_task(task)

        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 0
        assert logs[0]["error"] == "RuntimeError"
        assert store.get_task(task.id)["status"] == "paused"

    def test_execute_with_error(self, store):
        mock_system = MagicMock()
        mock_system.ask.side_effect = RuntimeError("engine down")
        sched = _make_scheduler(store, system=mock_system)

        task = _create_task(
            sched,
            "fail",
            "once",
            "2026-01-01T00:00:00+00:00",
        )
        sched._execute_task(task)

        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 0
        assert logs[0]["error"] == "RuntimeError"

    def test_execute_publishes_events(self, store):
        mock_bus = MagicMock()
        sched = _make_scheduler(store, bus=mock_bus)
        task = _create_task(
            sched,
            "test",
            "once",
            "2026-01-01T00:00:00+00:00",
        )
        sched._execute_task(task)

        assert mock_bus.publish.call_count == 2
        start_call = mock_bus.publish.call_args_list[0]
        end_call = mock_bus.publish.call_args_list[1]
        assert start_call[0][0] == "scheduler_task_start"
        assert end_call[0][0] == "scheduler_task_end"

    def test_execute_once_task_completed_after_run(self, store):
        mock_system = MagicMock()
        sched = _make_scheduler(store, system=mock_system)
        task = _create_task(
            sched,
            "one-shot",
            "once",
            "2026-01-01T00:00:00+00:00",
        )
        sched._execute_task(task)

        updated = store.get_task(task.id)
        assert updated["status"] == "completed"
        assert updated["next_run"] is None

    def test_execute_with_tools(self, store):
        mock_system = MagicMock()
        mock_system.ask.return_value = "4"
        sched = _make_scheduler(store, system=mock_system)

        task = _create_task(
            sched,
            "what is 2+2?",
            "once",
            "2026-01-01T00:00:00+00:00",
            tools="calculator,think",
        )
        sched._execute_task(task)

        call_kwargs = mock_system.ask.call_args[1]
        assert call_kwargs["tools"] == ["calculator", "think"]

    def test_execute_preserves_empty_tool_and_capability_snapshots(self, store):
        mock_system = MagicMock()
        policy = _policy()
        policy.grant(_TEST_OPERATOR, "memory:read", "memory:context")
        sched = _make_scheduler(store, system=mock_system, policy=policy)
        task = _create_task(
            sched,
            "isolated",
            "once",
            "2026-01-01T00:00:00+00:00",
            tools=[],
            capabilities=["schedule:create", "memory:read"],
        )

        sched._execute_task(task)

        call_kwargs = mock_system.ask.call_args.kwargs
        assert call_kwargs["tools"] == []
        assert call_kwargs["capability_scope"] == [
            "schedule:create",
            "memory:read",
        ]

    def test_durable_claim_prevents_two_workers_from_replaying_task(
        self,
        tmp_path,
    ):
        db_path = tmp_path / "shared-scheduler.db"
        first_store = SchedulerStore(db_path)
        second_store = SchedulerStore(db_path)
        mock_system = MagicMock()
        first = _make_scheduler(first_store, system=mock_system)
        second = _make_scheduler(second_store, system=mock_system)
        try:
            task = _create_task(
                first,
                "run once",
                "once",
                "2026-01-01T00:00:00+00:00",
            )
            first_copy = ScheduledTask.from_dict(first_store.get_task(task.id))
            second_copy = ScheduledTask.from_dict(second_store.get_task(task.id))

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(first._execute_task, first_copy),
                    pool.submit(second._execute_task, second_copy),
                ]
                for future in futures:
                    future.result()

            assert mock_system.ask.call_count == 1
            assert len(first_store.get_run_logs(task.id)) == 1
        finally:
            first_store.close()
            second_store.close()

    def test_execution_rechecks_revoked_capability(self, store):
        mock_system = MagicMock()
        policy = _policy()
        sched = _make_scheduler(store, system=mock_system, policy=policy)
        task = _create_task(
            sched,
            "sensitive action",
            "once",
            "2026-01-01T00:00:00+00:00",
        )
        policy.deny(_TEST_OPERATOR, "schedule:create")

        sched._execute_task(task)

        mock_system.ask.assert_not_called()
        logs = store.get_run_logs(task.id)
        assert logs[0]["success"] == 0
        assert logs[0]["error"] == "PermissionError"
        assert store.get_task(task.id)["status"] == "paused"


# -- Start / stop lifecycle ---------------------------------------------------


class TestLifecycle:
    def test_start_stop(self, scheduler):
        scheduler.start()
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()
        scheduler.stop()
        assert not scheduler._thread

    def test_double_start(self, scheduler):
        scheduler.start()
        t1 = scheduler._thread
        scheduler.start()  # Should not create a second thread
        assert scheduler._thread is t1
        scheduler.stop()

    def test_poll_loop_finds_due_tasks(self, store):
        mock_system = MagicMock()
        sched = _make_scheduler(store, system=mock_system)
        # Create a task that is already due
        task = _create_task(
            sched,
            "immediate",
            "once",
            "2020-01-01T00:00:00+00:00",
        )
        # Manually set next_run to the past
        d = store.get_task(task.id)
        d["next_run"] = "2020-01-01T00:00:00+00:00"
        store.update_task(d)

        sched.start()
        # Give the poll loop time to execute
        time.sleep(2.5)
        sched.stop()

        logs = store.get_run_logs(task.id)
        assert len(logs) >= 1
