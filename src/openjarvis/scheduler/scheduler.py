"""Task scheduler — cron/interval/once execution with background polling."""

from __future__ import annotations

import logging
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openjarvis.scheduler.store import SchedulerStore

logger = logging.getLogger(__name__)

# Event type strings (avoids editing core EventType enum)
SCHEDULER_TASK_START = "scheduler_task_start"
SCHEDULER_TASK_END = "scheduler_task_end"


@dataclass(slots=True)
class ScheduledTask:
    """A task scheduled for future or recurring execution."""

    id: str
    prompt: str
    schedule_type: str  # "cron" | "interval" | "once"
    schedule_value: str  # cron expression, interval seconds, ISO datetime
    context_mode: str = "isolated"
    status: str = "active"
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    agent: str = "simple"
    tools: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    operator_id: str = ""
    capabilities: List[str] = field(default_factory=list)
    consent: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for store persistence."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "schedule_type": self.schedule_type,
            "schedule_value": self.schedule_value,
            "context_mode": self.context_mode,
            "status": self.status,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "agent": self.agent,
            "tools": self.tools,
            "metadata": self.metadata,
            "operator_id": self.operator_id,
            "capabilities": self.capabilities,
            "consent": self.consent,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScheduledTask:
        """Deserialize from a plain dict."""
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            schedule_type=d["schedule_type"],
            schedule_value=d["schedule_value"],
            context_mode=d.get("context_mode", "isolated"),
            status=d.get("status", "active"),
            next_run=d.get("next_run"),
            last_run=d.get("last_run"),
            agent=d.get("agent", "simple"),
            tools=d.get("tools", ""),
            metadata=d.get("metadata", {}),
            operator_id=d.get("operator_id", ""),
            capabilities=d.get("capabilities", []),
            consent=d.get("consent", {}),
        )


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TaskScheduler:
    """Scheduler that polls for due tasks and executes them.

    Parameters
    ----------
    store:
        The persistence backend.
    system:
        Optional ``JarvisSystem`` instance for executing prompts.
    poll_interval:
        Seconds between poll cycles (default 60).
    bus:
        Optional event bus for publishing scheduler events.
    """

    def __init__(
        self,
        store: SchedulerStore,
        system: Any = None,
        *,
        poll_interval: int = 60,
        bus: Any = None,
        capability_policy: Any = None,
        default_operator_id: str = "",
        default_capabilities: Optional[List[str]] = None,
    ) -> None:
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(float(poll_interval))
            or poll_interval <= 0
        ):
            raise ValueError("Scheduler poll interval must be positive")
        self._store = store
        self._system = system
        self._poll_interval = float(poll_interval)
        self._bus = bus
        self._capability_policy = capability_policy
        self._default_operator_id = default_operator_id.strip()
        self._default_capabilities = list(default_capabilities or [])
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._worker_id = uuid.uuid4().hex

    def bind_system(self, system: Any) -> None:
        """Bind the execution system after the containing system is built."""
        self._system = system

    def _resolve_operator(self, operator_id: str | None) -> str:
        resolved = (operator_id or self._default_operator_id).strip()
        if not resolved:
            raise PermissionError("Scheduled operation requires an operator identity")
        return resolved

    def _check_capability(
        self,
        operator_id: str,
        capability: str,
        resource: str,
    ) -> None:
        if self._capability_policy is None:
            raise PermissionError("Scheduler capability policy is unavailable")
        try:
            allowed = self._capability_policy.check(
                operator_id,
                capability,
                resource,
            )
        except Exception as exc:
            raise PermissionError("Scheduler authorization failed") from exc
        if not allowed:
            raise PermissionError(
                f"Scheduler capability denied: {capability}"
            )

    @staticmethod
    def _validate_consent(
        schedule_type: str,
        consent: Dict[str, Any],
    ) -> None:
        if not isinstance(consent, dict):
            raise PermissionError("Scheduler consent record is invalid")
        granted_at = consent.get("granted_at")
        if not isinstance(granted_at, str) or not granted_at:
            raise PermissionError("Scheduler consent grant timestamp is missing")
        try:
            granted = datetime.fromisoformat(granted_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PermissionError(
                "Scheduler consent grant timestamp is invalid"
            ) from exc
        if granted.tzinfo is None:
            granted = granted.replace(tzinfo=timezone.utc)
        if granted > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise PermissionError("Scheduler consent grant timestamp is in the future")
        if schedule_type == "once":
            if consent.get("scope") != "once" or consent.get("used_at"):
                raise PermissionError("One-time scheduler consent is missing")
            return
        if (
            consent.get("scope") != "recurring"
            or consent.get("allow_replay") is not True
        ):
            raise PermissionError(
                "Recurring schedules require explicit replay consent"
            )
        expires_at = consent.get("expires_at")
        if not isinstance(expires_at, str) or not expires_at:
            raise PermissionError("Recurring scheduler consent must expire")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PermissionError(
                "Recurring scheduler consent expiry is invalid"
            ) from exc
        now = datetime.now(timezone.utc)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= now or expiry > now + timedelta(days=31):
            raise PermissionError(
                "Recurring scheduler consent must expire within 31 days"
            )

    # -- Public API ----------------------------------------------------------

    def start(self) -> None:
        """Start the background polling daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="jarvis-scheduler"
        )
        self._thread.start()
        logger.info("Scheduler started (poll_interval=%ss)", self._poll_interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 5)
            self._thread = None
        logger.info("Scheduler stopped")

    def create_task(
        self,
        prompt: str,
        schedule_type: str,
        schedule_value: str,
        **kwargs: Any,
    ) -> ScheduledTask:
        """Create and persist a new scheduled task."""
        operator_id = self._resolve_operator(kwargs.get("operator_id"))
        capabilities = list(
            dict.fromkeys(
                kwargs.get("capabilities")
                or self._default_capabilities
                or ["schedule:create"]
            )
        )
        if not all(
            isinstance(capability, str)
            and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.-]*:[A-Za-z][A-Za-z0-9_.-]*",
                capability,
            )
            for capability in capabilities
        ):
            raise PermissionError("Scheduled task capabilities are invalid")
        if "schedule:create" not in capabilities:
            capabilities.append("schedule:create")
        self._check_capability(
            operator_id,
            "schedule:create",
            "schedule:new",
        )
        if schedule_type not in {"cron", "interval", "once"}:
            raise ValueError("Unsupported schedule type")
        schedule_value = self._normalize_schedule_value(
            schedule_type,
            schedule_value,
        )
        if not prompt.strip() or len(prompt) > 20_000:
            raise ValueError("Scheduled prompt must contain 1 to 20000 characters")
        consent = dict(kwargs.get("consent") or {})
        self._validate_consent(schedule_type, consent)
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = uuid.uuid4().hex[:16]
        if (
            not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9:_-]{1,128}", task_id)
        ):
            raise ValueError("Scheduled task ID is invalid")
        raw_tools = kwargs.get("tools", "")
        if isinstance(raw_tools, str):
            tool_values = raw_tools.split(",") if raw_tools else []
        elif isinstance(raw_tools, list):
            tool_values = raw_tools
        else:
            raise ValueError("Scheduled tools must be a list or comma-separated string")
        normalized_tools: List[str] = []
        for raw_tool in tool_values:
            if not isinstance(raw_tool, str):
                raise ValueError("Scheduled tool names are invalid")
            tool_name = raw_tool.strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", tool_name):
                raise ValueError("Scheduled tool names are invalid")
            if tool_name not in normalized_tools:
                normalized_tools.append(tool_name)
        task = ScheduledTask(
            id=task_id,
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            agent=kwargs.get("agent", "simple"),
            tools=",".join(normalized_tools),
            context_mode=kwargs.get("context_mode", "isolated"),
            metadata=kwargs.get("metadata", {}),
            operator_id=operator_id,
            capabilities=capabilities,
            consent=consent,
        )
        task.next_run = self._compute_next_run(task)
        with self._lock:
            created = self._store.create_task(task.to_dict())
            if not created:
                existing = self._store.get_task(task.id)
                if existing is None:
                    raise RuntimeError(
                        "Scheduled task ID collision could not be reconciled"
                    )
                if existing.get("operator_id") != operator_id:
                    raise PermissionError(
                        "Scheduled task ID is owned by another operator"
                    )
                raise ValueError("Scheduled task ID already exists")
        return task

    def list_tasks(
        self,
        *,
        status: Optional[str] = None,
        operator_id: str | None = None,
    ) -> List[ScheduledTask]:
        """Return tasks, optionally filtered by *status*."""
        resolved_operator = self._resolve_operator(operator_id)
        self._check_capability(
            resolved_operator,
            "schedule:create",
            "schedule:list",
        )
        with self._lock:
            rows = self._store.list_tasks(status=status)
        return [
            ScheduledTask.from_dict(row)
            for row in rows
            if row.get("operator_id") == resolved_operator
        ]

    def pause_task(self, task_id: str, *, operator_id: str | None = None) -> None:
        """Pause an active task."""
        resolved_operator = self._resolve_operator(operator_id)
        with self._lock:
            d = self._store.get_task(task_id)
            if d is None:
                raise KeyError(f"Task not found: {task_id}")
            if d.get("operator_id") != resolved_operator:
                raise PermissionError("Scheduled task is owned by another operator")
            if d.get("status") == "running":
                raise RuntimeError(
                    "Scheduled task is currently running and cannot be paused"
                )
            if d.get("status") != "active":
                raise ValueError("Only an active scheduled task can be paused")
            self._check_capability(
                resolved_operator,
                "schedule:create",
                f"schedule:{task_id}",
            )
            if not self._store.pause_active_task(task_id, resolved_operator):
                raise RuntimeError(
                    "Scheduled task state changed before it could be paused"
                )

    def resume_task(self, task_id: str, *, operator_id: str | None = None) -> None:
        """Resume a paused task."""
        resolved_operator = self._resolve_operator(operator_id)
        with self._lock:
            d = self._store.get_task(task_id)
            if d is None:
                raise KeyError(f"Task not found: {task_id}")
            if d.get("operator_id") != resolved_operator:
                raise PermissionError("Scheduled task is owned by another operator")
            if d.get("status") == "running":
                raise RuntimeError(
                    "Scheduled task is currently running and cannot be resumed"
                )
            if d.get("status") != "paused":
                raise ValueError("Only a paused scheduled task can be resumed")
            self._check_capability(
                resolved_operator,
                "schedule:create",
                f"schedule:{task_id}",
            )
            self._validate_consent(
                d.get("schedule_type", ""),
                d.get("consent", {}),
            )
            d["schedule_value"] = self._normalize_schedule_value(
                d.get("schedule_type", ""),
                d.get("schedule_value", ""),
            )
            d["status"] = "active"
            # Recompute next_run from now
            task = ScheduledTask.from_dict(d)
            task.next_run = self._compute_next_run(task)
            if not self._store.resume_paused_task(
                task_id,
                resolved_operator,
                schedule_value=task.schedule_value,
                next_run=task.next_run,
            ):
                raise RuntimeError(
                    "Scheduled task state changed before it could be resumed"
                )

    def cancel_task(self, task_id: str, *, operator_id: str | None = None) -> None:
        """Cancel a task (sets status to cancelled)."""
        resolved_operator = self._resolve_operator(operator_id)
        with self._lock:
            d = self._store.get_task(task_id)
            if d is None:
                raise KeyError(f"Task not found: {task_id}")
            if d.get("operator_id") != resolved_operator:
                raise PermissionError("Scheduled task is owned by another operator")
            if d.get("status") == "running":
                raise RuntimeError(
                    "Scheduled task is currently running and cannot be cancelled"
                )
            if d.get("status") == "cancelled":
                return
            self._check_capability(
                resolved_operator,
                "schedule:create",
                f"schedule:{task_id}",
            )
            if not self._store.cancel_unclaimed_task(
                task_id,
                resolved_operator,
            ):
                raise RuntimeError(
                    "Scheduled task state changed before it could be cancelled"
                )

    # -- Background loop -----------------------------------------------------

    def _poll_loop(self) -> None:
        """Poll for due tasks and execute them until stopped."""
        while not self._stop_event.is_set():
            try:
                now = _now_iso()
                while not self._stop_event.is_set():
                    claim_token = f"{self._worker_id}:{uuid.uuid4().hex}"
                    with self._lock:
                        task_dict = self._store.claim_due_task(now, claim_token)
                    if task_dict is None:
                        break
                    self._execute_task(
                        ScheduledTask.from_dict(task_dict),
                        claim_token=claim_token,
                    )
            except Exception:
                logger.exception("Scheduler poll error")
            self._stop_event.wait(timeout=self._poll_interval)

    def _execute_task(
        self,
        task: ScheduledTask,
        *,
        claim_token: Optional[str] = None,
    ) -> None:
        """Execute a single due task and log the result."""
        started_at = _now_iso()
        if claim_token is None:
            claim_token = f"{self._worker_id}:{uuid.uuid4().hex}"
            with self._lock:
                claimed = self._store.claim_task(
                    task.id,
                    claim_token,
                    started_at,
                )
            if claimed is None:
                logger.warning(
                    "Scheduler task %s was not claimable; execution skipped",
                    task.id,
                )
                return
            task = ScheduledTask.from_dict(claimed)
        else:
            with self._lock:
                claimed = self._store.get_task(task.id)
            if (
                claimed is None
                or claimed.get("status") != "running"
                or claimed.get("claim_token") != claim_token
            ):
                logger.warning(
                    "Scheduler task %s claim ownership is invalid; "
                    "execution skipped",
                    task.id,
                )
                return
            task = ScheduledTask.from_dict(claimed)

        # Publish start event
        if self._bus is not None:
            self._bus.publish(
                SCHEDULER_TASK_START,
                {"task_id": task.id, "operator_id": task.operator_id},
            )

        success = False
        authorization_failed = False
        result_text = ""
        error_text = ""

        try:
            self._validate_consent(task.schedule_type, task.consent)
            if (
                "schedule:create" not in task.capabilities
                or not all(
                    isinstance(capability, str)
                    and re.fullmatch(
                        r"[A-Za-z][A-Za-z0-9_.-]*:"
                        r"[A-Za-z][A-Za-z0-9_.-]*",
                        capability,
                    )
                    for capability in task.capabilities
                )
            ):
                raise PermissionError(
                    "Scheduled task capability snapshot is invalid"
                )
            # The scheduler-management grant is checked against the schedule
            # resource. Other snapshot capabilities are enforced against the
            # real tool/provider resource by QueryOrchestrator's scoped policy.
            self._check_capability(
                task.operator_id,
                "schedule:create",
                f"schedule:{task.id}",
            )
            if self._system is not None:
                raw_tools = (
                    task.tools
                    if isinstance(task.tools, list)
                    else task.tools.split(",")
                )
                tools_list = (
                    [t.strip() for t in raw_tools if t.strip()] if task.tools else []
                )
                if not all(
                    re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}",
                        tool_name,
                    )
                    for tool_name in tools_list
                ):
                    raise PermissionError(
                        "Scheduled task tool snapshot is invalid"
                    )
                ask_kwargs: Dict[str, Any] = {
                    "agent": task.agent,
                    # An explicit empty list means no tools. Never expand an
                    # empty schedule snapshot to the system's full tool set.
                    "tools": tools_list,
                    "capability_scope": list(task.capabilities),
                }
                meta = task.metadata or {}
                if task.operator_id:
                    ask_kwargs["system_prompt"] = meta.get("system_prompt", "")
                    ask_kwargs["operator_id"] = task.operator_id
                self._system.ask(
                    task.prompt,
                    **ask_kwargs,
                )
                result_text = "completed"
            else:
                raise RuntimeError("Scheduler execution system is unavailable")
            success = True
        except PermissionError as exc:
            authorization_failed = True
            error_text = type(exc).__name__
            logger.error("Task %s authorization failed", task.id)
        except Exception as exc:
            error_text = type(exc).__name__
            logger.error("Task %s failed: %s", task.id, type(exc).__name__)

        finished_at = _now_iso()

        # Log the run
        with self._lock:
            self._store.log_run(
                task_id=task.id,
                started_at=started_at,
                finished_at=finished_at,
                success=success,
                result=result_text,
                error=error_text,
            )

            # Update task state
            d = self._store.get_task(task.id)
            if d is not None:
                d["last_run"] = finished_at
                if d.get("schedule_type") == "once":
                    consent = dict(d.get("consent") or {})
                    consent["used_at"] = finished_at
                    d["consent"] = consent
                next_run = self._compute_next_run(ScheduledTask.from_dict(d))
                d["next_run"] = next_run
                if authorization_failed:
                    d["status"] = "paused"
                    d["next_run"] = None
                elif next_run is not None:
                    d["status"] = "active"
                if next_run is None:
                    d["status"] = "completed" if success else "paused"
                finished = self._store.finish_claim(
                    task.id,
                    claim_token,
                    status=d["status"],
                    next_run=d["next_run"],
                    last_run=finished_at,
                    consent=d.get("consent", {}),
                )
                if not finished:
                    logger.error(
                        "Scheduler task %s lost its durable claim; "
                        "manual reconciliation required",
                        task.id,
                    )

        # Publish end event
        if self._bus is not None:
            self._bus.publish(
                SCHEDULER_TASK_END,
                {
                    "task_id": task.id,
                    "success": success,
                },
            )

    def _compute_next_run(self, task: ScheduledTask) -> Optional[str]:
        """Compute the next run time for a task.

        Returns an ISO 8601 string, or ``None`` if the task should not run again.
        """
        now = datetime.now(timezone.utc)

        if task.schedule_type == "once":
            # If already run, no more runs
            if task.last_run is not None:
                return None
            # Otherwise the schedule_value is the target ISO datetime.
            return self._normalize_schedule_value("once", task.schedule_value)

        if task.schedule_type == "interval":
            normalized = self._normalize_schedule_value(
                "interval",
                task.schedule_value,
            )
            seconds = float(normalized)
            next_time = now + timedelta(seconds=seconds)
            return next_time.isoformat()

        if task.schedule_type == "cron":
            return self._compute_next_cron(task.schedule_value, now)

        return None

    @classmethod
    def _normalize_schedule_value(
        cls,
        schedule_type: str,
        schedule_value: Any,
    ) -> str:
        """Validate and canonicalize schedule input before persistence/use."""
        if not isinstance(schedule_value, str):
            raise ValueError("Scheduled value must be a string")
        value = schedule_value.strip()
        if not value or len(value) > 512:
            raise ValueError("Scheduled value is invalid")

        if schedule_type == "interval":
            try:
                seconds = float(value)
            except ValueError as exc:
                raise ValueError(
                    "Scheduled interval must be a positive number"
                ) from exc
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Scheduled interval must be a positive number")
            return value

        if schedule_type == "once":
            try:
                target = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    "One-time schedule must be an ISO 8601 datetime"
                ) from exc
            if target.tzinfo is None or target.utcoffset() is None:
                raise ValueError(
                    "One-time schedule must include an explicit timezone"
                )
            return target.astimezone(timezone.utc).isoformat()

        if schedule_type == "cron":
            cls._validate_cron_expression(value)
            return value

        raise ValueError("Unsupported schedule type")

    @staticmethod
    def _validate_cron_expression(cron_expr: str) -> None:
        """Accept a valid five-field cron, failing closed without croniter."""
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError("Cron schedule must contain exactly five fields")

        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError:
            minute, hour, day, month, weekday = parts
            if day != "*" or month != "*" or weekday != "*":
                raise ValueError(
                    "This cron expression requires the optional croniter backend"
                )
            for field, lower, upper, label in (
                (minute, 0, 59, "minute"),
                (hour, 0, 23, "hour"),
            ):
                if field == "*":
                    continue
                if not field.isascii() or not field.isdecimal():
                    raise ValueError(
                        "This cron expression requires the optional croniter backend"
                    )
                parsed = int(field)
                if not lower <= parsed <= upper:
                    raise ValueError(f"Cron {label} is out of range")
            return

        try:
            valid = croniter.is_valid(cron_expr)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cron schedule is invalid") from exc
        if not valid:
            raise ValueError("Cron schedule is invalid")

    @staticmethod
    def _compute_next_cron(cron_expr: str, now: datetime) -> Optional[str]:
        """Compute the next run time from a cron expression.

        Uses ``croniter`` if available, otherwise falls back to a basic
        minute-granularity parser for simple expressions.
        """
        try:
            from croniter import croniter  # type: ignore[import-untyped]

            try:
                it = croniter(cron_expr, now)
                return it.get_next(datetime).isoformat()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Cron schedule is invalid") from exc
        except ImportError:
            pass

        # Basic fail-closed fallback: only "minute hour * * *" where minute
        # and hour are either "*" or one in-range integer.
        TaskScheduler._validate_cron_expression(cron_expr)
        parts = cron_expr.strip().split()
        minute_part, hour_part = parts[0], parts[1]
        target_minute = None if minute_part == "*" else int(minute_part)
        target_hour = None if hour_part == "*" else int(hour_part)

        # Search minute-by-minute. The restricted fallback always has a match
        # within 24 hours, and starting at the next minute matches croniter's
        # strictly-after-now semantics.
        candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(24 * 60 + 1):
            if (
                (target_minute is None or candidate.minute == target_minute)
                and (target_hour is None or candidate.hour == target_hour)
            ):
                return candidate.isoformat()
            candidate += timedelta(minutes=1)
        raise ValueError("Cron schedule has no reachable execution time")


__all__ = [
    "SCHEDULER_TASK_END",
    "SCHEDULER_TASK_START",
    "ScheduledTask",
    "TaskScheduler",
]
