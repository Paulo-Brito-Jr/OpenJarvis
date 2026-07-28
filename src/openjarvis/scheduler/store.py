"""SQLite-backed persistence for scheduled tasks and run logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_CREATE_TASKS_TABLE = """\
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id              TEXT PRIMARY KEY,
    prompt          TEXT    NOT NULL,
    schedule_type   TEXT    NOT NULL,
    schedule_value  TEXT    NOT NULL,
    context_mode    TEXT    NOT NULL DEFAULT 'isolated',
    status          TEXT    NOT NULL DEFAULT 'active',
    next_run        TEXT,
    last_run        TEXT,
    agent           TEXT    NOT NULL DEFAULT 'simple',
    tools           TEXT    NOT NULL DEFAULT '',
    metadata        TEXT    NOT NULL DEFAULT '{}',
    operator_id     TEXT    NOT NULL DEFAULT '',
    capabilities    TEXT    NOT NULL DEFAULT '[]',
    consent         TEXT    NOT NULL DEFAULT '{}',
    claim_token     TEXT,
    claim_started_at TEXT
);
"""

_CREATE_LOGS_TABLE = """\
CREATE TABLE IF NOT EXISTS task_run_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    finished_at TEXT,
    success     INTEGER NOT NULL DEFAULT 0,
    result      TEXT    NOT NULL DEFAULT '',
    error       TEXT    NOT NULL DEFAULT ''
);
"""

_UPSERT_TASK = """\
INSERT INTO scheduled_tasks
    (id, prompt, schedule_type, schedule_value, context_mode,
     status, next_run, last_run, agent, tools, metadata, operator_id,
     capabilities, consent)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    prompt = excluded.prompt,
    schedule_type = excluded.schedule_type,
    schedule_value = excluded.schedule_value,
    context_mode = excluded.context_mode,
    status = excluded.status,
    next_run = excluded.next_run,
    last_run = excluded.last_run,
    agent = excluded.agent,
    tools = excluded.tools,
    metadata = excluded.metadata,
    operator_id = excluded.operator_id,
    capabilities = excluded.capabilities,
    consent = excluded.consent
WHERE scheduled_tasks.operator_id = excluded.operator_id
"""

_INSERT_TASK = """\
INSERT INTO scheduled_tasks
    (id, prompt, schedule_type, schedule_value, context_mode,
     status, next_run, last_run, agent, tools, metadata, operator_id,
     capabilities, consent)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_LOG = """\
INSERT INTO task_run_logs
    (task_id, started_at, finished_at, success, result, error)
VALUES (?, ?, ?, ?, ?, ?)
"""


class SchedulerStore:
    """SQLite CRUD store for scheduled tasks and their run logs."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            from openjarvis.security.file_utils import secure_create

            secure_create(Path(self._db_path))
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TASKS_TABLE)
        self._conn.execute(_CREATE_LOGS_TABLE)
        self._migrate_security_columns()
        self._conn.commit()

    def _migrate_security_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(scheduled_tasks)"
            ).fetchall()
        }
        migrations = {
            "operator_id": "TEXT NOT NULL DEFAULT ''",
            "capabilities": "TEXT NOT NULL DEFAULT '[]'",
            "consent": "TEXT NOT NULL DEFAULT '{}'",
            "claim_token": "TEXT",
            "claim_started_at": "TEXT",
        }
        for column, declaration in migrations.items():
            if column not in columns:
                self._conn.execute(
                    f"ALTER TABLE scheduled_tasks ADD COLUMN {column} {declaration}"
                )

    # -- Task CRUD -----------------------------------------------------------

    @staticmethod
    def _task_values(task: Dict[str, Any]) -> tuple[Any, ...]:
        return (
            task["id"],
            task["prompt"],
            task["schedule_type"],
            task["schedule_value"],
            task.get("context_mode", "isolated"),
            task.get("status", "active"),
            task.get("next_run"),
            task.get("last_run"),
            task.get("agent", "simple"),
            task.get("tools", ""),
            json.dumps(task.get("metadata", {})),
            task.get("operator_id", ""),
            json.dumps(task.get("capabilities", [])),
            json.dumps(task.get("consent", {})),
        )

    def create_task(self, task: Dict[str, Any]) -> bool:
        """Insert a task without overwriting a concurrently-created ID."""
        try:
            self._conn.execute(_INSERT_TASK, self._task_values(task))
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            self._conn.rollback()
            return False

    def save_task(self, task: Dict[str, Any]) -> None:
        """Insert or update a task without crossing operator ownership."""
        cursor = self._conn.execute(_UPSERT_TASK, self._task_values(task))
        if cursor.rowcount != 1:
            self._conn.rollback()
            raise PermissionError("Scheduled task ID is owned by another operator")
        self._conn.commit()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single task by ID, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all tasks, optionally filtered by *status*."""
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM scheduled_tasks WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM scheduled_tasks").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_due_tasks(self, now_iso: str) -> List[Dict[str, Any]]:
        """Return active tasks whose ``next_run`` is at or before *now_iso*."""
        rows = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE status = 'active' "
            "AND next_run IS NOT NULL AND next_run <= ?",
            (now_iso,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def claim_due_task(
        self,
        now_iso: str,
        claim_token: str,
    ) -> Optional[Dict[str, Any]]:
        """Atomically transition one due task from active to running.

        A crashed worker intentionally leaves the task in ``running`` for
        manual reconciliation.  Automatically expiring a lease could replay
        an action whose external side effect actually completed.
        """
        if not claim_token:
            raise ValueError("Scheduler claim token is required")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM scheduled_tasks "
                "WHERE status = 'active' AND next_run IS NOT NULL "
                "AND next_run <= ? ORDER BY next_run, id LIMIT 1",
                (now_iso,),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            cursor = self._conn.execute(
                "UPDATE scheduled_tasks "
                "SET status = 'running', claim_token = ?, claim_started_at = ? "
                "WHERE id = ? AND status = 'active'",
                (claim_token, now_iso, row["id"]),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                return None
            claimed = self._conn.execute(
                "SELECT * FROM scheduled_tasks WHERE id = ?",
                (row["id"],),
            ).fetchone()
            self._conn.commit()
            return self._row_to_dict(claimed)
        except Exception:
            self._conn.rollback()
            raise

    def claim_task(
        self,
        task_id: str,
        claim_token: str,
        claimed_at: str,
    ) -> Optional[Dict[str, Any]]:
        """Atomically claim a specific active task for guarded execution."""
        if not claim_token:
            raise ValueError("Scheduler claim token is required")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                "UPDATE scheduled_tasks "
                "SET status = 'running', claim_token = ?, claim_started_at = ? "
                "WHERE id = ? AND status = 'active'",
                (claim_token, claimed_at, task_id),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                return None
            row = self._conn.execute(
                "SELECT * FROM scheduled_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            self._conn.commit()
            return self._row_to_dict(row)
        except Exception:
            self._conn.rollback()
            raise

    def finish_claim(
        self,
        task_id: str,
        claim_token: str,
        *,
        status: str,
        next_run: Optional[str],
        last_run: str,
        consent: Dict[str, Any],
    ) -> bool:
        """Finish only the durable claim held by *claim_token*."""
        cursor = self._conn.execute(
            "UPDATE scheduled_tasks SET status = ?, next_run = ?, "
            "last_run = ?, consent = ?, claim_token = NULL, "
            "claim_started_at = NULL "
            "WHERE id = ? AND status = 'running' AND claim_token = ?",
            (
                status,
                next_run,
                last_run,
                json.dumps(consent),
                task_id,
                claim_token,
            ),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def update_task(self, task: Dict[str, Any]) -> None:
        """Update an existing task through a non-destructive UPSERT."""
        self.save_task(task)

    def delete_task(self, task_id: str) -> None:
        """Delete a task by ID."""
        self._conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        self._conn.commit()

    def pause_active_task(self, task_id: str, operator_id: str) -> bool:
        """Pause only an unclaimed active task."""
        cursor = self._conn.execute(
            "UPDATE scheduled_tasks SET status = 'paused' "
            "WHERE id = ? AND operator_id = ? AND status = 'active' "
            "AND claim_token IS NULL",
            (task_id, operator_id),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def resume_paused_task(
        self,
        task_id: str,
        operator_id: str,
        *,
        schedule_value: str,
        next_run: Optional[str],
    ) -> bool:
        """Resume only an unclaimed paused task."""
        cursor = self._conn.execute(
            "UPDATE scheduled_tasks SET status = 'active', "
            "schedule_value = ?, next_run = ? "
            "WHERE id = ? AND operator_id = ? AND status = 'paused' "
            "AND claim_token IS NULL",
            (schedule_value, next_run, task_id, operator_id),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def cancel_unclaimed_task(self, task_id: str, operator_id: str) -> bool:
        """Cancel without racing or overwriting a worker's durable claim."""
        cursor = self._conn.execute(
            "UPDATE scheduled_tasks SET status = 'cancelled', next_run = NULL "
            "WHERE id = ? AND operator_id = ? "
            "AND status IN ('active', 'paused', 'completed') "
            "AND claim_token IS NULL",
            (task_id, operator_id),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    # -- Run logs ------------------------------------------------------------

    def log_run(
        self,
        task_id: str,
        started_at: str,
        finished_at: str,
        success: bool,
        result: str = "",
        error: str = "",
    ) -> None:
        """Record a single execution of a task."""
        self._conn.execute(
            _INSERT_LOG,
            (task_id, started_at, finished_at, int(success), result, error),
        )
        self._conn.commit()

    def get_run_logs(self, task_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent run logs for *task_id*."""
        rows = self._conn.execute(
            "SELECT * FROM task_run_logs WHERE task_id = ? ORDER BY id DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key, fallback in (
            ("metadata", {}),
            ("capabilities", []),
            ("consent", {}),
        ):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = fallback
        return d


__all__ = ["SchedulerStore"]
