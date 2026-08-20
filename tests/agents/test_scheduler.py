"""Tests for AgentScheduler tick scheduling."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def manager():
    from openjarvis.agents.manager import AgentManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AgentManager(db_path=str(Path(tmpdir) / "agents.db"))
        yield mgr
        mgr.close()


class TestSchedulerBasic:
    def test_create_scheduler(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(manager=manager, executor=executor)
        assert scheduler is not None

    def test_register_agent_with_interval(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(manager=manager, executor=executor)

        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={"schedule_type": "interval", "schedule_value": 60},
        )
        scheduler.register_agent(agent["id"])
        assert agent["id"] in scheduler.registered_agents

    def test_register_agent_with_cron(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(manager=manager, executor=executor)

        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={"schedule_type": "cron", "schedule_value": "0 9 * * *"},
        )
        try:
            import croniter  # noqa: F401
        except ImportError:
            with pytest.raises(RuntimeError, match="requires.*croniter"):
                scheduler.register_agent(agent["id"])
        else:
            scheduler.register_agent(agent["id"])
            assert agent["id"] in scheduler.registered_agents

    def test_deregister_agent(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(manager=manager, executor=executor)

        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={"schedule_type": "interval", "schedule_value": 60},
        )
        scheduler.register_agent(agent["id"])
        scheduler.deregister_agent(agent["id"])
        assert agent["id"] not in scheduler.registered_agents

    def test_manual_schedule_not_auto_registered(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(manager=manager, executor=executor)

        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={"schedule_type": "manual"},
        )
        scheduler.register_agent(agent["id"])
        # Manual agents are registered but never auto-fired
        assert agent["id"] in scheduler.registered_agents

    def test_start_stop(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(manager=manager, executor=executor)
        with pytest.raises(RuntimeError, match="durable multi-process"):
            scheduler.start()
        assert not scheduler.is_running
        scheduler.stop()
        assert not scheduler.is_running

    def test_automatic_tick_execution_is_disabled(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(
            manager=manager, executor=executor, tick_interval=0.1
        )

        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={"schedule_type": "interval", "schedule_value": 0.1},
        )
        scheduler.register_agent(agent["id"])
        with pytest.raises(RuntimeError, match="durable multi-process"):
            scheduler.start()

        executor.execute_tick.assert_not_called()

    def test_skips_paused_agents(self, manager):
        from openjarvis.agents.scheduler import AgentScheduler

        executor = MagicMock()
        scheduler = AgentScheduler(
            manager=manager, executor=executor, tick_interval=0.1
        )

        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={"schedule_type": "interval", "schedule_value": 0.01},
        )
        manager.pause_agent(agent["id"])
        scheduler.register_agent(agent["id"])
        time.sleep(0.02)
        scheduler._check_due_agents()

        executor.execute_tick.assert_not_called()

    @pytest.mark.parametrize("schedule_value", [0, -1, "nan", "inf"])
    def test_register_rejects_invalid_interval(self, manager, schedule_value):
        from openjarvis.agents.scheduler import AgentScheduler

        scheduler = AgentScheduler(manager=manager, executor=MagicMock())
        agent = manager.create_agent(
            name="test",
            agent_type="monitor_operative",
            config={
                "schedule_type": "interval",
                "schedule_value": schedule_value,
            },
        )

        with pytest.raises(ValueError, match="positive number"):
            scheduler.register_agent(agent["id"])
