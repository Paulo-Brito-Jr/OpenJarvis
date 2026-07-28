"""Slack Socket Mode daemon — listens for DMs and responds with DeepResearch.

Run as a standalone process or import start_slack_daemon() to spawn
from the server. Uses slack-bolt for reliable Socket Mode handling.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

from openjarvis.core.paths import get_config_dir

logger = logging.getLogger(__name__)

_PID_FILE = str(get_config_dir() / "slack-daemon.pid")


def _to_slack_fmt(text: str) -> str:
    """Convert markdown to Slack mrkdwn format."""
    # Headers → bold
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # Bold: **text** → *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Strikethrough: ~~text~~ → ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)
    # Links: [text](url) → <url|text>
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"<\2|\1>", text)
    # Remove LaTeX
    text = re.sub(r"\$\$.+?\$\$", "", text)
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_slack_daemon(
    bot_token: str,
    app_token: str,
    model: str = "qwen3.5:9b",
    *,
    agent_id: str = "",
    allowed_sender_ids: list[str] | None = None,
) -> None:
    """Run the Slack daemon (blocking). Handles DMs with DeepResearch."""
    raise RuntimeError(
        "Slack daemon is disabled until every request is bound to the "
        "authenticated sender principal"
    )

    import threading

    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    from openjarvis.agents.deep_research import DeepResearchAgent
    from openjarvis.core.config import load_config
    from openjarvis.core.events import EventBus
    from openjarvis.engine.ollama import OllamaEngine
    from openjarvis.security import setup_security
    from openjarvis.server.agent_manager_routes import (
        _build_deep_research_tools,
    )
    from openjarvis.server.channel_bridge import ChannelBridge

    allowed_senders = frozenset(
        sender.strip()
        for sender in (allowed_sender_ids or [])
        if isinstance(sender, str) and sender.strip()
    )
    if not agent_id.strip():
        raise RuntimeError("Slack daemon requires an explicit agent identity")
    if not allowed_senders:
        raise RuntimeError("Slack daemon requires an explicit sender allowlist")

    # Write PID
    pid_path = Path(_PID_FILE)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    # Build agent
    config = load_config()
    bus = EventBus(record_history=False)
    engine = OllamaEngine()
    security = setup_security(config, engine, bus)
    engine = security.engine
    tools = _build_deep_research_tools(engine=engine, model=model)
    agent = DeepResearchAgent(
        engine=engine,
        model=model,
        tools=tools,
        bus=bus,
        max_turns=5,
    )
    agent.bind_security(
        security.capability_policy,
        agent_id,
        security.boundary_guard,
    )
    logger.info("Slack daemon: agent ready with %d tools", len(tools))

    app = App(token=bot_token)

    processing = threading.Event()

    @app.event("message")
    def handle_dm(event: dict, say: Any) -> None:
        text = event.get("text", "")
        sender_id = event.get("user", "")
        if (
            not text
            or event.get("bot_id")
            or event.get("subtype")
            or sender_id not in allowed_senders
        ):
            return
        principal = ChannelBridge.principal_for("slack", sender_id)
        if not security.capability_policy.check(
            principal,
            "tool:invoke",
            "agent:deep_research",
        ):
            logger.warning("Rejected unauthorized Slack sender")
            return

        ts = event.get("ts", "")
        logger.info("Slack DM accepted (%d chars)", len(text))

        say(text="Message received! Working on it now...", thread_ts=ts)
        processing.set()

        # Progress updater
        stop = threading.Event()

        def _progress() -> None:
            while not stop.is_set():
                stop.wait(60)
                if stop.is_set():
                    break
                if processing.is_set():
                    say(
                        text="Still working! Will reply ASAP",
                        thread_ts=ts,
                    )

        pt = threading.Thread(target=_progress, daemon=True)
        pt.start()

        try:
            result = agent.run(text)
            reply = _to_slack_fmt(result.content or "No results found.")
        except Exception as exc:
            reply = "Sorry, I couldn't process that request."
            logger.error("Slack daemon error: %s", type(exc).__name__)
        finally:
            processing.clear()
            stop.set()

        say(text=reply, thread_ts=ts)
        logger.info("Slack DM reply sent (%d chars)", len(reply))

    handler = SocketModeHandler(app, app_token)

    # Graceful shutdown
    def _stop(signum: int, frame: Any) -> None:
        logger.info("Slack daemon stopping...")
        handler.close()
        if pid_path.exists():
            pid_path.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info("Slack daemon started, listening for DMs...")
    handler.start()  # Blocks until close()


def start_slack_daemon(
    bot_token: str,
    app_token: str,
    model: str = "qwen3.5:9b",
    *,
    agent_id: str = "",
    allowed_sender_ids: list[str] | None = None,
) -> int:
    """Spawn the Slack daemon as a background subprocess.

    Returns the PID of the spawned process.
    """
    raise RuntimeError(
        "Slack daemon is disabled; credentials will not be placed in "
        "process arguments"
    )


def is_running() -> bool:
    """Check if the Slack daemon is running."""
    pid_path = Path(_PID_FILE)
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        pid_path.unlink(missing_ok=True)
        return False


def stop_daemon() -> bool:
    """Stop the running Slack daemon."""
    pid_path = Path(_PID_FILE)
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_path.unlink(missing_ok=True)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        pid_path.unlink(missing_ok=True)
        return False


if __name__ == "__main__":
    raise SystemExit(
        "Slack daemon is disabled until sender-scoped execution is available"
    )
