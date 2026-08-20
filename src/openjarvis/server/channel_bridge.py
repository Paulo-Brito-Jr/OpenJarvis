"""ChannelBridge — unified orchestrator for multi-channel messaging."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

from openjarvis.channels._stubs import BaseChannel, ChannelStatus
from openjarvis.core.events import EventBus, EventType
from openjarvis.server.session_store import SessionStore

logger = logging.getLogger(__name__)

_DEFAULT_MAX_LENGTH = 4000
_SMS_MAX_LENGTH = 1600

_HELP_TEXT = """\
Available commands:
/agents — list running agents
/agent <id> status — agent state and current task
/agent <id> <message> — unavailable until operator identity is preserved
/agent <id> pause — pause an agent
/agent <id> resume — resume an agent
/notify <channel> — set where to receive notifications
/sessions — list your active sessions
/more — get the rest of a truncated response
/help — show this message\
"""

# Events the bridge subscribes to for notifications
_NOTIFICATION_EVENTS = [
    EventType.AGENT_TICK_END,
    EventType.AGENT_TICK_ERROR,
    EventType.AGENT_BUDGET_EXCEEDED,
    EventType.SCHEDULER_TASK_END,
]


class ChannelBridge:
    """Orchestrates incoming messages across multiple channel adapters.

    Provides backward-compatible ``send()``/``status()``/``list_channels()``
    so it can replace the old single-channel bridge in ``app.state``.
    """

    def __init__(
        self,
        channels: Dict[str, BaseChannel],
        session_store: SessionStore,
        bus: EventBus,
        system: Any = None,
        agent_manager: Any = None,
        deep_research_agent: Any = None,
        sender_allowlist: Mapping[str, Iterable[str]] | None = None,
        capability_policy: Any = None,
    ) -> None:
        self._channels = channels
        self._session_store = session_store
        self._bus = bus
        self._system = system
        self._agent_manager = agent_manager
        self._deep_research_agent = deep_research_agent
        self._capability_policy = capability_policy
        self._sender_allowlist: dict[str, frozenset[str]] = {}
        for channel, senders in (sender_allowlist or {}).items():
            self.set_sender_allowlist(channel, senders)
        self._notification_timestamps: Dict[str, float] = {}
        self._subscribe_notifications()

    @staticmethod
    def principal_for(channel_type: str, sender_id: str) -> str:
        """Return a stable, non-PII authorization principal for a sender."""
        channel = channel_type.strip().lower()
        sender = sender_id.strip()
        if not channel or not sender:
            return ""
        digest = hashlib.sha256(f"{channel}\0{sender}".encode()).hexdigest()[:24]
        return f"channel:{channel}:{digest}"

    def set_sender_allowlist(
        self,
        channel_type: str,
        senders: Iterable[str],
    ) -> None:
        """Replace the exact sender allowlist for one channel."""
        channel = channel_type.strip().lower()
        if not channel:
            raise ValueError("channel_type must be non-empty")
        allowed: set[str] = set()
        for sender in senders:
            if not isinstance(sender, str):
                raise TypeError("sender allowlist entries must be strings")
            normalized = sender.strip()
            if (
                not normalized
                or len(normalized) > 256
                or any(ord(char) < 32 for char in normalized)
            ):
                raise ValueError("sender allowlist contains an invalid identity")
            allowed.add(normalized)
        self._sender_allowlist[channel] = frozenset(allowed)

    def is_sender_allowed(self, channel_type: str, sender_id: str) -> bool:
        """Return whether the exact channel/sender pair is trusted."""
        channel = channel_type.strip().lower()
        sender = sender_id.strip()
        return bool(
            channel
            and sender
            and sender in self._sender_allowlist.get(channel, frozenset())
        )

    def _authorized(
        self,
        principal: str,
        capability: str,
        resource: str,
    ) -> bool:
        if not principal or self._capability_policy is None:
            return False
        try:
            grants = self._capability_policy.list_grants(principal)
            explicitly_granted = any(
                fnmatch.fnmatch(capability, grant.capability)
                and (grant.pattern == "*" or fnmatch.fnmatch(resource, grant.pattern))
                for grant in grants
            )
            if not explicitly_granted:
                return False
            return bool(
                self._capability_policy.check(
                    principal,
                    capability,
                    resource,
                )
            )
        except Exception:
            logger.warning("Channel capability check failed; action denied")
            return False

    # --------------------------------------------------------------
    # Backward-compatible BaseChannel interface
    # --------------------------------------------------------------

    def connect(self) -> None:
        for ch in self._channels.values():
            ch.connect()

    def disconnect(self) -> None:
        for ch in self._channels.values():
            ch.disconnect()

    def list_channels(self) -> List[str]:
        result: List[str] = []
        for ch in self._channels.values():
            result.extend(ch.list_channels())
        return result

    def status(self) -> ChannelStatus:
        statuses = [ch.status() for ch in self._channels.values()]
        if not statuses:
            return ChannelStatus.DISCONNECTED
        if any(s == ChannelStatus.CONNECTED for s in statuses):
            return ChannelStatus.CONNECTED
        if all(s == ChannelStatus.ERROR for s in statuses):
            return ChannelStatus.ERROR
        return ChannelStatus.DISCONNECTED

    def send(
        self,
        channel: str,
        content: str,
        *,
        conversation_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        for ch in self._channels.values():
            if channel in ch.list_channels():
                return ch.send(
                    channel,
                    content,
                    conversation_id=conversation_id,
                    metadata=metadata,
                )
        logger.warning("No adapter found for channel %s", channel)
        return False

    # --------------------------------------------------------------
    # Incoming message handling
    # --------------------------------------------------------------

    def handle_incoming(
        self,
        sender_id: str,
        content: str,
        channel_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> str:
        if not self.is_sender_allowed(channel_type, sender_id):
            logger.warning(
                "Rejected inbound message from a sender outside the %s allowlist",
                channel_type,
            )
            return "Unauthorized sender."
        principal = self.principal_for(channel_type, sender_id)
        self._session_store.get_or_create(sender_id, channel_type)

        # Command routing
        stripped = content.strip()
        if stripped.startswith("/"):
            result = self._handle_command(
                sender_id,
                stripped,
                channel_type,
                principal,
            )
            if result is not None:
                return result

        # Regular chat — route to JarvisSystem.ask()
        return self._handle_chat(
            sender_id,
            stripped,
            channel_type,
            max_length,
            principal,
        )

    # --------------------------------------------------------------
    # Command parsing
    # --------------------------------------------------------------

    def _handle_command(
        self,
        sender_id: str,
        content: str,
        channel_type: str,
        principal: str,
    ) -> Optional[str]:
        parts = content.split(None, 2)
        cmd = parts[0].lower()

        if cmd == "/help":
            return _HELP_TEXT

        if cmd == "/more":
            return self._handle_more(sender_id, channel_type)

        if cmd == "/notify" and len(parts) >= 2:
            pref = parts[1]
            self._session_store.set_notification_preference(
                sender_id, channel_type, pref
            )
            return f"Notifications will be sent to {pref}."

        if cmd == "/sessions":
            return self._handle_sessions(sender_id)

        if cmd == "/agents":
            if not self._authorized(principal, "system:admin", "agents:list"):
                return "Not authorized."
            return self._handle_agents_list()

        if cmd == "/agent" and len(parts) >= 2:
            agent_id = parts[1]
            rest = parts[2] if len(parts) > 2 else "status"
            if not self._authorized(
                principal,
                "system:admin",
                f"agent:{agent_id}",
            ):
                return "Not authorized."
            return self._handle_agent_command(agent_id, rest)

        # Unknown command — fall through to chat
        return None

    def _handle_more(self, sender_id: str, channel_type: str) -> str:
        session = self._session_store.get_or_create(sender_id, channel_type)
        pending = session.get("pending_response")
        if pending:
            self._session_store.clear_pending_response(sender_id, channel_type)
            return pending
        return "No pending response."

    def _handle_agents_list(self) -> str:
        if not self._agent_manager:
            return "No agent manager configured."
        agents = self._agent_manager.list_agents()
        if not agents:
            return "No agents currently running."
        lines = []
        for a in agents:
            name = a.get("name", a.get("agent_id", "unknown"))
            status = a.get("status", "unknown")
            lines.append(f"  {name} — {status}")
        return "Running agents:\n" + "\n".join(lines)

    def _handle_agent_command(self, agent_id: str, action: str) -> str:
        if not self._agent_manager:
            return "No agent manager configured."
        action_lower = action.strip().lower()
        if action_lower == "status":
            state = self._agent_manager.get_agent(agent_id)
            if state is None:
                return f"Agent '{agent_id}' not found."
            name = state.get("name", agent_id)
            status = state.get("status", "unknown")
            return f"Agent '{name}': {status}"
        if action_lower == "pause":
            self._agent_manager.pause_agent(agent_id)
            return f"Agent '{agent_id}' paused."
        if action_lower == "resume":
            self._agent_manager.resume_agent(agent_id)
            return f"Agent '{agent_id}' resumed."
        # Pending managed-agent messages do not yet store the authenticated
        # channel principal.  Enqueuing here would make a later service-agent
        # tick execute with the agent's grants (a confused deputy).
        return (
            "Agent messaging from channels is disabled until queued messages "
            "preserve the authenticated operator identity."
        )

    # --------------------------------------------------------------
    # Chat handling
    # --------------------------------------------------------------

    def _handle_sessions(self, sender_id: str) -> str:
        targets = self._session_store.get_notification_targets()
        user_sessions = [t for t in targets if t["sender_id"] == sender_id]
        if not user_sessions:
            return "No active sessions with notification preferences."
        lines = []
        for s in user_sessions:
            lines.append(
                f"  {s['channel_type']} -> "
                f"notifications: {s['preferred_notification_channel']}"
            )
        return "Your sessions:\n" + "\n".join(lines)

    def _handle_chat(
        self,
        sender_id: str,
        content: str,
        channel_type: str,
        max_length: int,
        principal: str,
    ) -> str:
        self._session_store.append_message(sender_id, channel_type, "user", content)

        # Build context from conversation history
        session = self._session_store.get_or_create(sender_id, channel_type)
        history = session.get("conversation_history", [])
        context_lines = []
        for msg in history[:-1]:  # exclude the message we just appended
            context_lines.append(f"{msg['role']}: {msg['content']}")
        context_str = "\n".join(context_lines)

        query = content
        if context_str:
            query = (
                f"Previous conversation:\n{context_str}\n\nCurrent message: {content}"
            )

        # A shared DeepResearchAgent carries mutable executor security state.
        # Rebinding it per message would race with concurrent callers and
        # running it under its configured agent identity would make it a
        # confused deputy. Until the bridge can construct an isolated agent
        # per authenticated sender, never invoke that shared instance.
        if self._deep_research_agent is not None:
            logger.warning(
                "Shared DeepResearch execution is disabled for external channels"
            )

        if self._system is not None:
            if not self._authorized(
                principal,
                "tool:invoke",
                "agent:system",
            ):
                response_text = "Not authorized."
                self._session_store.append_message(
                    sender_id,
                    channel_type,
                    "assistant",
                    response_text,
                )
                return response_text
            try:
                result = self._system.ask(query, operator_id=principal)
                response_text = result.get("content", str(result))
            except Exception as exc:
                logger.error(
                    "JarvisSystem channel request failed (%s)",
                    type(exc).__name__,
                )
                error_msg = (
                    "Sorry, I couldn't process that right now. Try again in a moment."
                )
                self._session_store.append_message(
                    sender_id, channel_type, "assistant", error_msg
                )
                return error_msg
        elif self._deep_research_agent is not None:
            response_text = (
                "Deep research is unavailable on external channels until "
                "per-sender capability isolation is configured."
            )
            self._session_store.append_message(
                sender_id,
                channel_type,
                "assistant",
                response_text,
            )
            return response_text
        else:
            error_msg = (
                "Sorry, I couldn't process that right now. Try again in a moment."
            )
            self._session_store.append_message(
                sender_id, channel_type, "assistant", error_msg
            )
            return error_msg

        # Format and possibly truncate
        formatted = self._format_response(
            sender_id, channel_type, response_text, max_length
        )
        self._session_store.append_message(
            sender_id, channel_type, "assistant", response_text
        )
        return formatted

    def _format_response(
        self,
        sender_id: str,
        channel_type: str,
        response: str,
        max_length: int,
    ) -> str:
        if len(response) <= max_length:
            return response
        # Truncate and store full response for /more retrieval
        truncation_notice = "\n\n... (reply /more for full response)"
        cut_at = max_length - len(truncation_notice)
        truncated = response[:cut_at] + truncation_notice
        self._session_store.set_pending_response(sender_id, channel_type, response)
        return truncated

    # --------------------------------------------------------------
    # Notifications
    # --------------------------------------------------------------

    def _subscribe_notifications(self) -> None:
        for event_type in _NOTIFICATION_EVENTS:
            self._bus.subscribe(event_type, self._on_notification_event)

    def _on_notification_event(self, event) -> None:  # noqa: ANN001
        event_key = str(event.event_type)
        now = time.time()

        # Rate limit: max 1 per event type per 5 minutes
        last = self._notification_timestamps.get(event_key, 0)
        if now - last < 300:
            return
        self._notification_timestamps[event_key] = now

        message = self._format_notification(event)
        if not message:
            return

        targets = self._session_store.get_notification_targets()
        for target in targets:
            pref_channel = target["preferred_notification_channel"]
            sender_id = target["sender_id"]
            self._send_notification(pref_channel, sender_id, message)

    def _format_notification(  # noqa: ANN201
        self,
        event,  # noqa: ANN001
    ) -> Optional[str]:
        data = event.data or {}
        name = data.get("agent_name", data.get("name", "unknown"))

        if event.event_type == EventType.AGENT_TICK_END:
            summary = data.get("summary", data.get("result", ""))
            return f"Agent '{name}' finished." if summary else None
        if event.event_type == EventType.AGENT_TICK_ERROR:
            return f"Agent '{name}' failed."
        if event.event_type == EventType.AGENT_BUDGET_EXCEEDED:
            return f"Agent '{name}' hit budget limit."
        if event.event_type == EventType.SCHEDULER_TASK_END:
            if data.get("success", True):
                return f"Scheduled task '{name}' completed."
            return f"Scheduled task '{name}' failed."
        return None

    def _send_notification(
        self,
        channel_type: str,
        sender_id: str,
        message: str,
    ) -> None:
        ch = self._channels.get(channel_type)
        if ch is None:
            logger.warning(
                "No adapter for notification channel %s",
                channel_type,
            )
            return
        try:
            ch.send(sender_id, message)
        except Exception:
            logger.exception("Failed to send notification to %s", channel_type)
