"""One-time OAuth state storage for local server authorization flows."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OAuthAuthorization:
    connector_id: str
    principal: str
    expires_at: float


class OAuthStateStore:
    """Issue and atomically consume short-lived, process-local OAuth states."""

    def __init__(self, *, ttl_seconds: float = 600.0, max_pending: int = 256):
        if ttl_seconds <= 0 or max_pending <= 0:
            raise ValueError("OAuth state limits must be positive")
        self._ttl_seconds = float(ttl_seconds)
        self._max_pending = int(max_pending)
        self._pending: dict[str, OAuthAuthorization] = {}
        self._lock = threading.Lock()

    def issue(
        self,
        connector_id: str,
        principal: str,
        *,
        now: float | None = None,
    ) -> str:
        connector = connector_id.strip()
        operator = principal.strip()
        if not connector or not operator:
            raise ValueError("OAuth state requires connector and principal")
        issued_at = time.monotonic() if now is None else float(now)
        authorization = OAuthAuthorization(
            connector_id=connector,
            principal=operator,
            expires_at=issued_at + self._ttl_seconds,
        )
        with self._lock:
            self._prune_locked(issued_at)
            if len(self._pending) >= self._max_pending:
                # Refuse to evict a valid in-flight authorization silently.
                raise RuntimeError("Too many pending OAuth authorizations")
            state = secrets.token_urlsafe(32)
            while state in self._pending:
                state = secrets.token_urlsafe(32)
            self._pending[state] = authorization
        return state

    def consume(
        self,
        state: str,
        connector_id: str,
        *,
        now: float | None = None,
    ) -> OAuthAuthorization | None:
        candidate = state.strip()
        connector = connector_id.strip()
        if not candidate or not connector:
            return None
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            self._prune_locked(current)
            authorization = self._pending.pop(candidate, None)
        if authorization is None:
            return None
        if authorization.connector_id != connector:
            return None
        if authorization.expires_at <= current:
            return None
        return authorization

    def _prune_locked(self, now: float) -> None:
        expired = [
            state
            for state, authorization in self._pending.items()
            if authorization.expires_at <= now
        ]
        for state in expired:
            self._pending.pop(state, None)
