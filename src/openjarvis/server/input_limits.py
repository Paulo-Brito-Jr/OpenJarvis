"""Small, dependency-free limits shared by HTTP/WebSocket boundaries."""

from __future__ import annotations

MAX_WS_FRAME_BYTES = 64 * 1024
MAX_WS_MESSAGE_BYTES = 32 * 1024
MAX_WS_FILTER_BYTES = 256


def utf8_size_exceeds(value: object, limit: int) -> bool:
    """Return True for non-strings or strings larger than *limit* UTF-8 bytes."""
    if not isinstance(value, str) or limit < 0:
        return True
    # UTF-8 uses at least one byte per code point.  This avoids encoding an
    # already-obviously-oversized attacker-controlled value.
    if len(value) > limit:
        return True
    return len(value.encode("utf-8")) > limit
