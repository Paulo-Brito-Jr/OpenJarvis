"""Startup quarantine for legacy managed-channel state."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def quarantine_legacy_sendblue_bindings(app: Any) -> None:
    """Intentionally avoid reading or reconnecting legacy SendBlue rows."""
    del app
    logger.warning(
        "Legacy SendBlue binding restore is disabled; bindings remain quarantined"
    )
