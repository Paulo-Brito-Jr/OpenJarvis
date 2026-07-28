"""Tests for speech backend auto-discovery."""

import builtins
from unittest.mock import MagicMock, patch

from openjarvis.core.config import JarvisConfig


def test_get_speech_backend_explicit():
    """Explicit backend selection works."""
    from openjarvis.speech._discovery import get_speech_backend

    config = JarvisConfig()
    config.speech.backend = "faster-whisper"

    with patch("openjarvis.speech._discovery._create_backend") as mock_create:
        mock_backend = type(
            "MockBackend",
            (),
            {
                "backend_id": "faster-whisper",
                "health": lambda self: True,
            },
        )()
        mock_create.return_value = mock_backend

        result = get_speech_backend(config)
        assert result is not None
        assert result.backend_id == "faster-whisper"


def test_get_speech_backend_returns_none_if_nothing_available():
    """Returns None when no backend can be created."""
    from openjarvis.speech._discovery import get_speech_backend

    config = JarvisConfig()
    config.speech.backend = "nonexistent"

    result = get_speech_backend(config)
    assert result is None


def test_auto_discovery_priority():
    """Auto mode tries backends in priority order."""
    from openjarvis.speech._discovery import DISCOVERY_ORDER

    assert DISCOVERY_ORDER[0] == "faster-whisper"
    assert "openai" in DISCOVERY_ORDER
    assert "deepgram" in DISCOVERY_ORDER


def test_disabled_backend_returns_none_without_registration_or_env(monkeypatch):
    import openjarvis.speech._discovery as discovery

    config = JarvisConfig()
    config.speech.backend = "disabled"
    guarded_environ = MagicMock()
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "openjarvis.speech":
            raise AssertionError("disabled speech must not register backends")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(discovery.os, "environ", guarded_environ)
    with (
        patch.object(discovery, "_create_backend") as create_backend,
        patch.object(builtins, "__import__", side_effect=guarded_import),
    ):
        result = discovery.get_speech_backend(config)

    assert result is None
    create_backend.assert_not_called()
    assert guarded_environ.mock_calls == []
