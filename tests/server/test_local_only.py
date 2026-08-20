"""Fail-closed tests for the server's local-only configuration."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.core.config import (  # noqa: E402
    HardwareInfo,
    JarvisConfig,
    ServerConfig,
    generate_default_toml,
)
from openjarvis.server.app import create_app  # noqa: E402
from openjarvis.server.routes import reload_cloud_engine  # noqa: E402


def _local_only_config() -> JarvisConfig:
    config = JarvisConfig()
    config.analytics.enabled = False
    config.traces.enabled = False
    config.server.cloud_enabled = False
    return config


def _engine_that_must_not_run() -> MagicMock:
    engine = MagicMock()
    engine.generate.side_effect = AssertionError("local engine must not run")
    engine.stream.side_effect = AssertionError("local stream must not run")
    engine.stream_full.side_effect = AssertionError("tool stream must not run")
    return engine


def test_cloud_enabled_defaults_true_for_upstream_compatibility() -> None:
    assert ServerConfig().cloud_enabled is True
    default_toml = generate_default_toml(HardwareInfo())
    assert "cloud_enabled = true" in default_toml


@pytest.mark.parametrize("stream", [False, True])
def test_local_only_rejects_cloud_chat_before_dispatch(stream: bool) -> None:
    engine = _engine_that_must_not_run()
    app = create_app(engine, "local-model", config=_local_only_config())
    client = TestClient(app)

    with patch("openjarvis.server.cloud_router.stream_cloud") as stream_cloud:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Cloud models are disabled by server configuration."
    )
    engine.generate.assert_not_called()
    engine.stream.assert_not_called()
    engine.stream_full.assert_not_called()
    stream_cloud.assert_not_called()


def test_local_only_still_allows_local_chat() -> None:
    engine = MagicMock()
    engine.generate.return_value = {
        "content": "local",
        "usage": {},
        "finish_reason": "stop",
    }
    app = create_app(engine, "local-model", config=_local_only_config())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    engine.generate.assert_called_once()


def test_local_only_rejects_cloud_reload_before_keys_or_files() -> None:
    config = _local_only_config()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=config)),
        json=AsyncMock(
            return_value={"keys": {"OPENJARVIS_TEST_API_KEY": "must-not-be-written"}}
        ),
    )
    sentinel_key = "OPENJARVIS_TEST_API_KEY"

    with (
        patch.dict(os.environ, {sentinel_key: "unchanged"}, clear=False),
        patch("openjarvis.engine.cloud.CloudEngine") as cloud_engine,
        patch("openjarvis.server.routes.get_config_dir") as config_dir,
    ):
        before = dict(os.environ)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(reload_cloud_engine(request))
        after = dict(os.environ)

    assert exc_info.value.status_code == 403
    assert before == after
    request.json.assert_not_awaited()
    cloud_engine.assert_not_called()
    config_dir.assert_not_called()
