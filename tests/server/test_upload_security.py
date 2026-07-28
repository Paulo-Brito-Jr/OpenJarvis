"""Security limits for document ingestion endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server.upload_router import router  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    store = MagicMock()
    monkeypatch.setattr(
        "openjarvis.server.upload_router._get_store",
        MagicMock(return_value=store),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), store


def test_paste_limit_is_checked_before_store_creation(client):
    test_client, store = client

    response = test_client.post(
        "/v1/connectors/upload/ingest",
        json={"content": "x" * (1024 * 1024 + 1)},
    )

    assert response.status_code == 413
    store.store.assert_not_called()


def test_file_count_limit_is_checked_before_parsing(client):
    test_client, store = client
    files = [("files", (f"{index}.txt", b"safe", "text/plain")) for index in range(9)]

    response = test_client.post(
        "/v1/connectors/upload/ingest/files",
        files=files,
    )

    assert response.status_code == 413
    store.store.assert_not_called()


def test_file_size_limit_is_checked_before_parsing(client):
    test_client, store = client

    response = test_client.post(
        "/v1/connectors/upload/ingest/files",
        files={
            "files": (
                "large.txt",
                b"x" * (8 * 1024 * 1024 + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    store.store.assert_not_called()


def test_extension_and_content_type_must_agree(client):
    test_client, store = client

    response = test_client.post(
        "/v1/connectors/upload/ingest/files",
        files={"files": ("document.pdf", b"not-pdf", "text/plain")},
    )

    assert response.status_code == 415
    store.store.assert_not_called()
