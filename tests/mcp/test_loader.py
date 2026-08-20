"""Regression tests for openjarvis.mcp.loader.load_mcp_tools_from_config.

Closes the gap that #461 surfaced — MCP tools were silently dropped on
`jarvis ask` and `jarvis serve` because neither path read
`config.tools.mcp.servers`. The loader is the shared helper they now
both call.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_mcp_cfg(*, enabled=True, servers):
    """Build a duck-typed MCPConfig with enabled flag + servers JSON."""
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.servers = json.dumps(servers) if not isinstance(servers, str) else servers
    return cfg


def _fake_tool(name, *, namespace="x"):
    t = MagicMock()
    t.spec.name = f"mcp__{namespace}__{name}"
    t.spec.metadata = {"mcp": {"remote_name": name}}
    return t


@pytest.fixture
def _mock_mcp_stack():
    """Patch MCPClient / transports / MCPToolProvider so no real I/O happens."""
    with (
        patch("openjarvis.mcp.client.MCPClient") as MockClient,
        patch("openjarvis.mcp.transport.StreamableHTTPTransport") as MockHttp,
        patch("openjarvis.mcp.transport.StdioTransport") as MockStdio,
        patch("openjarvis.tools.mcp_adapter.MCPToolProvider") as MockProvider,
    ):
        # Default: any provider discovers no tools (per-test overrides as needed)
        MockProvider.return_value.discover.return_value = []
        MockClient.return_value.initialize.return_value = None
        yield {
            "client": MockClient,
            "http": MockHttp,
            "stdio": MockStdio,
            "provider": MockProvider,
        }


class TestLoaderEarlyReturns:
    def test_disabled_returns_empty(self, _mock_mcp_stack):
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = _make_mcp_cfg(enabled=False, servers=[{"url": "http://x"}])
        tools, clients = load_mcp_tools_from_config(cfg)
        assert tools == []
        assert clients == []
        # Critical: when disabled, we don't even instantiate transports
        _mock_mcp_stack["http"].assert_not_called()

    def test_empty_servers_returns_empty(self, _mock_mcp_stack):
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = _make_mcp_cfg(enabled=True, servers=[])
        tools, clients = load_mcp_tools_from_config(cfg)
        assert tools == []
        assert clients == []

    def test_malformed_json_fails_closed(self, _mock_mcp_stack):
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = MagicMock()
        cfg.enabled = True
        cfg.servers = "{not valid json"
        with pytest.raises(RuntimeError, match="parse MCP servers config"):
            load_mcp_tools_from_config(cfg)


class TestLoaderTokenPlumbing:
    def test_token_passed_to_streamable_http(self, _mock_mcp_stack):
        """Regression for #461 — token in cfg → StreamableHTTPTransport(token=...)."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[
                {
                    "name": "home_assistant",
                    "url": "http://homeassistant.local:8123/mcp",
                    "token": "ha-llat-secret",
                }
            ],
        )
        load_mcp_tools_from_config(cfg)
        _mock_mcp_stack["http"].assert_called_once_with(
            url="http://homeassistant.local:8123/mcp",
            token="ha-llat-secret",
        )

    def test_no_token_passes_none(self, _mock_mcp_stack):
        """Missing token in cfg → token=None (not 'undefined' or KeyError)."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[{"name": "open", "url": "http://localhost:9583/mcp"}],
        )
        load_mcp_tools_from_config(cfg)
        _mock_mcp_stack["http"].assert_called_once_with(
            url="http://localhost:9583/mcp",
            token=None,
        )

    def test_stdio_server_does_not_get_token_kwarg(self, _mock_mcp_stack):
        """StdioTransport doesn't take a token (unix-socket auth is OOB)."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[
                {"name": "local_mcp", "command": "mcp-server-foo", "args": ["--flag"]}
            ],
        )
        load_mcp_tools_from_config(cfg)
        _mock_mcp_stack["stdio"].assert_called_once_with(
            command=["mcp-server-foo", "--flag"]
        )


class TestLoaderFiltering:
    def test_allowed_names_filter_applied(self, _mock_mcp_stack):
        """allowed_names limits the returned tools to that set."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        _mock_mcp_stack["provider"].return_value.discover.return_value = [
            _fake_tool("alpha"),
            _fake_tool("beta"),
            _fake_tool("gamma"),
        ]
        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[{"name": "x", "url": "http://x"}],
        )
        tools, _ = load_mcp_tools_from_config(cfg, allowed_names={"beta"})
        assert [t.spec.name for t in tools] == ["mcp__x__beta"]

    def test_include_tools_per_server(self, _mock_mcp_stack):
        """Per-server include_tools restricts to just those names."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        _mock_mcp_stack["provider"].return_value.discover.return_value = [
            _fake_tool("alpha"),
            _fake_tool("beta"),
        ]
        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[{"name": "x", "url": "http://x", "include_tools": ["alpha"]}],
        )
        tools, _ = load_mcp_tools_from_config(cfg)
        assert [t.spec.name for t in tools] == ["mcp__x__alpha"]

    def test_exclude_tools_per_server(self, _mock_mcp_stack):
        """Per-server exclude_tools drops the named tools."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        _mock_mcp_stack["provider"].return_value.discover.return_value = [
            _fake_tool("alpha"),
            _fake_tool("beta"),
        ]
        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[{"name": "x", "url": "http://x", "exclude_tools": ["alpha"]}],
        )
        tools, _ = load_mcp_tools_from_config(cfg)
        assert [t.spec.name for t in tools] == ["mcp__x__beta"]


class TestLoaderClientLifetime:
    def test_returns_live_clients_for_caller_to_hold(self, _mock_mcp_stack):
        """Critical lifetime contract — caller must hold `clients` so
        the transports' httpx sessions stay open. (#461 adversarial
        review caught this.) The list returned MUST contain a client
        per successfully-initialized server."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[
                {"name": "s1", "url": "http://x1"},
                {"name": "s2", "url": "http://x2"},
            ],
        )
        _, clients = load_mcp_tools_from_config(cfg)
        assert len(clients) == 2


class TestLoaderFailureIsolation:
    def test_one_server_failure_aborts_batch_and_closes_clients(self, _mock_mcp_stack):
        """A partial MCP surface must never be exposed after discovery failure."""
        from openjarvis.mcp.loader import load_mcp_tools_from_config

        bad_client = MagicMock()
        bad_client.initialize.side_effect = RuntimeError("can't reach server")
        good_client = MagicMock()
        good_client.initialize.return_value = None
        _mock_mcp_stack["client"].side_effect = [bad_client, good_client]
        _mock_mcp_stack["provider"].return_value.discover.return_value = [
            _fake_tool("survivor")
        ]

        cfg = _make_mcp_cfg(
            enabled=True,
            servers=[
                {"name": "broken", "url": "http://broken"},
                {"name": "working", "url": "http://working"},
            ],
        )
        with pytest.raises(RuntimeError, match="can't reach server"):
            load_mcp_tools_from_config(cfg)
        bad_client.close.assert_called_once_with()
        good_client.initialize.assert_not_called()
