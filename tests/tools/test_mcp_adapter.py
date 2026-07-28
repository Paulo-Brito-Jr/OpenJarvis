"""Tests for the MCP tool adapter — round-trip through MCPServer + MCPClient."""

from __future__ import annotations

import pytest

from openjarvis.mcp.client import MCPClient
from openjarvis.mcp.server import MCPServer
from openjarvis.mcp.transport import InProcessTransport
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolSpec
from openjarvis.tools.calculator import CalculatorTool
from openjarvis.tools.mcp_adapter import MCPToolAdapter, MCPToolProvider
from openjarvis.tools.think import ThinkTool

_NAMESPACE = "test_server"
_PROVENANCE = "inprocess:test"


def _adapter(client, spec):
    return MCPToolAdapter(
        client,
        spec,
        namespace=_NAMESPACE,
        provenance=_PROVENANCE,
    )


def _provider(client):
    return MCPToolProvider(
        client,
        namespace=_NAMESPACE,
        provenance=_PROVENANCE,
    )


def _server(tools):
    policy = CapabilityPolicy()
    policy.grant("mcp-adapter-test", "*", "*")
    return MCPServer(
        tools,
        capability_policy=policy,
        agent_id="mcp-adapter-test",
    )


@pytest.fixture
def server():
    """Create an MCP server with calculator and think tools."""
    return _server([CalculatorTool(), ThinkTool()])


@pytest.fixture
def client(server):
    """Create an MCP client connected to the server via in-process transport."""
    transport = InProcessTransport(server)
    c = MCPClient(transport)
    c.initialize()
    return c


class TestMCPToolAdapter:
    def test_adapter_spec(self, client):
        spec = ToolSpec(
            name="calculator",
            description="Evaluate a math expression",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        )
        adapter = _adapter(client, spec)
        assert adapter.spec.name == "mcp__test_server__calculator"
        assert adapter.spec.description == "Evaluate a math expression"
        assert adapter.spec.requires_confirmation is True
        assert "mcp:test_server:invoke" in adapter.spec.required_capabilities
        assert adapter.is_local is False

    def test_adapter_execute_success(self, client):
        spec = ToolSpec(
            name="calculator",
            description="Calculator",
            parameters={},
        )
        adapter = _adapter(client, spec)
        result = adapter.execute(expression="2+2")
        assert result.success is True
        assert "4" in result.content
        assert result.tool_name == "mcp__test_server__calculator"
        assert result.metadata["_taint"]["labels"] == ["external"]

    def test_adapter_execute_think(self, client):
        spec = ToolSpec(
            name="think",
            description="Think tool",
            parameters={},
        )
        adapter = _adapter(client, spec)
        result = adapter.execute(thought="Let me analyze this.")
        assert result.success is True
        assert "analyze" in result.content

    def test_adapter_execute_error(self, client):
        spec = ToolSpec(
            name="calculator",
            description="Calculator",
            parameters={},
        )
        adapter = _adapter(client, spec)
        result = adapter.execute(expression="1/0")
        assert result.success is True
        assert result.content == "inf"

    def test_adapter_unknown_tool(self, client):
        spec = ToolSpec(
            name="nonexistent",
            description="Does not exist",
            parameters={},
        )
        adapter = _adapter(client, spec)
        result = adapter.execute(x="y")
        assert result.success is False
        assert "error" in result.content.lower() or "MCP" in result.content

    def test_adapter_tool_id(self, client):
        spec = ToolSpec(name="calculator", description="Calc", parameters={})
        adapter = _adapter(client, spec)
        assert adapter.tool_id == "mcp_adapter"


class TestMCPToolProvider:
    def test_discover_returns_adapters(self, client):
        provider = _provider(client)
        tools = provider.discover()
        assert len(tools) == 2
        names = {t.spec.name for t in tools}
        assert "mcp__test_server__calculator" in names
        assert "mcp__test_server__think" in names

    def test_discovered_tools_are_base_tool(self, client):
        from openjarvis.tools._stubs import BaseTool

        provider = _provider(client)
        tools = provider.discover()
        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_discovered_tool_execution(self, client):
        provider = _provider(client)
        tools = provider.discover()
        calc = next(
            t for t in tools
            if t.spec.name == "mcp__test_server__calculator"
        )
        result = calc.execute(expression="3*7")
        assert result.success is True
        assert "21" in result.content


class TestMCPAdapterRoundTrip:
    """End-to-end: Server → InProcessTransport → Client → Adapter → execute."""

    def test_full_round_trip(self):
        server = _server([CalculatorTool(), ThinkTool()])
        transport = InProcessTransport(server)
        client = MCPClient(transport)
        client.initialize()
        provider = _provider(client)
        tools = provider.discover()

        calc = next(
            t for t in tools
            if t.spec.name == "mcp__test_server__calculator"
        )
        result = calc.execute(expression="10+20")
        assert result.success is True
        assert "30" in result.content

    def test_round_trip_error_handling(self):
        server = _server([CalculatorTool()])
        transport = InProcessTransport(server)
        client = MCPClient(transport)
        client.initialize()
        provider = _provider(client)
        tools = provider.discover()

        calc = tools[0]
        result = calc.execute(expression="1/0")
        assert result.success is True
        assert result.content == "inf"

    def test_empty_server_discover(self):
        server = MCPServer([])
        transport = InProcessTransport(server)
        client = MCPClient(transport)
        client.initialize()
        provider = _provider(client)
        tools = provider.discover()
        assert tools == []

    def test_duplicate_remote_tool_names_are_rejected(self, client):
        client.list_tools = lambda: [
            ToolSpec(name="same", description="", parameters={}),
            ToolSpec(name="same", description="", parameters={}),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            _provider(client).discover()

    def test_ambiguous_normalized_names_are_rejected(self, client):
        client.list_tools = lambda: [
            ToolSpec(name="foo-bar", description="", parameters={}),
            ToolSpec(name="foo.bar", description="", parameters={}),
        ]
        with pytest.raises(ValueError, match="ambiguous"):
            _provider(client).discover()
