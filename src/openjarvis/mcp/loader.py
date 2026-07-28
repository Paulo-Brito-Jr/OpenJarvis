"""Shared helper for loading MCP server tools from a TOML config blob.

Used by ``cli/ask.py``, ``cli/serve.py``, ``system/builder.py`` and
``server/agent_manager_routes.py`` so each call site doesn't reimplement
the server-config → transport → client → discovered-tools pipeline.

The returned tuple of ``(tools, clients)`` is load-bearing: the caller
MUST hold a reference to ``clients`` for as long as the tools are used,
otherwise the MCP transport sessions get garbage-collected and the
underlying HTTP connections close mid-execution (see #461 adversarial
review). The recommended pattern is to stash the client list on the
agent so they share its lifetime:

    tools, mcp_clients = load_mcp_tools_from_config(config.tools.mcp)
    agent = AgentCls(tools=tools, ...)
    agent._mcp_clients = mcp_clients   # keep transports alive
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from openjarvis.core.types import ToolSpec  # noqa: F401
    from openjarvis.mcp.client import MCPClient
    from openjarvis.tools._stubs import BaseTool

logger = logging.getLogger(__name__)


def load_mcp_tools_from_config(
    mcp_cfg: Any,
    *,
    allowed_names: Optional[set[str]] = None,
) -> tuple[list["BaseTool"], list["MCPClient"]]:
    """Load tools from every server in ``mcp_cfg.servers``.

    Returns ``(tools, clients)``. ``clients`` is the list of live
    ``MCPClient`` instances — keep a reference or the transports get
    GC'd. Once MCP is configured, invalid/ambiguous configuration and
    discovery failures abort the batch so callers never run with a silently
    different tool surface.

    ``allowed_names`` is an outer filter applied after each server's
    own include/exclude filter. Pass the caller's `--tools`/`enabled`
    list to honour CLI scoping; pass ``None`` to take every tool.

    Returns ``([], [])`` when mcp is disabled or no servers are
    configured — no exception, no warning.
    """
    # ``enabled`` and ``servers`` come from openjarvis.core.config's
    # MCPConfig dataclass; accept duck-typed equivalents for tests.
    enabled = getattr(mcp_cfg, "enabled", False)
    servers_blob = getattr(mcp_cfg, "servers", None)
    if not enabled or not servers_blob:
        return [], []

    try:
        server_list = (
            json.loads(servers_blob) if isinstance(servers_blob, str) else servers_blob
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Failed to parse MCP servers config") from exc
    if not isinstance(server_list, list):
        raise RuntimeError("MCP servers config must be a JSON array")

    # Imported lazily so that `openjarvis.mcp.loader` can be imported
    # cheaply from CLI startup paths without dragging in the heavy MCP
    # client stack until something actually wants to discover tools.
    from openjarvis.mcp.client import MCPClient
    from openjarvis.mcp.transport import StdioTransport, StreamableHTTPTransport
    from openjarvis.tools.mcp_adapter import (
        MCPToolProvider,
        validate_mcp_namespace,
    )

    tools: list["BaseTool"] = []
    clients: list["MCPClient"] = []
    seen_servers: set[str] = set()
    seen_tools: set[str] = set()

    try:
        for server_cfg in server_list:
            cfg = json.loads(server_cfg) if isinstance(server_cfg, str) else server_cfg
            if not isinstance(cfg, dict):
                raise RuntimeError("Each MCP server config must be an object")
            name = validate_mcp_namespace(cfg.get("name", ""))
            if name in seen_servers:
                raise RuntimeError(f"Duplicate MCP namespace: {name}")
            seen_servers.add(name)
            url = cfg.get("url")
            token = cfg.get("token")
            command = cfg.get("command", "")
            args = cfg.get("args", [])
            if bool(url) == bool(command):
                raise RuntimeError(
                    f"MCP server '{name}' must configure exactly one transport"
                )
            if not isinstance(args, list) or not all(
                isinstance(arg, str) for arg in args
            ):
                raise RuntimeError(f"MCP server '{name}' args must be a string array")

            if url:
                from urllib.parse import urlsplit

                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise RuntimeError(f"MCP server '{name}' URL is invalid")
                transport = StreamableHTTPTransport(url=url, token=token)
                provenance = f"{parsed.scheme}://{parsed.hostname}"
            elif command:
                from pathlib import Path

                transport = StdioTransport(command=[command] + args)
                provenance = f"stdio:{Path(command).name}"
            else:
                raise AssertionError("transport validation should be exhaustive")

            client = MCPClient(transport)
            clients.append(client)
            client.initialize()

            provider = MCPToolProvider(
                client,
                namespace=name,
                provenance=provenance,
            )
            discovered = provider.discover()

            include_tools = set(cfg.get("include_tools", []))
            exclude_tools = set(cfg.get("exclude_tools", []))
            if include_tools:
                discovered = [
                    tool
                    for tool in discovered
                    if tool.spec.metadata["mcp"]["remote_name"] in include_tools
                ]
            if exclude_tools:
                discovered = [
                    tool
                    for tool in discovered
                    if tool.spec.metadata["mcp"]["remote_name"] not in exclude_tools
                ]
            if allowed_names:
                discovered = [
                    tool
                    for tool in discovered
                    if (
                        tool.spec.name in allowed_names
                        or tool.spec.metadata["mcp"]["remote_name"] in allowed_names
                    )
                ]

            for tool in discovered:
                if tool.spec.name in seen_tools:
                    raise RuntimeError(f"Duplicate MCP tool name: {tool.spec.name}")
                seen_tools.add(tool.spec.name)
                tools.append(tool)
            logger.info(
                "Discovered %d MCP tools from server '%s'", len(discovered), name
            )
    except Exception:
        for client in clients:
            try:
                client.close()
            except Exception:
                logger.debug("Failed to close MCP client after aborted load")
        raise

    return tools, clients
