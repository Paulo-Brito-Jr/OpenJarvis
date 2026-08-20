"""MCP tool adapter — wraps external MCP server tools as native BaseTool instances."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, List

from openjarvis.core.types import ToolResult
from openjarvis.mcp.client import MCPClient
from openjarvis.security.taint import external_taint
from openjarvis.tools._stubs import BaseTool, ToolSpec

_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TOOL_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9_]+")
_MAX_MCP_RESULT_CHARS = 1_048_576


def validate_mcp_namespace(namespace: str) -> str:
    """Validate a stable namespace used in model-visible MCP tool names."""
    if not isinstance(namespace, str) or not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError("MCP namespace must match ^[a-z][a-z0-9_]{0,31}$")
    return namespace


def _exposed_tool_name(namespace: str, remote_name: str) -> str:
    if not isinstance(remote_name, str) or not remote_name.strip():
        raise ValueError("MCP tool name must be a non-empty string")
    segment = _TOOL_SEGMENT_RE.sub("_", remote_name.strip()).strip("_")
    if not segment:
        raise ValueError("MCP tool name has no safe identifier characters")
    exposed = f"mcp__{namespace}__{segment}"
    if len(exposed) > 64:
        raise ValueError("Namespaced MCP tool name exceeds 64 characters")
    return exposed


class MCPToolAdapter(BaseTool):
    """Wraps a single MCP-hosted tool as a native BaseTool.

    This adapter enables tools discovered from external MCP servers to
    be used seamlessly within OpenJarvis agents via the ``ToolExecutor``.

    Parameters
    ----------
    client:
        The ``MCPClient`` connected to the external MCP server.
    tool_spec:
        The ``ToolSpec`` describing this tool (from ``MCPClient.list_tools()``).
    """

    tool_id = "mcp_adapter"
    is_local = False

    def __init__(
        self,
        client: MCPClient,
        tool_spec: ToolSpec,
        *,
        namespace: str,
        provenance: str,
    ) -> None:
        self._client = client
        self._namespace = validate_mcp_namespace(namespace)
        self._remote_name = tool_spec.name
        self._provenance = provenance.strip() or "external-mcp"
        metadata = dict(tool_spec.metadata)
        metadata["mcp"] = {
            "namespace": self._namespace,
            "remote_name": self._remote_name,
            "provenance": self._provenance,
        }
        capabilities = list(
            dict.fromkeys(
                [
                    *tool_spec.required_capabilities,
                    "tool:invoke",
                    f"mcp:{self._namespace}:invoke",
                ]
            )
        )
        # MCP schemas cannot be trusted to accurately describe side effects.
        # Every opaque remote call therefore requires contemporaneous consent.
        self._spec = replace(
            tool_spec,
            name=_exposed_tool_name(self._namespace, self._remote_name),
            required_capabilities=capabilities,
            requires_confirmation=True,
            metadata=metadata,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, **params: Any) -> ToolResult:
        """Execute the remote MCP tool and return a ToolResult."""
        try:
            result = self._client.call_tool(self._remote_name, params)
            content_parts = result.get("content", [])
            text = "\n".join(
                p.get("text", "") for p in content_parts if isinstance(p, dict)
            )
            truncated = len(text) > _MAX_MCP_RESULT_CHARS
            if truncated:
                text = text[:_MAX_MCP_RESULT_CHARS]
            return ToolResult(
                tool_name=self._spec.name,
                content=text,
                success=not result.get("isError", False),
                metadata={
                    "mcp_namespace": self._namespace,
                    "remote_tool": self._remote_name,
                    "truncated": truncated,
                    **external_taint(f"mcp:{self._namespace}:{self._remote_name}"),
                },
            )
        except Exception:
            return ToolResult(
                tool_name=self._spec.name,
                content="MCP tool request failed.",
                success=False,
                metadata={
                    "mcp_namespace": self._namespace,
                    "remote_tool": self._remote_name,
                },
            )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        del params
        return f"mcp:{self._namespace}:{self._remote_name}"


class MCPToolProvider:
    """Discovers tools from an MCP server and returns BaseTool adapters.

    Parameters
    ----------
    client:
        The ``MCPClient`` connected to the MCP server.
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        namespace: str,
        provenance: str,
    ) -> None:
        self._client = client
        self._namespace = validate_mcp_namespace(namespace)
        self._provenance = provenance

    def discover(self) -> List[BaseTool]:
        """Discover available tools and return them as BaseTool adapters."""
        specs = self._client.list_tools()
        adapters: list[BaseTool] = []
        seen_remote: set[str] = set()
        seen_exposed: set[str] = set()
        for spec in specs:
            if spec.name in seen_remote:
                raise ValueError(
                    f"MCP server '{self._namespace}' advertised duplicate "
                    f"tool '{spec.name}'"
                )
            adapter = MCPToolAdapter(
                self._client,
                spec,
                namespace=self._namespace,
                provenance=self._provenance,
            )
            if adapter.spec.name in seen_exposed:
                raise ValueError(
                    f"MCP server '{self._namespace}' advertised ambiguous "
                    "tool names after namespacing"
                )
            seen_remote.add(spec.name)
            seen_exposed.add(adapter.spec.name)
            adapters.append(adapter)
        return adapters


__all__ = [
    "MCPToolAdapter",
    "MCPToolProvider",
    "validate_mcp_namespace",
]
