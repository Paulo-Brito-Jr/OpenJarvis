"""HTTP request tool — make HTTP requests with SSRF protection."""

from __future__ import annotations

import logging
import socket
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.ssrf import check_ssrf, is_private_ip
from openjarvis.security.taint import external_taint
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

# Maximum response body size: 1 MB
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_REQUEST_BODY_BYTES = 262_144
_MAX_URL_LENGTH = 8_192
_MAX_HEADER_COUNT = 32
_MAX_HEADER_NAME_LENGTH = 128
_MAX_HEADER_VALUE_LENGTH = 8_192
_MAX_HEADER_BYTES = 32_768

_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"})

# Cap redirect chains so a malicious server cannot loop us indefinitely.
_MAX_REDIRECTS = 5

_CROSS_ORIGIN_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "accept-encoding",
        "accept-language",
        "cache-control",
        "if-modified-since",
        "if-none-match",
        "pragma",
        "range",
        "user-agent",
    }
)


class _SSRFRedirectError(Exception):
    """Raised when a redirect target fails the SSRF check."""


@dataclass(frozen=True)
class _StreamedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    truncated: bool
    logical_url: str


@ToolRegistry.register("http_request")
class HttpRequestTool(BaseTool):
    """Make HTTP requests to external APIs with SSRF protection."""

    tool_id = "http_request"
    is_local = False

    def __init__(
        self,
        *,
        resolver: Callable[..., list[tuple[Any, ...]]] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._resolver = resolver or socket.getaddrinfo
        self._transport = transport

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="http_request",
            description=(
                "Make an HTTP request to a URL."
                " Supports GET, POST, PUT, DELETE, PATCH,"
                " and HEAD methods. Includes SSRF protection"
                " against private IPs and cloud metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to send the request to.",
                    },
                    "method": {
                        "type": "string",
                        "description": (
                            "HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD)."
                            " Defaults to GET."
                        ),
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST, PUT, PATCH).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds. Defaults to 30.",
                    },
                },
                "required": ["url"],
            },
            category="network",
            required_capabilities=["network:fetch"],
        )

    def authorization_capabilities(self, params: dict[str, Any]) -> list[str]:
        capabilities = super().authorization_capabilities(params)
        method = str(params.get("method", "GET")).upper()
        if method not in {"GET", "HEAD"}:
            capabilities.append("network:mutate")
        return list(dict.fromkeys(capabilities))

    def requires_confirmation_for(self, params: dict[str, Any]) -> bool:
        method = str(params.get("method", "GET")).upper()
        return method not in {"GET", "HEAD"}

    def execute(self, **params: Any) -> ToolResult:
        url = params.get("url", "")
        if not url:
            return ToolResult(
                tool_name="http_request",
                content="No URL provided.",
                success=False,
            )
        if not isinstance(url, str) or len(url) > _MAX_URL_LENGTH:
            return ToolResult(
                tool_name="http_request",
                content="URL must be a string no longer than 8192 characters.",
                success=False,
            )

        method = params.get("method", "GET").upper()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                tool_name="http_request",
                content=(
                    f"Unsupported HTTP method: {method}."
                    f" Allowed: {', '.join(sorted(_ALLOWED_METHODS))}."
                ),
                success=False,
            )

        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ToolResult(
                tool_name="http_request",
                content=(
                    "URL must use HTTP(S), include a hostname, and must not "
                    "contain embedded credentials."
                ),
                success=False,
            )

        try:
            headers = self._validate_headers(params.get("headers") or {})
        except (TypeError, ValueError) as exc:
            return ToolResult(
                tool_name="http_request",
                content=f"Invalid request headers: {exc}",
                success=False,
            )
        body = params.get("body")
        if body is not None and not isinstance(body, str):
            return ToolResult(
                tool_name="http_request",
                content="Request body must be a string.",
                success=False,
            )
        if (
            isinstance(body, str)
            and len(body.encode("utf-8")) > _MAX_REQUEST_BODY_BYTES
        ):
            return ToolResult(
                tool_name="http_request",
                content="Request body exceeds the 256 KiB limit.",
                success=False,
            )
        if method in {"GET", "HEAD"} and body not in (None, ""):
            return ToolResult(
                tool_name="http_request",
                content=f"{method} requests cannot include a body.",
                success=False,
            )
        try:
            timeout = min(max(float(params.get("timeout", 30)), 0.1), 60.0)
        except (TypeError, ValueError):
            return ToolResult(
                tool_name="http_request",
                content="Timeout must be a number between 0.1 and 60 seconds.",
                success=False,
            )

        try:
            t0 = time.time()
            # Follow redirects manually so each hop is re-checked for SSRF — an
            # allowed public URL must not be able to 30x-redirect us to an
            # internal/metadata address.
            response = self._request_following_redirects(
                method,
                url,
                headers=headers,
                content=body,
                timeout=timeout,
            )
            elapsed_ms = (time.time() - t0) * 1000

            content_type = response.headers.get("content-type", "")
            encoding = self._encoding_from_content_type(content_type)
            content = response.body.decode(encoding, errors="replace")
            if response.truncated:
                content += "\n\n[Response truncated at 1 MB]"

            taint_metadata = external_taint(
                "http:"
                f"{urllib.parse.urlsplit(response.logical_url).hostname or 'unknown'}"
            )
            return ToolResult(
                tool_name="http_request",
                content=content,
                success=True,
                metadata={
                    "status_code": response.status_code,
                    "headers": self._safe_response_headers(response.headers),
                    "content_type": content_type,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "truncated": response.truncated,
                    **taint_metadata,
                },
            )
        except httpx.TimeoutException:
            return ToolResult(
                tool_name="http_request",
                content=f"Request timed out after {timeout:g}s.",
                success=False,
            )
        except _SSRFRedirectError as exc:
            return ToolResult(
                tool_name="http_request",
                content=f"SSRF protection blocked redirect: {exc}",
                success=False,
            )
        except httpx.RequestError:
            return ToolResult(
                tool_name="http_request",
                content="Request failed.",
                success=False,
            )
        except Exception:
            logger.exception("Unexpected HTTP request failure")
            return ToolResult(
                tool_name="http_request",
                content="Unexpected request failure.",
                success=False,
            )

    @staticmethod
    def _validate_headers(headers: Any) -> dict[str, str]:
        if not isinstance(headers, Mapping):
            raise TypeError("headers must be an object")
        if len(headers) > _MAX_HEADER_COUNT:
            raise ValueError("too many headers")
        validated: dict[str, str] = {}
        normalized_names: set[str] = set()
        aggregate_bytes = 0
        blocked = {
            "connection",
            "content-length",
            "host",
            "proxy-authorization",
            "proxy-connection",
            "transfer-encoding",
            "upgrade",
        }
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("header names and values must be strings")
            name = key.strip()
            normalized_name = name.lower()
            if (
                not name
                or len(name) > _MAX_HEADER_NAME_LENGTH
                or len(value) > _MAX_HEADER_VALUE_LENGTH
                or normalized_name in blocked
                or normalized_name in normalized_names
                or any(char in name or char in value for char in "\r\n")
            ):
                raise ValueError("header is forbidden or malformed")
            aggregate_bytes += len(name.encode("utf-8")) + len(value.encode("utf-8"))
            if aggregate_bytes > _MAX_HEADER_BYTES:
                raise ValueError("headers exceed the 32 KiB limit")
            # Values are transmitted literally. Environment interpolation here
            # would turn prompt-controlled ``$TOKEN`` text into secret exfiltration.
            validated[name] = value
            normalized_names.add(normalized_name)
        return validated

    @staticmethod
    def _encoding_from_content_type(content_type: str) -> str:
        for part in content_type.split(";")[1:]:
            key, _, value = part.strip().partition("=")
            if key.lower() == "charset" and value.strip():
                return value.strip().strip("\"'")
        return "utf-8"

    @staticmethod
    def _safe_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
        allowed = {
            "cache-control",
            "content-language",
            "content-length",
            "content-type",
            "etag",
            "last-modified",
        }
        return {
            key.lower(): value
            for key, value in headers.items()
            if key.lower() in allowed
        }

    @staticmethod
    def _origin(url: str) -> tuple[str, str, int]:
        """Return a normalized RFC 6454-style origin for redirect checks."""
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port or (443 if scheme == "https" else 80)
        return scheme, hostname, port

    @staticmethod
    def _strip_cross_origin_credentials(
        headers: Mapping[str, str],
    ) -> dict[str, str]:
        """Keep only explicitly safe headers for a different origin.

        Credential header names are not standardized (for example
        ``Private-Token`` and ``X-Skynet-Token``), so a denylist is not a
        reliable security boundary.
        """
        return {
            key: value
            for key, value in headers.items()
            if key.lower() in _CROSS_ORIGIN_HEADER_ALLOWLIST
        }

    def _pin_destination(self, url: str) -> tuple[str, str, str]:
        """Resolve, validate, and pin one logical URL to a concrete public IP."""
        error = check_ssrf(url)
        if error:
            raise _SSRFRedirectError(error)
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            records = self._resolver(
                hostname,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise _SSRFRedirectError("DNS resolution failed.") from exc
        addresses = sorted({record[4][0] for record in records if record[4]})
        if not addresses:
            raise _SSRFRedirectError("DNS resolution returned no addresses.")
        blocked = [address for address in addresses if is_private_ip(address)]
        if blocked:
            raise _SSRFRedirectError("Destination resolves to a non-public address.")
        selected = addresses[0]
        rendered_ip = f"[{selected}]" if ":" in selected else selected
        default_port = 443 if parsed.scheme == "https" else 80
        pinned_netloc = rendered_ip if port == default_port else f"{rendered_ip}:{port}"
        pinned_url = urllib.parse.urlunsplit(
            (
                parsed.scheme,
                pinned_netloc,
                parsed.path or "/",
                parsed.query,
                "",
            )
        )
        host_header = hostname if port == default_port else f"{hostname}:{port}"
        return pinned_url, host_header, hostname

    def _request_following_redirects(
        self,
        method: str,
        url: str,
        *,
        headers: dict,
        content: Any,
        timeout: float,
    ) -> _StreamedResponse:
        """Issue the request, re-checking SSRF on every redirect hop.

        httpx's built-in ``follow_redirects`` would chase a 30x ``Location``
        without re-validating it, letting a public URL bounce us to an internal
        host. We follow manually and run :func:`check_ssrf` on each target.
        """
        current_url = url
        current_method = method
        current_headers = dict(headers)
        body = content
        with httpx.Client(
            transport=self._transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                pinned_url, host_header, sni_hostname = self._pin_destination(
                    current_url
                )
                hop_headers = dict(current_headers)
                hop_headers["Host"] = host_header
                extensions = (
                    {"sni_hostname": sni_hostname}
                    if current_url.startswith("https://")
                    else None
                )
                with client.stream(
                    current_method,
                    pinned_url,
                    headers=hop_headers,
                    content=body,
                    timeout=timeout,
                    extensions=extensions,
                ) as response:
                    chunks: list[bytes] = []
                    total = 0
                    truncated = False
                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        remaining = _MAX_RESPONSE_BYTES - total
                        if remaining <= 0:
                            truncated = True
                            break
                        if len(chunk) > remaining:
                            chunks.append(chunk[:remaining])
                            total += remaining
                            truncated = True
                            break
                        chunks.append(chunk)
                        total += len(chunk)
                    streamed = _StreamedResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        body=b"".join(chunks),
                        truncated=truncated,
                        logical_url=current_url,
                    )
                if streamed.status_code not in (301, 302, 303, 307, 308):
                    return streamed
                location = streamed.headers.get("location", "")
                if not location:
                    return streamed
                # Resolve relative redirects against the logical URL, never
                # against the pinned IP URL exposed to the transport.
                next_url = urllib.parse.urljoin(current_url, location)
                if self._origin(current_url) != self._origin(next_url):
                    if streamed.status_code in (307, 308) and current_method not in {
                        "GET",
                        "HEAD",
                    }:
                        raise _SSRFRedirectError(
                            "Cross-origin redirect for a mutating request denied."
                        )
                    current_headers = self._strip_cross_origin_credentials(
                        current_headers
                    )
                current_url = next_url
                # Per RFC 7231, 301/302/303 turn the method into GET and drop
                # the body (except for HEAD).
                if streamed.status_code in (301, 302, 303) and current_method != "HEAD":
                    current_method = "GET"
                    body = None
        raise _SSRFRedirectError(f"Exceeded maximum of {_MAX_REDIRECTS} redirects.")


__all__ = ["HttpRequestTool"]
