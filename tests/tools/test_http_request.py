"""Tests for the HTTP request tool with SSRF protection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import openjarvis.tools.http_request as http_request_module
from openjarvis.tools.http_request import HttpRequestTool, _SSRFRedirectError

_REAL_PIN_DESTINATION = HttpRequestTool._pin_destination


@pytest.fixture(autouse=True)
def _keep_respx_on_logical_hosts():
    """Keep legacy respx fixtures deterministic while preserving hop checks."""

    def _passthrough_pin(self, url):
        error = http_request_module.check_ssrf(url)
        if error:
            raise _SSRFRedirectError(error)
        parsed = httpx.URL(url)
        host = parsed.host
        port = parsed.port
        default_port = 443 if parsed.scheme == "https" else 80
        host_header = host if port in (None, default_port) else f"{host}:{port}"
        return url, host_header, host

    with patch.object(HttpRequestTool, "_pin_destination", _passthrough_pin):
        yield


class TestHttpRequestTool:
    def test_spec_name_and_category(self):
        tool = HttpRequestTool()
        assert tool.spec.name == "http_request"
        assert tool.spec.category == "network"

    def test_spec_required_capabilities(self):
        tool = HttpRequestTool()
        assert "network:fetch" in tool.spec.required_capabilities

    def test_mutations_require_extra_capability_and_confirmation(self):
        tool = HttpRequestTool()
        assert tool.authorization_capabilities({"method": "GET"}) == [
            "network:fetch"
        ]
        assert "network:mutate" in tool.authorization_capabilities(
            {"method": "POST"}
        )
        assert tool.requires_confirmation_for({"method": "GET"}) is False
        assert tool.requires_confirmation_for({"method": "POST"}) is True

    def test_spec_parameters_require_url(self):
        tool = HttpRequestTool()
        assert "url" in tool.spec.parameters["properties"]
        assert "url" in tool.spec.parameters["required"]

    def test_tool_id(self):
        tool = HttpRequestTool()
        assert tool.tool_id == "http_request"

    def test_to_openai_function(self):
        tool = HttpRequestTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "http_request"
        assert "url" in fn["function"]["parameters"]["properties"]

    def test_no_url(self):
        tool = HttpRequestTool()
        result = tool.execute()
        assert result.success is False
        assert "No URL" in result.content

    def test_empty_url(self):
        tool = HttpRequestTool()
        result = tool.execute(url="")
        assert result.success is False
        assert "No URL" in result.content

    def test_ssrf_blocked_private_ip(self):
        """Request to private IP should be blocked by SSRF protection."""
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf") as mock_ssrf:
            mock_ssrf.return_value = "URL resolves to private IP: 192.168.1.1"
            result = tool.execute(url="http://192.168.1.1/admin")
        assert result.success is False
        assert "SSRF protection" in result.content
        assert "private IP" in result.content

    def test_ssrf_blocked_metadata_endpoint(self):
        """Request to cloud metadata endpoint should be blocked."""
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf") as mock_ssrf:
            mock_ssrf.return_value = (
                "Blocked host: 169.254.169.254 (cloud metadata endpoint)"
            )
            result = tool.execute(url="http://169.254.169.254/latest/meta-data/")
        assert result.success is False
        assert "SSRF protection" in result.content
        assert "metadata" in result.content.lower() or "Blocked host" in result.content

    @respx.mock
    def test_successful_get(self):
        """Successful GET request returns response content and metadata."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(
                200,
                text='{"key": "value"}',
                headers={"content-type": "application/json"},
            )
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(url="https://api.example.com/data")
        assert result.success is True
        assert '"key": "value"' in result.content
        assert result.metadata["status_code"] == 200
        assert "application/json" in result.metadata["content_type"]
        assert "elapsed_ms" in result.metadata

    @respx.mock
    def test_post_with_body(self):
        """POST request with body sends content correctly."""
        respx.post("https://api.example.com/submit").mock(
            return_value=httpx.Response(
                201,
                text='{"id": 42}',
                headers={"content-type": "application/json"},
            )
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://api.example.com/submit",
                method="POST",
                body='{"name": "test"}',
                headers={"Content-Type": "application/json"},
            )
        assert result.success is True
        assert '"id": 42' in result.content
        assert result.metadata["status_code"] == 201

    @respx.mock
    def test_put_method(self):
        """PUT request works correctly."""
        respx.put("https://api.example.com/resource/1").mock(
            return_value=httpx.Response(200, text="updated")
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://api.example.com/resource/1",
                method="PUT",
                body="new data",
            )
        assert result.success is True
        assert "updated" in result.content

    @respx.mock
    def test_delete_method(self):
        """DELETE request works correctly."""
        respx.delete("https://api.example.com/resource/1").mock(
            return_value=httpx.Response(204, text="")
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://api.example.com/resource/1",
                method="DELETE",
            )
        assert result.success is True

    @respx.mock
    def test_head_method(self):
        """HEAD request works correctly."""
        respx.head("https://api.example.com/check").mock(
            return_value=httpx.Response(
                200,
                text="",
                headers={"x-custom": "header-value"},
            )
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://api.example.com/check",
                method="HEAD",
            )
        assert result.success is True
        assert result.metadata["status_code"] == 200

    def test_timeout_handling(self):
        """Timeout should produce a clear error."""
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            with patch.object(
                HttpRequestTool,
                "_request_following_redirects",
                side_effect=httpx.TimeoutException("timed out"),
            ):
                result = tool.execute(url="https://slow.example.com", timeout=5)
        assert result.success is False
        assert "timed out" in result.content.lower()

    def test_request_error(self):
        """Connection error should produce a clear error."""
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            with patch.object(
                HttpRequestTool,
                "_request_following_redirects",
                side_effect=httpx.ConnectError("Connection refused"),
            ):
                result = tool.execute(url="https://down.example.com")
        assert result.success is False
        assert result.content == "Request failed."

    @respx.mock
    def test_redirect_to_private_ip_blocked(self):
        """A redirect to an internal/metadata host must be re-checked + blocked."""
        respx.get("https://public.example.com/start").mock(
            return_value=httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/"}
            )
        )
        tool = HttpRequestTool()
        # First check (initial URL) passes; the redirect target is blocked.
        with patch(
            "openjarvis.tools.http_request.check_ssrf",
            side_effect=[None, "Blocked host: 169.254.169.254"],
        ):
            result = tool.execute(url="https://public.example.com/start")
        assert result.success is False
        assert "SSRF protection blocked redirect" in result.content

    @respx.mock
    def test_safe_redirect_is_followed(self):
        """A redirect to another public URL is followed normally."""
        respx.get("https://public.example.com/start").mock(
            return_value=httpx.Response(
                302, headers={"location": "https://public.example.com/final"}
            )
        )
        respx.get("https://public.example.com/final").mock(
            return_value=httpx.Response(200, text="done")
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(url="https://public.example.com/start")
        assert result.success is True
        assert "done" in result.content

    def test_cross_origin_redirect_strips_all_credentials(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "public.example.com":
                return httpx.Response(
                    302,
                    headers={"location": "https://other.example.net/final"},
                )
            return httpx.Response(200, text="done")

        tool = HttpRequestTool(transport=httpx.MockTransport(handler))
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://public.example.com/start",
                headers={
                    "Authorization": "Bearer first-origin-only",
                    "Cookie": "session=first-origin-only",
                    "X-Api-Key": "first-origin-only",
                    "Vendor-Access-Token": "first-origin-only",
                    "X-Skynet-Token": "first-origin-only",
                    "Private-Token": "first-origin-only",
                    "X-Harmless": "not-allowlisted",
                    "Accept-Language": "pt-BR",
                },
            )

        assert result.success is True
        assert len(requests) == 2
        assert requests[0].headers["authorization"] == (
            "Bearer first-origin-only"
        )
        for name in (
            "authorization",
            "cookie",
            "x-api-key",
            "vendor-access-token",
            "x-skynet-token",
            "private-token",
            "x-harmless",
        ):
            assert name not in requests[1].headers
        assert requests[1].headers["accept-language"] == "pt-BR"

    def test_same_origin_redirect_preserves_credentials(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/final"})
            return httpx.Response(200, text="done")

        tool = HttpRequestTool(transport=httpx.MockTransport(handler))
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://public.example.com/start",
                headers={
                    "Authorization": "Bearer same-origin",
                    "Cookie": "session=same-origin",
                },
            )

        assert result.success is True
        assert len(requests) == 2
        assert requests[1].headers["authorization"] == "Bearer same-origin"
        assert requests[1].headers["cookie"] == "session=same-origin"

    def test_scheme_or_port_change_is_cross_origin(self):
        tool = HttpRequestTool()
        assert tool._origin("https://example.com/a") == (
            "https",
            "example.com",
            443,
        )
        assert tool._origin("https://EXAMPLE.com.:443/b") == (
            "https",
            "example.com",
            443,
        )
        assert tool._origin("http://example.com/a") != tool._origin(
            "https://example.com/a"
        )
        assert tool._origin("https://example.com:8443/a") != tool._origin(
            "https://example.com/a"
        )

    def test_cross_origin_307_never_replays_mutating_body(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                307,
                headers={"location": "https://other.example.net/final"},
            )

        tool = HttpRequestTool(transport=httpx.MockTransport(handler))
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://public.example.com/start",
                method="POST",
                body='{"secret":"must-not-replay"}',
            )

        assert result.success is False
        assert "mutating request denied" in result.content
        assert len(requests) == 1

    def test_method_validation(self):
        """Invalid HTTP method should be rejected."""
        tool = HttpRequestTool()
        result = tool.execute(url="https://example.com", method="TRACE")
        assert result.success is False
        assert "Unsupported HTTP method" in result.content
        assert "TRACE" in result.content

    def test_request_limits_reject_oversized_body_and_headers(self):
        tool = HttpRequestTool()
        oversized_body = tool.execute(
            url="https://example.com",
            method="POST",
            body="x" * 262_145,
        )
        assert oversized_body.success is False
        assert "256 KiB" in oversized_body.content

        too_many_headers = tool.execute(
            url="https://example.com",
            headers={f"X-Test-{index}": "ok" for index in range(33)},
        )
        assert too_many_headers.success is False
        assert "too many headers" in too_many_headers.content

        oversized_header = tool.execute(
            url="https://example.com",
            headers={"X-Test": "x" * 8_193},
        )
        assert oversized_header.success is False
        assert "forbidden or malformed" in oversized_header.content

    def test_method_case_insensitive(self):
        """Method should be case-insensitive."""
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            with respx.mock:
                respx.get("https://api.example.com/data").mock(
                    return_value=httpx.Response(200, text="ok")
                )
                result = tool.execute(url="https://api.example.com/data", method="get")
        assert result.success is True

    @respx.mock
    def test_response_truncation(self):
        """Response larger than 1 MB should be truncated."""
        large_body = "x" * 2_000_000  # 2 MB
        respx.get("https://api.example.com/large").mock(
            return_value=httpx.Response(200, text=large_body)
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(url="https://api.example.com/large")
        assert result.success is True
        assert "[Response truncated at 1 MB]" in result.content
        assert result.metadata["truncated"] is True

    @respx.mock
    def test_response_not_truncated_when_small(self):
        """Response smaller than 1 MB should not be truncated."""
        small_body = "hello world"
        respx.get("https://api.example.com/small").mock(
            return_value=httpx.Response(200, text=small_body)
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(url="https://api.example.com/small")
        assert result.success is True
        assert result.content == "hello world"
        assert result.metadata["truncated"] is False

    @respx.mock
    def test_metadata_includes_headers(self):
        """Response metadata should include headers dict."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(
                200,
                text="ok",
                headers={
                    "content-type": "text/plain",
                    "x-request-id": "abc123",
                },
            )
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(url="https://api.example.com/data")
        assert isinstance(result.metadata["headers"], dict)
        assert "x-request-id" not in result.metadata["headers"]

    @respx.mock
    def test_header_environment_variables_are_never_expanded(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("OPENJARVIS_TEST_SECRET", "must-not-leak")
        route = respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, text="ok")
        )
        tool = HttpRequestTool()
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            result = tool.execute(
                url="https://api.example.com/data",
                headers={"X-Test": "$OPENJARVIS_TEST_SECRET"},
            )
        assert result.success
        assert route.calls[0].request.headers["x-test"] == (
            "$OPENJARVIS_TEST_SECRET"
        )

    def test_dns_resolution_is_pinned_to_validated_public_ip(self):
        resolver = MagicMock(
            return_value=[
                (
                    2,
                    1,
                    6,
                    "",
                    ("93.184.216.34", 443),
                )
            ]
        )
        tool = HttpRequestTool(resolver=resolver)
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            pinned, host_header, sni = _REAL_PIN_DESTINATION(
                tool,
                "https://example.com/path",
            )
        assert pinned == "https://93.184.216.34/path"
        assert host_header == "example.com"
        assert sni == "example.com"

    def test_dns_rebinding_candidate_blocks_entire_hop(self):
        resolver = MagicMock(
            return_value=[
                (2, 1, 6, "", ("93.184.216.34", 443)),
                (2, 1, 6, "", ("127.0.0.1", 443)),
            ]
        )
        tool = HttpRequestTool(resolver=resolver)
        with patch("openjarvis.tools.http_request.check_ssrf", return_value=None):
            with pytest.raises(_SSRFRedirectError):
                _REAL_PIN_DESTINATION(tool, "https://example.com/")


__all__ = ["TestHttpRequestTool"]
