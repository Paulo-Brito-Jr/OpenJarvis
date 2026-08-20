"""Security-disabled OpenCode headless-server adapter.

The legacy implementation spawned an external coding agent with authority
outside OpenJarvis's capability and confirmation boundary. Construction and
``run`` now avoid credentials and external processes until a verified isolated
sandbox bridge exists.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)

_LISTENING_RE = re.compile(r"listening on\s+(https?://\S+)", re.IGNORECASE)


def is_opencode_available() -> bool:
    """Return True if the ``opencode`` binary is on PATH."""
    return shutil.which("opencode") is not None


def _derive_openai_base_url(engine: Any) -> str:
    """Best-effort OpenAI-compatible base URL for an OpenJarvis engine.

    HTTP engines (Ollama, vLLM, llama.cpp, SGLang, LM Studio, …) expose a
    ``_host`` and serve an OpenAI-compatible API at ``<host>/v1``. Engines are
    often wrapped (e.g. ``InstrumentedEngine`` for telemetry stores the real
    engine at ``_inner``), so we unwrap a few layers to find the host. Returns
    "" when it cannot be derived (caller then needs an explicit base URL or a
    ``provider/model`` that opencode already knows).
    """
    seen: set[int] = set()
    cur = engine
    for _ in range(6):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        for attr in ("openai_base_url", "base_url"):
            val = getattr(cur, attr, "")
            if val:
                return str(val).rstrip("/")
        host = getattr(cur, "_host", "") or getattr(cur, "host", "")
        if host:
            host = str(host).rstrip("/")
            return host if host.endswith("/v1") else f"{host}/v1"
        # Unwrap common wrapper attributes (InstrumentedEngine -> _inner, etc.)
        cur = (
            getattr(cur, "_inner", None)
            or getattr(cur, "_engine", None)
            or getattr(cur, "_wrapped", None)
        )
    return ""


def _extract_text(parts: List[dict]) -> str:
    """Join the assistant's text parts from an opencode message response."""
    return "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and p.get("type") == "text"
    ).strip()


def _extract_tool_results(parts: List[dict]) -> List[ToolResult]:
    """Map opencode ``tool`` parts to OpenJarvis ToolResults (best-effort)."""
    results: List[ToolResult] = []
    for p in parts:
        if not isinstance(p, dict) or p.get("type") != "tool":
            continue
        state = p.get("state", {}) if isinstance(p.get("state"), dict) else {}
        status = state.get("status", "")
        output = state.get("output")
        if output is None:
            output = state.get("title", "") or (json.dumps(state) if state else "")
        results.append(
            ToolResult(
                tool_name=p.get("tool", p.get("name", "unknown")),
                content=str(output),
                # opencode tool state: completed | error | running | pending
                success=status == "completed",
            )
        )
    return results


@AgentRegistry.register("opencode")
class OpenCodeAgent(BaseAgent):
    """Agent that delegates coding tasks to a local ``opencode`` server.

    The ``engine`` is used to wire opencode at an OpenAI-compatible provider so
    inference runs on OpenJarvis's selected local model. ``agent`` selects
    opencode's built-in agent: ``build`` (full access) or ``plan`` (read-only).
    """

    agent_id = "opencode"
    accepts_tools = False
    requires_security_context = True
    _default_temperature = 0.7
    _default_max_tokens = 1024

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        workspace: str = "",
        agent: str = "build",
        provider_id: str = "openjarvis",
        provider_base_url: str = "",
        model_id: str = "",
        api_key: str = "",
        permission: Optional[Any] = None,
        hostname: str = "127.0.0.1",
        port: int = 0,
        server_password: str = "",
        timeout: int = 600,
        opencode_bin: str = "",
    ) -> None:
        super().__init__(
            engine,
            model,
            bus=bus,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._workspace = workspace or os.getcwd()
        self._agent = agent
        self._provider_id = provider_id
        self._provider_base_url = provider_base_url or _derive_openai_base_url(engine)
        self._model_id = model_id or model
        # The adapter is disabled, so retaining provider/server credentials in
        # this object has no legitimate purpose.
        del api_key, server_password
        self._api_key = ""
        self._permission = permission
        self._hostname = hostname
        self._port = port
        self._server_password = ""
        self._timeout = timeout
        self._opencode_bin = opencode_bin or shutil.which("opencode") or "opencode"
        self._proc: Optional[subprocess.Popen] = None
        self._base: str = ""
        self._config_dir: Optional[str] = None
        self._capability_policy = None
        self._security_agent_id = ""

    def bind_security(
        self,
        capability_policy: Optional[Any],
        agent_id: Optional[str] = None,
        boundary_guard: Optional[Any] = None,
    ) -> None:
        self._capability_policy = capability_policy
        self._security_agent_id = agent_id if isinstance(agent_id, str) else ""
        self._boundary_guard = boundary_guard

    def bind_boundary_guard(self, boundary_guard: Optional[Any]) -> None:
        self._boundary_guard = boundary_guard

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _default_permission(self) -> dict:
        """Deterministic, hang-proof permission policy for headless use.

        opencode's interactive default *asks* before some actions (and ``plan``
        asks before bash) — that would block forever with no TTY. We set
        explicit ``allow``/``deny`` so the server never waits on a prompt:

        - ``build``: allow edits + bash (a coding agent the user invoked).
        - ``plan``: deny edits + bash (read-only), matching opencode's intent.
        """
        return {"edit": "deny", "bash": "deny", "webfetch": "deny"}

    def _build_config(self) -> dict:
        """Build the opencode config (provider wiring + permission policy)."""
        cfg: dict = {
            "$schema": "https://opencode.ai/config.json",
            # The adapter is disabled until an enforceable sandbox bridge
            # exists.  Never serialize a caller-provided allow policy that
            # could be reused by invoking the binary out of band.
            "permission": self._default_permission(),
        }
        if self._provider_base_url:
            options: dict = {"baseURL": self._provider_base_url}
            if self._api_key:
                options["apiKey"] = self._api_key
            cfg["provider"] = {
                self._provider_id: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenJarvis Local",
                    "options": options,
                    "models": {self._model_id: {"name": self._model_id}},
                }
            }
        return cfg

    def _ensure_server(self) -> str:
        """Spawn ``opencode serve`` (once) and return its base URL."""
        raise RuntimeError(
            "OpenCode adapter disabled: no verified isolated sandbox and "
            "OpenJarvis capability bridge is available."
        )

        # Kept below for an eventual audited adapter implementation.  It is
        # unreachable while the fail-closed guard above is in force.
        if self._base and self._proc and self._proc.poll() is None:
            return self._base
        if not is_opencode_available() and not Path(self._opencode_bin).exists():
            raise RuntimeError(
                "OpenCodeAgent requires the 'opencode' binary. Install it with "
                "`npm i -g opencode-ai` or `brew install anomalyco/tap/opencode` "
                "(see https://opencode.ai)."
            )
        # Write our config to a private file referenced via OPENCODE_CONFIG, so
        # the engine provider + permission policy apply WITHOUT writing an
        # opencode.json into the user's workspace.
        import tempfile

        self._config_dir = tempfile.mkdtemp(prefix="openjarvis-opencode-")
        config_path = Path(self._config_dir) / "opencode.json"
        config_path.write_text(json.dumps(self._build_config(), indent=2), "utf-8")
        env = dict(os.environ)
        env["OPENCODE_CONFIG"] = str(config_path)
        if self._server_password:
            env["OPENCODE_SERVER_PASSWORD"] = self._server_password
        self._proc = subprocess.Popen(
            [
                self._opencode_bin,
                "serve",
                "--port",
                str(self._port),
                "--hostname",
                self._hostname,
            ],
            cwd=self._workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Parse the "listening on <url>" line from startup output.
        deadline = time.monotonic() + 60
        base = ""
        assert self._proc.stdout is not None
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError("opencode server exited during startup")
                continue
            m = _LISTENING_RE.search(line)
            if m:
                base = m.group(1).rstrip("/")
                break
        if not base:
            self.close()
            raise RuntimeError("opencode server did not report a listening URL")
        self._base = base
        return base

    def _client(self):
        import httpx

        headers = {}
        if self._server_password:
            import base64

            token = base64.b64encode(
                f"opencode:{self._server_password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {token}"
        return httpx.Client(base_url=self._base, headers=headers, timeout=self._timeout)

    def close(self) -> None:
        """Dispose the opencode session/server and terminate the process."""
        if self._base:
            try:
                with self._client() as c:
                    c.post("/global/dispose")
            except Exception:
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._base = ""
        if self._config_dir:
            shutil.rmtree(self._config_dir, ignore_errors=True)
            self._config_dir = None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run a coding task through opencode and return the assistant result."""
        self._emit_turn_start(input)
        self._emit_turn_end(turns=0, error=True)
        return AgentResult(
            content=(
                "OpenCode adapter disabled: no verified isolated sandbox and "
                "OpenJarvis capability bridge is available."
            ),
            turns=0,
            metadata={
                "error": True,
                "security_disabled": True,
                "reason": "unverified_external_sandbox",
            },
        )


__all__ = ["OpenCodeAgent", "is_opencode_available"]
