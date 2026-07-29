"""REPL namespace for the RLM agent.

Provides a persistent namespace with injected helper functions
(``llm_query``, ``llm_batch``, ``FINAL``, ``FINAL_VAR``) that the RLM
agent's generated code uses.  Code execution is disabled unless the caller
supplies a verified isolated sandbox executor.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from openjarvis.core.cancellation import AgentCancelledError

# Safe stdlib modules pre-injected into the REPL namespace
_SAFE_MODULES = [
    "json",
    "re",
    "math",
    "collections",
    "itertools",
    "functools",
    "textwrap",
    "string",
    "copy",
    "datetime",
]

# Patterns blocked for security
_BLOCKED_PATTERNS = [
    "os.system",
    "os.popen",
    "subprocess",
    "shutil.rmtree",
    "__import__",
    "open(",
    "ctypes",
    "socket",
    "http.client",
    "urllib",
]


class RLMRepl:
    """Persistent RLM namespace backed by an explicit isolated executor.

    Parameters
    ----------
    llm_query_fn:
        Callback invoked when REPL code calls ``llm_query(prompt)``.
    llm_batch_fn:
        Callback invoked when REPL code calls ``llm_batch(prompts)``.
    max_output_chars:
        Maximum characters captured from stdout per execution.
    sandbox_executor:
        Trusted isolated execution adapter.  Host ``exec`` is never used as a
        fallback.
    """

    def __init__(
        self,
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_batch_fn: Optional[Callable[[List[str]], List[str]]] = None,
        tool_call_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        tool_arg_names: Optional[Dict[str, Optional[str]]] = None,
        *,
        max_output_chars: int = 10000,
        sandbox_executor: Optional[Callable[[str, Dict[str, Any], int], str]] = None,
    ) -> None:
        self._max_output_chars = max_output_chars
        self._terminated = False
        self._final_value: Any = None
        self._tool_call_fn = tool_call_fn
        self._sandbox_executor = sandbox_executor

        # Build namespace
        self._namespace: Dict[str, Any] = {}

        # Inject safe stdlib modules
        for mod_name in _SAFE_MODULES:
            try:
                import importlib

                self._namespace[mod_name] = importlib.import_module(mod_name)
            except ImportError:
                pass

        # answer dict — code can set answer["ready"] = True, answer["value"] = ...
        self._namespace["answer"] = {"ready": False, "value": None}

        # Inject FINAL / FINAL_VAR
        self._namespace["FINAL"] = self._final
        self._namespace["FINAL_VAR"] = self._final_var

        # Inject llm_query / llm_batch
        if llm_query_fn is not None:
            self._namespace["llm_query"] = llm_query_fn
        if llm_batch_fn is not None:
            self._namespace["llm_batch"] = llm_batch_fn
        if tool_call_fn is not None:
            self._namespace["tool_call"] = self._tool_call

            # Expose each tool as a direct Python helper (e.g.
            # file_read(path="...")) so the model does not need to invent
            # pseudo-tool syntax or fall back to blocked file I/O.
            for tool_name, primary_arg in (tool_arg_names or {}).items():
                self._namespace[tool_name] = self._make_tool_wrapper(
                    tool_name,
                    primary_arg,
                )

    # ------------------------------------------------------------------
    # Termination helpers
    # ------------------------------------------------------------------

    def _final(self, value: Any) -> None:
        """Mark the REPL as terminated with a final answer."""
        self._terminated = True
        self._final_value = value

    def _final_var(self, var_name: str) -> None:
        """Mark the REPL as terminated, using a namespace variable as the answer."""
        value = self._namespace.get(var_name)
        self._terminated = True
        self._final_value = value

    @property
    def is_terminated(self) -> bool:
        """Check if FINAL/FINAL_VAR was called or answer["ready"] is True."""
        if self._terminated:
            return True
        answer = self._namespace.get("answer", {})
        if isinstance(answer, dict) and answer.get("ready"):
            return True
        return False

    @property
    def final_answer(self) -> Any:
        """Return the termination value."""
        if self._terminated:
            return self._final_value
        answer = self._namespace.get("answer", {})
        if isinstance(answer, dict) and answer.get("ready"):
            return answer.get("value")
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _tool_call(self, tool_name: str, *args: Any, **kwargs: Any) -> str:
        """Execute an injected OpenJarvis tool from within the REPL.

        Supported forms:
        - ``tool_call("file_read", {"path": "foo.txt"})``
        - ``tool_call("file_read", path="foo.txt")``
        """
        if self._tool_call_fn is None:
            raise RuntimeError("tool_call is not available in this REPL")

        if args:
            if len(args) != 1 or kwargs:
                raise TypeError(
                    "tool_call expects either a single dict argument or keyword args"
                )
            if not isinstance(args[0], dict):
                raise TypeError("tool_call positional argument must be a dict")
            params = dict(args[0])
        else:
            params = dict(kwargs)

        return self._tool_call_fn(tool_name, params)

    def _make_tool_wrapper(
        self,
        tool_name: str,
        primary_arg: Optional[str],
    ) -> Callable[..., str]:
        """Return a Python helper that dispatches to ``tool_call``."""

        def _wrapper(*args: Any, **kwargs: Any) -> str:
            if kwargs:
                params = dict(kwargs)
            elif len(args) == 1 and primary_arg is not None:
                params = {primary_arg: args[0]}
            elif len(args) == 1 and isinstance(args[0], dict):
                params = dict(args[0])
            elif not args:
                params = {}
            else:
                message = (
                    f"{tool_name} expects keyword args or a single "
                    f"{primary_arg!r} argument"
                )
                raise TypeError(message)
            return self._tool_call(tool_name, params)

        _wrapper.__name__ = tool_name
        return _wrapper

    def security_check(self, code: str) -> Optional[str]:
        """Check code for dangerous patterns. Returns error message or None."""
        for pattern in _BLOCKED_PATTERNS:
            if pattern in code:
                return f"Blocked: code contains prohibited pattern '{pattern}'"
        return None

    def execute(self, code: str) -> str:
        """Execute *code* only through the configured isolated adapter."""
        # Security check
        violation = self.security_check(code)
        if violation is not None:
            return f"Error: {violation}"

        if self._sandbox_executor is None:
            return (
                "Error: RLM REPL disabled because no verified isolated "
                "sandbox executor is configured."
            )
        try:
            output = self._sandbox_executor(
                code,
                self._namespace,
                self._max_output_chars,
            )
        except AgentCancelledError:
            raise
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        if not isinstance(output, str):
            return "RuntimeError: isolated sandbox returned a non-string result"

        # Truncate if needed
        if len(output) > self._max_output_chars:
            output = output[: self._max_output_chars] + "\n... (output truncated)"

        return output

    # ------------------------------------------------------------------
    # Namespace access
    # ------------------------------------------------------------------

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the REPL namespace."""
        self._namespace[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL namespace."""
        return self._namespace.get(name)


__all__ = ["RLMRepl"]
