"""GuardrailsEngine — security-aware inference engine wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message
from openjarvis.engine._stubs import InferenceEngine, StreamChunk
from openjarvis.security._stubs import BaseScanner
from openjarvis.security.scanner import PIIScanner, SecretScanner
from openjarvis.security.types import RedactionMode, ScanResult


class SecurityBlockError(Exception):
    """Raised when mode is BLOCK and security findings are detected."""


class GuardrailsEngine(InferenceEngine):
    """Wraps an existing ``InferenceEngine`` with security scanning.

    Not registered in ``EngineRegistry`` — instantiated dynamically to wrap
    any engine at runtime.

    Parameters
    ----------
    engine:
        The wrapped inference engine.
    scanners:
        List of scanners to run.  Defaults to ``SecretScanner`` + ``PIIScanner``.
    mode:
        Action taken on findings: WARN, REDACT, or BLOCK.
    scan_input:
        Whether to scan input messages.
    scan_output:
        Whether to scan output content.
    bus:
        Optional event bus for publishing security events.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        scanners: Optional[List[BaseScanner]] = None,
        mode: RedactionMode = RedactionMode.WARN,
        scan_input: bool = True,
        scan_output: bool = True,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._engine = engine
        self._scanners: List[BaseScanner] = (
            scanners
            if scanners is not None
            else [
                SecretScanner(),
                PIIScanner(),
            ]
        )
        self._mode = mode
        self._scan_input = scan_input
        self._scan_output = scan_output
        self._bus = bus

    # -- properties ----------------------------------------------------------

    @property
    def engine_id(self) -> str:  # type: ignore[override]
        """Delegate to the wrapped engine."""
        return self._engine.engine_id

    # -- scanning helpers ----------------------------------------------------

    def _scan_text(self, text: str) -> ScanResult:
        """Run all scanners on *text* and merge findings."""
        merged = ScanResult()
        for scanner in self._scanners:
            result = scanner.scan(text)
            merged.findings.extend(result.findings)
        return merged

    def _redact_text(self, text: str) -> str:
        """Run all scanners' redact() on *text*."""
        result = text
        for scanner in self._scanners:
            result = scanner.redact(result)
        return result

    def _handle_findings(
        self,
        text: str,
        result: ScanResult,
        direction: str,
    ) -> str:
        """Apply the configured mode to findings.

        Parameters
        ----------
        text:
            The original text.
        result:
            Scan result containing findings.
        direction:
            ``"input"`` or ``"output"`` — used in event data.

        Returns
        -------
        str
            Possibly modified text (unchanged for WARN, redacted for REDACT).

        Raises
        ------
        SecurityBlockError
            If mode is BLOCK.
        """
        finding_dicts = [
            {
                "pattern": f.pattern_name,
                "threat": f.threat_level.value,
                "description": f.description,
            }
            for f in result.findings
        ]

        if self._mode == RedactionMode.WARN:
            if self._bus:
                self._bus.publish(
                    EventType.SECURITY_ALERT,
                    {
                        "direction": direction,
                        "findings": finding_dicts,
                        "mode": "warn",
                    },
                )
            return text

        if self._mode == RedactionMode.REDACT:
            if self._bus:
                self._bus.publish(
                    EventType.SECURITY_ALERT,
                    {
                        "direction": direction,
                        "findings": finding_dicts,
                        "mode": "redact",
                    },
                )
            return self._redact_text(text)

        # BLOCK mode
        if self._bus:
            self._bus.publish(
                EventType.SECURITY_BLOCK,
                {
                    "direction": direction,
                    "findings": finding_dicts,
                    "mode": "block",
                },
            )
        raise SecurityBlockError(
            f"Security scan blocked {direction}: "
            f"{len(result.findings)} finding(s) detected"
        )

    def _process_messages(
        self,
        messages: Sequence[Message],
    ) -> Sequence[Message]:
        """Scan/redact an input batch before the wrapped engine sees it."""
        if not self._scan_input:
            return messages
        processed = list(messages)
        for index, message in enumerate(processed):
            if not message.content:
                continue
            result = self._scan_text(message.content)
            if result.clean:
                continue
            processed[index] = Message(
                role=message.role,
                content=self._handle_findings(
                    message.content,
                    result,
                    "input",
                ),
                name=message.name,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                metadata=message.metadata,
                images=message.images,
            )
        return processed

    def _process_output(self, content: str) -> str:
        """Scan one complete output before it crosses the caller boundary."""
        if not self._scan_output or not content:
            return content
        result = self._scan_text(content)
        if result.clean:
            return content
        return self._handle_findings(content, result, "output")

    # -- InferenceEngine interface -------------------------------------------

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Scan input, call wrapped engine, scan output."""
        messages = self._process_messages(messages)

        # Call wrapped engine
        response = self._engine.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Scan output
        content = response.get("content", "")
        if content:
            response["content"] = self._process_output(content)

        return response

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream safely.

        BLOCK and REDACT modes buffer the complete response before yielding a
        single byte.  Scanning only after yielding made a later block purely
        cosmetic because the secret/PII had already crossed the boundary.
        WARN mode keeps live streaming semantics and emits its alert after the
        stream; it is explicitly observational.
        """
        messages = self._process_messages(messages)
        must_prebuffer = self._scan_output and self._mode in (
            RedactionMode.BLOCK,
            RedactionMode.REDACT,
        )
        accumulated: list[str] = []
        async for token in self._engine.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            accumulated.append(token)
            if not must_prebuffer:
                yield token

        full_output = "".join(accumulated)
        if must_prebuffer:
            processed = self._process_output(full_output)
            if processed == full_output:
                for token in accumulated:
                    yield token
            elif processed:
                yield processed
        elif self._scan_output and full_output:
            # WARN mode: publish the finding without attempting an impossible
            # retroactive block/redaction.
            self._process_output(full_output)

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator["StreamChunk"]:
        """Stream rich chunks with the same pre-buffer guarantee as stream()."""
        messages = self._process_messages(messages)
        must_prebuffer = self._scan_output and self._mode in (
            RedactionMode.BLOCK,
            RedactionMode.REDACT,
        )
        chunks: list[StreamChunk] = []
        accumulated: list[str] = []
        async for chunk in self._engine.stream_full(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            chunks.append(chunk)
            if chunk.content:
                accumulated.append(chunk.content)
            if not must_prebuffer:
                yield chunk

        full_output = "".join(accumulated)
        if must_prebuffer:
            processed = self._process_output(full_output)
            if processed == full_output:
                for chunk in chunks:
                    yield chunk
                return

            content_emitted = False
            for chunk in chunks:
                if chunk.content and not content_emitted:
                    yield replace(chunk, content=processed)
                    content_emitted = True
                elif chunk.content:
                    yield replace(chunk, content=None)
                else:
                    yield chunk
        elif self._scan_output and full_output:
            self._process_output(full_output)

    def list_models(self) -> List[str]:
        """Delegate to wrapped engine."""
        return self._engine.list_models()

    def health(self) -> bool:
        """Delegate to wrapped engine."""
        return self._engine.health()


__all__ = ["GuardrailsEngine", "SecurityBlockError"]
