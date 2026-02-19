"""Utilities for capturing output from remotely executed functions."""

import io
import logging
import os
import queue


_TEE_FUNCTION_OUTPUT = os.environ.get("SKYTORCH_TEE_FUNCTION_OUTPUT", "false").lower() in (
    "1",
    "true",
    "yes",
)


class OutputCapture(io.TextIOBase):
    """Captures writes to stdout/stderr and puts them onto a queue.

    When ``tee=True``, also writes to the original stream so server-side logs
    are preserved.  Controlled by the ``SKYTORCH_TEE_FUNCTION_OUTPUT`` env var
    (default: false, since output is streamed to the client).
    """

    def __init__(
        self,
        stream_name: str,
        original: io.TextIOBase,
        queue: queue.SimpleQueue,
        tee: bool = _TEE_FUNCTION_OUTPUT,
    ):
        self._stream_name = stream_name
        self._original = original
        self._queue = queue
        self._tee = tee

    def write(self, s: str) -> int:
        if s:
            self._queue.put((self._stream_name, s))
            if self._tee:
                self._original.write(s)
        return len(s)

    def flush(self):
        self._original.flush()

    def fileno(self):
        return self._original.fileno()

    @property
    def encoding(self):
        return self._original.encoding

    def isatty(self):
        return False


class LogCapture(logging.Handler):
    """Logging handler that captures log records onto a queue.

    This intercepts output from libraries (e.g. httpx, transformers) that log
    via Python's logging module rather than printing to stdout/stderr.

    When ``tee=False``, the existing handlers on the root logger are temporarily
    disabled so that log records are only forwarded to the client.
    """

    def __init__(self, log_queue: queue.SimpleQueue, tee: bool = _TEE_FUNCTION_OUTPUT):
        super().__init__()
        self._queue = log_queue
        self._tee = tee
        self._muted_handlers: list[logging.Handler] = []

    def install(self) -> None:
        """Attach to the root logger, optionally muting existing handlers."""
        root = logging.getLogger()
        if not self._tee:
            for h in root.handlers:
                self._muted_handlers.append(h)
                h.addFilter(_reject_all)
        root.addHandler(self)

    def uninstall(self) -> None:
        """Remove from the root logger and restore muted handlers."""
        root = logging.getLogger()
        root.removeHandler(self)
        for h in self._muted_handlers:
            h.removeFilter(_reject_all)
        self._muted_handlers.clear()

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record) + "\n"
            self._queue.put(("stderr", msg))
        except Exception:
            self.handleError(record)


def _reject_all(record: logging.LogRecord) -> bool:
    """Filter that rejects all log records."""
    return False
