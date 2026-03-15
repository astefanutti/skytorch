"""
Async streamer wrapper for HuggingFace Transformers.

Offloads the streamer's put() to a background thread so the generate
loop isn't blocked by DeferredScalarTensor resolution (tolist() calls).
"""

import queue
import threading

import torch
from transformers.generation.streamers import BaseStreamer


class AsyncStreamer(BaseStreamer):
    """Wraps a streamer to offload put() to a background thread.

    This prevents the generate thread from blocking on DeferredScalarTensor
    resolution (tolist() inside the inner streamer's put()). The generate
    loop sees put() return immediately, allowing it to dispatch the next
    forward pass while the streamer processes tokens in parallel.

    Args:
        inner: The streamer to wrap (e.g., TextStreamer, TextIteratorStreamer).
            Must implement put(), end(), and optionally reset().
    """

    _SENTINEL = object()

    def __init__(self, inner: BaseStreamer):
        self._inner = inner
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                self._inner.put(item)
        except Exception as e:
            self._error = e

    def put(self, value: torch.Tensor) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(value)

    def reset(self) -> None:
        if self._thread.is_alive():
            self._queue.put(self._SENTINEL)
            self._thread.join()
        self._queue = queue.SimpleQueue()
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._inner.reset()

    def end(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join()
        if self._error is not None:
            raise self._error
        self._inner.end()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
