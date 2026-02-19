"""
SkyTorch Async-to-Sync Bridge - Background loop with debugger fallback.

Uses a dedicated background event loop for all async operations.
This ensures proper ordering when operations come from multiple threads
(main thread and autograd threads).

When waiting for results, uses a short timeout with retry to handle
debugger scenarios where the background loop may be paused.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

# Global event loop running in a background thread
_global_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()
_loop_ready = threading.Event()


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    """Run the event loop forever in a background thread."""
    asyncio.set_event_loop(loop)
    _loop_ready.set()
    loop.run_forever()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the global event loop, starting it if necessary.

    Returns:
        The global event loop running in the background thread.
    """
    global _global_loop, _loop_thread

    if _global_loop is not None and _global_loop.is_running():
        return _global_loop

    with _loop_lock:
        # Double-check after acquiring lock
        if _global_loop is not None and _global_loop.is_running():
            return _global_loop

        # Create a new event loop
        _global_loop = asyncio.new_event_loop()
        _global_loop.set_task_factory(asyncio.eager_task_factory)

        # Start the loop in a background daemon thread
        _loop_thread = threading.Thread(
            target=_run_loop_forever,
            args=(_global_loop,),
            daemon=True,
            name="sky-async-loop"
        )
        _loop_thread.start()

        # Wait for the loop to be ready
        _loop_ready.wait()

    return _global_loop


def stop_global_loop() -> None:
    """Stop the global event loop. Called at exit."""
    global _global_loop, _loop_thread

    if _global_loop is not None and _global_loop.is_running():
        _global_loop.call_soon_threadsafe(_global_loop.stop)

    if _loop_thread is not None:
        _loop_thread.join(timeout=1.0)
        _loop_thread = None

    _global_loop = None


# Register cleanup on exit
atexit.register(stop_global_loop)


def run_async(coro: Coroutine[Any, Any, T]) -> Future[T]:
    """
    Submit an async coroutine to the global event loop.

    All coroutines are submitted to a single global event loop running
    in a background thread, ensuring proper ordering across all threads.

    Args:
        coro: Async coroutine to execute

    Returns:
        A concurrent.futures.Future that resolves to the coroutine result.
    """
    loop = get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)
