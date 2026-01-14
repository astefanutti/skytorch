"""
KPU Async-to-Sync Bridge - Run async operations from synchronous code.

This module provides utilities to run async gRPC operations from
synchronous PyTorch dispatch callbacks. PyTorch's dispatch mechanism
calls our Python code synchronously, but we need to make async gRPC calls.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

# Store the main event loop for cross-thread async calls
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread_id: Optional[int] = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Set the event loop for cross-thread async calls.

    Call this from the main async context (e.g., when entering Compute)
    so that worker threads can schedule coroutines on this loop.

    Args:
        loop: The event loop to use for async operations
    """
    global _event_loop, _loop_thread_id
    _event_loop = loop
    _loop_thread_id = threading.current_thread().ident


def clear_event_loop() -> None:
    """Clear the stored event loop reference."""
    global _event_loop, _loop_thread_id
    _event_loop = None
    _loop_thread_id = None


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from synchronous code.

    This handles the common case where PyTorch's dispatch mechanism
    calls our Python code synchronously, but we need to make async
    gRPC calls to transfer tensor data.

    Strategy:
    1. If there's a stored event loop running in a different thread,
       use run_coroutine_threadsafe to schedule on that loop.
    2. If we're in the same thread as a running loop, run the coroutine
       in a new event loop in a worker thread.
    3. If a loop exists but isn't running, use run_until_complete()
    4. If no loop exists, create one with asyncio.run()

    Args:
        coro: Async coroutine to execute

    Returns:
        Result of the coroutine
    """
    current_thread_id = threading.current_thread().ident

    # Check if there's a running loop in the current thread
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    # If we have a stored event loop running in a different thread, use it
    if (
        _event_loop is not None
        and _loop_thread_id is not None
        and _loop_thread_id != current_thread_id
    ):
        try:
            if _event_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, _event_loop)
                return future.result()
        except RuntimeError:
            pass  # Loop was closed, fall through to other methods

    # We're in the same thread as the event loop, or no stored loop available
    # Run in a new event loop in a worker thread
    if current_loop is not None and current_loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    # Loop exists but isn't running - use it directly
    if current_loop is not None:
        return current_loop.run_until_complete(coro)

    # No loop at all - create one
    return asyncio.run(coro)
