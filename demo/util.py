"""
SkyTorch utility functions.
"""

import asyncio
import sys


async def async_input(prompt: str = "") -> str:
    """Non-blocking input that keeps the event loop free for async tasks."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    print(prompt, end="", flush=True)

    def _on_ready():
        loop.remove_reader(sys.stdin)
        future.set_result(sys.stdin.readline().rstrip("\n"))

    loop.add_reader(sys.stdin.fileno(), _on_ready)
    return await future
