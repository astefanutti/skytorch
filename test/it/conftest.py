import asyncio
import os
import subprocess
import sys
import time

import pytest

from kpu.torch.backend import _async
from kpu.torch.server import Compute


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    # Patch the loop BEFORE pytest starts using it for reentrant support
    _async.apply(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def kpu_server():
    """Start KPU PyTorch server for integration tests."""
    port = int(os.environ.get("KPU_TEST_PORT", "50052"))

    # Start server subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "kpu.torch.server", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    time.sleep(2)

    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Server failed to start:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    yield f"localhost:{port}"

    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
async def compute(kpu_server):
    """Provide a connected Compute instance."""
    async with Compute(kpu_server) as c:
        yield c


@pytest.fixture
async def device(compute):
    """Provide a KPU device."""
    return compute.device("cpu")
