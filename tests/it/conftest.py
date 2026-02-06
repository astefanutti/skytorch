import asyncio
import os
import socket
import threading

import grpc
import pytest

from skytorch.torch.server import Compute, serve


def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


class ServerThread(threading.Thread):
    """Run gRPC server in a separate thread with its own event loop."""

    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.server: grpc.aio.Server | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started = threading.Event()
        self._stop_event = threading.Event()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._start_server())
        finally:
            self.loop.close()

    async def _start_server(self):
        self.server = grpc.aio.server()
        # Start serving in background task
        task = asyncio.create_task(
            serve(self.server, host="localhost", port=self.port)
        )
        # Wait briefly for server to bind, then signal ready
        await asyncio.sleep(0.5)
        self.started.set()
        # Wait for stop signal or termination
        await task

    def stop(self):
        if self.server and self.loop and self.loop.is_running():
            # Schedule stop on the server's event loop
            future = asyncio.run_coroutine_threadsafe(
                self.server.stop(grace=0), self.loop
            )
            try:
                future.result(timeout=5)
            except TimeoutError:
                pass  # Server stopped, loop may have exited
        # Wait for thread to finish
        self.join(timeout=5)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_sky_state():
    """
    Reset SkyTorch device and runtime state before each test.

    This fixture works around a PyTorch autograd engine limitation:
    PyTorch sizes its internal `device_ready_queues_` once during the first
    backward() call based on deviceCount(), and never resizes it. Without
    this reset, each test would get an incrementing device index (0, 1, 2...),
    but the autograd queues remain sized for device 0 only, causing:

        RuntimeError: 0 <= device.index() && device.index() <
        static_cast<c10::DeviceIndex>(device_ready_queues_.size())
        INTERNAL ASSERT FAILED

    To properly reset PyTorch's autograd engine would require:
    1. Calling Engine::reinitialize() which destroys/reconstructs the engine
    2. This is only exposed via fork() handling (pthread_atfork)
    3. Alternative: run tests with pytest --forked for full isolation

    By resetting the device index to 0 between tests, all tests use the same
    device index that the autograd engine already has queues for.
    """
    from skytorch.torch.backend._device import device_manager
    from skytorch.torch.backend._runtime import runtime_manager

    # Reset before test
    device_manager.reset()
    runtime_manager.reset()

    yield

    # Cleanup after test
    device_manager.reset()
    runtime_manager.reset()


@pytest.fixture(scope="session")
def sky_server():
    """Start SkyTorch PyTorch server in-process for integration tests."""
    port = int(os.environ.get("SKYTORCH_TEST_PORT", "50052"))

    # Pre-flight check: fail fast if port is in use
    if not check_port_available(port):
        pytest.fail(
            f"Port {port} is already in use. "
            f"Check for stale processes with: lsof -i :{port}"
        )

    # Start server in separate thread with its own event loop
    server_thread = ServerThread(port)
    server_thread.start()

    # Wait for server to be ready
    if not server_thread.started.wait(timeout=10):
        pytest.fail("Server failed to start within timeout")

    yield f"localhost:{port}"

    # Graceful shutdown
    server_thread.stop()


@pytest.fixture
async def compute(sky_server):
    """Provide a connected Compute instance."""
    async with Compute(sky_server) as c:
        yield c


@pytest.fixture
async def device(compute):
    """Provide a SkyTorch device."""
    return compute.device("cpu")
