"""
Pytest configuration and shared fixtures for SkyTorch E2E tests.
"""

import asyncio
import os
import pytest


# Test configuration constants
@pytest.fixture(scope="session")
def test_image():
    """Get the test image for Compute resources."""
    return os.getenv(
        "SKYTORCH_TEST_IMAGE",
        "ghcr.io/astefanutti/skytorch-server"
    )


# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup_delay():
    """Add small delay between tests to allow Kubernetes cleanup."""
    yield
    # Small delay after each test to allow resources to be fully cleaned up
    await asyncio.sleep(0.5)
