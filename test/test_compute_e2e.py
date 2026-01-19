"""
End-to-end tests for the Compute API.
"""

import asyncio
import pytest
import torch

import kpu.torch.backend  # noqa: F401 - Register 'kpu' device
from kpu.client import Compute, log_event


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compute_managed(test_image):
    """
    Test managing a Compute with a context manager.

    Covers:
    - Creating Compute with context manager
    - Creating tensors on KPU device
    - Performing PyTorch operations remotely
    - Copying results back to CPU
    - Verifying actual computation results
    - Automatic cleanup
    """
    async with Compute(
        name="test-managed",
        image=test_image,
        on_events=log_event,
    ) as compute:
        assert compute.is_ready()
        assert compute.name == "test-managed"

        # Get KPU device mapped to remote CPU
        device = compute.device("cpu")

        # Create reference tensors on CPU
        x_cpu = torch.randn(10, 10)
        y_cpu = torch.randn(10, 10)

        # Transfer to KPU
        x = x_cpu.to(device)
        y = y_cpu.to(device)

        # Perform operations on KPU
        z = x + y
        w = torch.matmul(x, y)

        # Copy results back to CPU
        z_result = z.cpu()
        w_result = w.cpu()

        # Verify actual computation results
        expected_z = x_cpu + y_cpu
        expected_w = torch.matmul(x_cpu, y_cpu)
        assert torch.allclose(z_result, expected_z)
        assert torch.allclose(w_result, expected_w)

    # Compute is automatically deleted when exiting context
    assert not compute.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compute_manual(test_image):
    """
    Test managing a Compute manually.

    Covers:
    - Creating Compute without context manager
    - ready() with timeout
    - CPU to KPU data transfer
    - KPU to CPU data transfer
    - Verifying actual computation results
    - Manual delete()
    """
    compute = Compute(
        name="test-manual",
        image=test_image,
    )

    try:
        await compute.ready(timeout=300)
        assert compute.is_ready()

        device = compute.device("cpu")

        # Transfer CPU tensor to KPU
        cpu_input = torch.randn(100, 100)
        kpu_tensor = cpu_input.to(device)

        # Perform operation
        result = kpu_tensor * 2

        # Transfer back to CPU and verify actual computation result
        cpu_result = result.cpu()
        expected = cpu_input * 2
        assert torch.allclose(cpu_result, expected)

    finally:
        await compute.delete()
        assert not compute.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compute_events(test_image):
    """
    Test watching for Compute events.

    Covers:
    - Event callback registration
    - Event handling
    """
    events_received = []

    def custom_event_handler(event):
        events_received.append({
            'reason': event.reason,
            'message': event.message,
            'type': event.type
        })

    async with Compute(
        name="test-events",
        image=test_image,
        on_events=custom_event_handler,
    ) as compute:
        assert compute.is_ready()
        await asyncio.sleep(5)
        # Events are received asynchronously


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compute_resource(test_image):
    """
    Test accessing a Compute resource.

    Covers:
    - Resource property access
    - Status conditions
    - Metadata
    """
    async with Compute(
        name="test-resource",
        image=test_image,
    ) as compute:
        resource = compute.resource

        assert resource is not None
        assert resource.metadata is not None
        assert resource.metadata.name == "test-resource"
        assert resource.status is not None

        if resource.status.conditions:
            ready_condition = next(
                (c for c in resource.status.conditions if c.type == "Ready"),
                None
            )
            if ready_condition:
                assert ready_condition.status == "True"
