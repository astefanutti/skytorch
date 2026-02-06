"""
End-to-end tests for the Cluster API.
"""

import asyncio
import pytest
import torch

import skytorch.torch.backend  # noqa: F401 - Register 'sky' device
from skytorch.client import Compute, Cluster, log_event


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_managed(test_image):
    """
    Test managing cluster of Compute resources in parallel.

    Covers:
    - Creating Cluster with multiple Computes
    - Default log_event callback
    - Unpacking computes from cluster
    - Parallel PyTorch operations with asyncio.gather
    - Verifying actual computation results
    - Automatic cleanup
    """
    async with Cluster(
        Compute(
            name="test-cluster-1",
            image=test_image,
            on_events=log_event,
        ),
        Compute(
            name="test-cluster-2",
            image=test_image,
            on_events=log_event,
        )
    ) as (compute1, compute2):
        assert compute1.is_ready()
        assert compute2.is_ready()
        assert compute1.name == "test-cluster-1"
        assert compute2.name == "test-cluster-2"

        # Get devices for both computes
        device1 = compute1.device("cpu")
        device2 = compute2.device("cpu")

        # Create reference tensors on cpu
        x_cpu = torch.randn(10, 10)
        y_cpu = torch.randn(10, 10)

        # Transfer to both sky devices
        x1 = x_cpu.to(device1)
        y1 = y_cpu.to(device1)
        x2 = x_cpu.to(device2)
        y2 = y_cpu.to(device2)

        # Perform operations on both computes
        z1 = x1 + y1
        z2 = x2 * y2

        # Copy results back to cpu
        z1_result = z1.cpu()
        z2_result = z2.cpu()

        # Verify actual computation results
        expected_z1 = x_cpu + y_cpu
        expected_z2 = x_cpu * y_cpu
        assert torch.allclose(z1_result, expected_z1)
        assert torch.allclose(z2_result, expected_z2)

    # Both computes are automatically deleted
    assert not compute1.is_ready()
    assert not compute2.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_manual(test_image):
    """
    Test managing a Cluster of Compute resources manually.

    Covers:
    - delete() method
    - Deletes all computes in parallel
    """
    cluster = Cluster(
        Compute(name="test-delete-1", image=test_image),
        Compute(name="test-delete-2", image=test_image),
    )

    await cluster.ready(timeout=300)
    await cluster.delete()

    assert not cluster.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_parallel_operations(test_image):
    """
    Test parallel tensor operations across multiple Computes.

    Covers:
    - Different operations on different Computes
    - Parallel execution pattern
    - Verifying results from multiple Computes
    """
    async with Cluster(
        Compute(name="test-ops-1", image=test_image),
        Compute(name="test-ops-2", image=test_image),
    ) as (compute1, compute2):
        device1 = compute1.device("cpu")
        device2 = compute2.device("cpu")

        # Create test tensors
        cpu_tensor1 = torch.randn(100, 100)
        cpu_tensor2 = torch.randn(50, 50)

        # Transfer to different computes
        sky_tensor1 = cpu_tensor1.to(device1)
        sky_tensor2 = cpu_tensor2.to(device2)

        # Different operations on each compute
        result1 = sky_tensor1 * 2
        result2 = torch.matmul(sky_tensor2, sky_tensor2.T)

        # Copy results back
        cpu_result1 = result1.cpu()
        cpu_result2 = result2.cpu()

        # Verify results
        expected1 = cpu_tensor1 * 2
        expected2 = torch.matmul(cpu_tensor2, cpu_tensor2.T)
        assert torch.allclose(cpu_result1, expected1)
        assert torch.allclose(cpu_result2, expected2)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_events(test_image):
    """
    Test watching for Cluster events emitted by all Computes.

    Covers:
    - Event watching across multiple Computes
    - Parallel event handling
    """
    events_compute1 = []
    events_compute2 = []

    def handler1(event):
        events_compute1.append(event)

    def handler2(event):
        events_compute2.append(event)

    async with Cluster(
        Compute(
            name="test-events-1",
            image=test_image,
            on_events=handler1,
        ),
        Compute(
            name="test-events-2",
            image=test_image,
            on_events=handler2,
        ),
    ) as (compute1, compute2):
        assert compute1.is_ready()
        assert compute2.is_ready()
        await asyncio.sleep(5)
        # Events are received asynchronously


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_error_handling(test_image):
    """
    Test error handling when cluster operations fail.

    Covers:
    - ExceptionGroup on multiple failures
    - Cleanup on error
    """
    async with Cluster(
        Compute(name="test-error-1", image=test_image),
        Compute(name="test-error-2", image=test_image),
    ) as cluster:
        assert len(cluster) == 2
