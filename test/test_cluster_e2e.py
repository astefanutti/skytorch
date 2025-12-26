"""
End-to-end tests for the Cluster API.
"""

import asyncio
import pytest
import torch

from kpu.client import Compute, Cluster, log_event


@pytest.fixture
def test_tensors():
    """Create test tensors for operations."""
    return {
        'tensor1': torch.randn(100, 100),
        'tensor2': torch.randn(50, 50),
        'tensor3': torch.randn(20, 20)
    }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_managed(test_image, test_tensors):
    """
    Test managing cluster of Compute resources in parallel.

    Covers:
    - Creating Cluster with multiple Computes
    - Default log_event callback
    - Unpacking computes from cluster
    - Parallel operations with asyncio.gather
    - Automatic cleanup
    """
    tensor1 = test_tensors['tensor1']
    tensor2 = test_tensors['tensor2']

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
        # Verify both computes are ready
        assert compute1.is_ready()
        assert compute2.is_ready()
        assert compute1.name == "test-cluster-1"
        assert compute2.name == "test-cluster-2"

        # Send tensors to both servers in parallel
        response1, response2 = await asyncio.gather(
            compute1.send_tensors(tensor1),
            compute2.send_tensors(tensor2),
        )

        assert response1.success is True
        assert response2.success is True

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

    # Wait for ready
    await cluster.ready(timeout=300)

    # Manual delete
    await cluster.delete()

    # Verify all are deleted
    assert not cluster.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_parallel_receive(test_image, test_tensors):
    """
    Test receiving tensors from multiple Computes in parallel.

    Covers:
    - Parallel receive_tensors operations
    - Gathering results from multiple Computes
    """
    async with Cluster(
        Compute(name="test-recv-1", image=test_image),
        Compute(name="test-recv-2", image=test_image),
    ) as (compute1, compute2):
        # Receive from both in parallel
        tensors1, tensors2 = await asyncio.gather(
            compute1.receive_tensors(count=2, parameters={'shape': '10,10'}),
            compute2.receive_tensors(count=2, parameters={'shape': '20,20'}),
        )

        # Verify results
        assert len(tensors1) == 2
        assert len(tensors2) == 2
        assert all(t.shape == (10, 10) for t in tensors1)
        assert all(t.shape == (20, 20) for t in tensors2)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_parallel_streaming(test_image, test_tensors):
    """
    Test bidirectional streaming across multiple Computes.

    Covers:
    - Parallel stream_tensors operations
    - Processing on multiple Computes simultaneously
    """
    tensor1 = test_tensors['tensor1']
    tensor2 = test_tensors['tensor2']

    async with Cluster(
        Compute(name="test-stream-1", image=test_image),
        Compute(name="test-stream-2", image=test_image),
    ) as (compute1, compute2):
        # Stream to both in parallel
        processed1, processed2 = await asyncio.gather(
            compute1.stream_tensors(tensor1),
            compute2.stream_tensors(tensor2),
        )

        # Verify results
        assert len(processed1) >= 1
        assert len(processed2) >= 1


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

        # Wait a bit for events
        await asyncio.sleep(5)

        # TODO: assert for events


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cluster_error_handling(test_image):
    """
    Test error handling when cluster operations fail.

    Covers:
    - ExceptionGroup on multiple failures
    - Cleanup on error
    """
    # This test intentionally creates computes that might have issues
    # to test error handling
    # For now, just verify normal cleanup works
    async with Cluster(
        Compute(name="test-error-1", image=test_image),
        Compute(name="test-error-2", image=test_image),
    ) as cluster:
        assert len(cluster) == 2
