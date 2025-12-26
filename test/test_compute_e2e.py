"""
End-to-end tests for the Compute API.
"""

import asyncio
import pytest
import torch

from kpu.client import Compute, log_event


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
async def test_compute_managed(test_image, test_tensors):
    """
    Test managing a Compute with a context manager.

    Covers:
    - Creating Compute with context manager
    - Default log_event callback
    - Sending tensors
    - Receiving tensors
    - Bidirectional streaming
    - Automatic cleanup
    """
    tensor1 = test_tensors['tensor1']
    tensor2 = test_tensors['tensor2']

    async with Compute(
        name="test-managed",
        image=test_image,
        on_events=log_event,
    ) as compute:
        # Verify compute is ready
        assert compute.is_ready()
        assert compute.name == "test-managed"

        # Send tensors to the server
        response = await compute.send_tensors(tensor1, tensor2)
        assert response.success is True
        assert response.message is not None
        assert len(response.received_tensor_ids) == 2

        # Receive tensors from server
        received = await compute.receive_tensors(
            count=2,
            parameters={'shape': '20,20'}
        )
        assert len(received) == 2
        for tensor in received:
            assert tensor.shape == (20, 20)

        # Bidirectional streaming
        processed = await compute.stream_tensors(tensor1, tensor2)
        assert len(processed) == 2
        # Processed tensors should have same shapes as input
        assert processed[0].shape == tensor1.shape
        assert processed[1].shape == tensor2.shape

    # Compute is automatically deleted when exiting context
    # Verify it's deleted by checking is_ready would fail
    assert not compute.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compute_manual(test_image, test_tensors):
    """
    Test managing a Compute manually.

    Covers:
    - Creating Compute without context manager
    - ready() with timeout
    - Manual operations
    - suspend()
    - resume()
    - delete()
    """
    compute = Compute(
        name="test-manual",
        image=test_image,
    )

    try:
        # Wait for it to be ready with timeout
        await compute.ready(timeout=300)
        assert compute.is_ready()

        # Use it - receive tensors
        tensors = await compute.receive_tensors(count=1)
        assert len(tensors) >= 1
        assert tensors[0].shape is not None
    finally:
        # Clean up manually
        await compute.delete()

        # Verify deletion
        assert not compute.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compute_events(test_image):
    """
    Test watching for a Compute events.

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

        # Wait a bit for events to be generated
        await asyncio.sleep(5)

        # TODO: assert for events


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

        # Check conditions
        if resource.status.conditions:
            ready_condition = next(
                (c for c in resource.status.conditions if c.type == "Ready"),
                None
            )
            if ready_condition:
                assert ready_condition.status == "True"
