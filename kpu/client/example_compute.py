"""
Example usage of the KPU Compute API.

This demonstrates how to create a Compute resource, wait for it to be ready,
and use it to stream PyTorch tensors.
"""

import asyncio
import logging
import torch

from kpu.client import Compute, log_event


async def main():
    """Example usage of the Compute API."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # No configuration needed! The Kubernetes client and namespace auto-configure:
    # 1. Tries default kubeconfig first (~/.kube/config)
    # 2. Falls back to in-cluster config (when running in a pod)
    # 3. Namespace auto-detects from kubeconfig context or service account
    #
    # For advanced configuration (optional):
    # from kubernetes import client
    # from kpu.client import init
    # config = client.Configuration()
    # config.host = "https://my-cluster.example.com"
    # init(client_config=config, namespace="my-namespace")

    # Create example tensors
    tensor1 = torch.randn(100, 100)
    tensor2 = torch.randn(50, 50)

    # Example 1: Using context manager (recommended)
    # The Compute resource is created/updated when instantiated,
    # then we wait for it to be ready, use it, and delete it on exit
    # Namespace is auto-detected from kubeconfig context or service account
    async with Compute(
        name="test-sdk",
        image="ghcr.io/astefanutti/kpu-torch-server@sha256:6ae85f768ce84fb7002e5e0e1536a585ad8e99442dab34763f73d664923140ab",
        env={
            "LOG_LEVEL": "INFO",
        },
        on_events=log_event,
    ) as compute:
        print(f"Compute is ready: {compute.is_ready()}")

        # Send tensors to the server
        print("Sending tensors to server...")
        response = await compute.send_tensors(tensor1, tensor2)
        print(f"Server response: {response.message}")
        print(f"Received tensor IDs: {response.received_tensor_ids}")

        # Receive tensors from server
        print("Receiving tensors from server...")
        received = await compute.receive_tensors(
            count=2,
            parameters={'shape': '20,20'}
        )
        print(f"Received {len(received)} tensors")
        for i, tensor in enumerate(received):
            print(f"  Tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")

        # Bidirectional streaming
        print("Bidirectional streaming...")
        processed = await compute.stream_tensors(tensor1, tensor2)
        print(f"Received {len(processed)} processed tensors")
        for i, tensor in enumerate(processed):
            print(f"  Processed tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Compute is automatically deleted when exiting the context manager

    # Example 2: Manual lifecycle management
    compute = Compute(
        name="my-compute-2",
        image="localhost:5001/kpu-torch-server:latest",
    )

    # Wait for it to be ready
    await compute.ready(timeout=300)

    # Use it
    tensors = await compute.receive_tensors(count=1)
    print(f"Received tensor: shape={tensors[0].shape}")

    # Suspend it (scale to zero)
    await compute.suspend()
    print("Compute suspended")

    # Resume it
    await compute.resume()
    print("Compute resumed and ready")

    # Clean up manually
    await compute.delete()

    # Example 3: Connecting to an existing Compute from outside the cluster
    # If you're running outside the cluster, override the host
    # To use a specific namespace: from kpu.client import init; init(namespace="my-namespace")
    compute_external = Compute(
        name="my-compute-3",
        image="localhost:5001/kpu-torch-server:latest",
        host="localhost",  # Override to use port-forwarding
        port=50051,
    )

    async with compute_external:
        tensors = await compute_external.receive_tensors(count=1)
        print(f"Received from external: shape={tensors[0].shape}")


if __name__ == '__main__':
    asyncio.run(main())
