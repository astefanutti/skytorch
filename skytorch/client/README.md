# SkyTorch Python Client

High-level Python API for creating and managing SkyTorch Compute resources in Kubernetes with integrated gRPC tensor streaming.

## Features

- **Zero Configuration**: Auto-detects in-cluster config or default kubeconfig
- **Declarative API**: Create Compute resources using simple Python objects
- **Server-Side Apply**: Automatically creates or updates resources using Kubernetes server-side apply
- **Async/Await**: Full async support for concurrent operations
- **Context Manager**: Automatic resource cleanup with Python context managers
- **Integrated gRPC**: Direct tensor streaming without manual connection setup

## Quick Start

### `@compute` Decorator

```python
import asyncio
import torch
from skytorch.client import Compute, compute

@compute(
    name="my-compute",
    image="ghcr.io/astefanutti/skytorch-server:latest",
)
async def main(node: Compute):
    device = node.device("cuda")

    x = torch.randn(4, 4, device=device)
    y = x @ x.T
    print(y)

asyncio.run(main())
```

### `Cluster` manager

Use the `Cluster` class to manage multiple Compute resources in parallel, for example
to run GRPO training with a trainer and a vLLM inference server on separate GPUs:

```python
import asyncio
import copy
from transformers import AutoModelForCausalLM
from skytorch.client import Compute, Cluster

async def main():
    async with Cluster(
        Compute(name="trainer"),
        Compute(name="vllm"),
    ) as (trainer, vllm):
        trainer_device = trainer.device("cuda")
        vllm_device = vllm.device("cuda")

        # Load the policy model on the trainer and copy it to vLLM
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
        model.to(trainer_device)
        ref_model = copy.deepcopy(model).to(vllm_device)

        for step in range(10):
            # GRPO training step on the trainer device
            # ...

            # Sync weights from trainer to vLLM
            for p, ref_p in zip(model.parameters(), ref_model.parameters()):
                ref_p.data.copy_(p.data)

asyncio.run(main())
```

> **Note:** Cross-compute tensor copy is not supported yet. This example illustrates a future capability.

### Configuration

The Kubernetes client auto-configures on first use:
1. Tries default kubeconfig first (`~/.kube/config`)
2. Falls back to in-cluster configuration (when running in a pod)

**No configuration needed for most use cases!**

For advanced configuration (optional):

```python
from kubernetes import client
from skytorch.client import init

# Advanced custom configuration
config = client.Configuration()
config.host = "https://my-k8s-cluster.example.com"
config.api_key_prefix['authorization'] = 'Bearer'
config.api_key['authorization'] = 'my-token'
init(client_config=config)

# Or just let it auto-detect (recommended)
# No need to call init() at all
```

### Events

You can monitor Events for your Compute resource by providing an event callback:

```python
from skytorch.client import Compute, log_event

# Option 1: Use the default logging callback
compute = Compute(
    name="my-compute",
    on_events=log_event  # Automatically logs all events
)

# Option 2: Provide a custom callback
def my_event_handler(event):
    print(f"Event: {event.reason} - {event.message}")

compute = Compute(
    name="my-compute",
    on_events=my_event_handler
)
```

### Readiness

Wait for a Compute resource to become ready before using it.
The context manager does this automatically, but you can also wait explicitly:

```python
# Wait for the Compute to become ready (default timeout: 300s)
await compute.ready()

# Custom timeout
await compute.ready(timeout=600)

# Using context manager (automatically waits)
async with compute:
    # Compute is guaranteed to be ready here
    pass
```

### Status

Inspect the current state of a Compute resource, including its readiness and conditions:

```python
# Check if ready
if compute.is_ready():
    print("Compute is ready!")

# Access the underlying Kubernetes resource
resource = compute.resource
if resource and resource.status:
    for condition in resource.status.conditions:
        print(f"{condition.type}={condition.status}")
```

### Metrics

Stream GPU metrics from a Compute resource by providing an `on_metrics` callback:

```python
from skytorch.client import Compute, compute

@compute(
    name="my-compute",
    on_metrics=lambda snapshot: print(snapshot),
)
async def main(node: Compute):
    device = node.device("cuda")
    # Metrics are streamed in the background while the function runs
    # ...
```
