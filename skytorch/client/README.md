# SkyTorch Python Client

High-level Python API for creating and managing SkyTorch Compute resources in Kubernetes with integrated gRPC tensor streaming.

## Features

- **Zero Configuration**: Auto-detects in-cluster config or default kubeconfig
- **Declarative API**: Create Compute resources using simple Python objects
- **Server-Side Apply**: Automatically creates or updates resources using Kubernetes server-side apply
- **Async/Await**: Full async support for concurrent operations
- **Context Manager**: Automatic resource cleanup with Python context managers
- **Integrated gRPC**: Direct tensor streaming without manual connection setup

## Installation

```bash
pip install kubernetes torch grpcio
```

## Quick Start

### Basic Usage

```python
import asyncio
import torch
from skytorch.client import Compute

async def main():
    # Create a Compute resource and use it
    # Namespace auto-detects from kubeconfig context or service account
    async with Compute(
        name="my-compute",
        image="localhost:5001/skytorch.torch-server:latest"
    ) as compute:
        # Send tensors
        tensor = torch.randn(100, 100)
        response = await compute.send_tensors(tensor)

        # Receive tensors
        tensors = await compute.receive_tensors(count=1)
        print(f"Received: {tensors[0].shape}")

asyncio.run(main())
```

### Managing Compute Cluster

Use the `Cluster` class to manage multiple Compute resources in parallel:

```python
import asyncio
import torch
from skytorch.client import Compute, Cluster

async def main():
    # Create a cluster of Compute resources
    async with Cluster(
        Compute(name="compute-1", image="localhost:5001/skytorch.torch-server:latest"),
        Compute(name="compute-2", image="localhost:5001/skytorch.torch-server:latest"),
        Compute(name="compute-3", image="localhost:5001/skytorch.torch-server:latest"),
    ) as (compute1, compute2, compute3):
        # All computes are ready in parallel

        # Send tensors to all computes in parallel
        tensor = torch.randn(100, 100)
        responses = await asyncio.gather(
            compute1.send_tensors(tensor),
            compute2.send_tensors(tensor),
            compute3.send_tensors(tensor),
        )

    # All computes are automatically deleted when exiting the context

asyncio.run(main())
```

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

### Creating a Compute

The Compute resource is automatically created or updated in Kubernetes when you instantiate the `Compute` class:

```python
compute = Compute(
    name="my-compute",
    image="localhost:5001/skytorch.torch-server:latest",
    env={
        "LOG_LEVEL": "DEBUG",
        "CHUNK_SIZE": "2097152",
    },
    labels={
        "team": "ml",
        "env": "prod",
    },
)
```

**Namespace configuration:**
- The namespace is determined globally via the `init()` function
- Auto-detects based on the Kubernetes client configuration:
  - **From kubeconfig**: Uses the namespace from your current kubeconfig context
  - **In-cluster**: Reads the namespace from `/var/run/secrets/kubernetes.io/serviceaccount/namespace`
  - **Default**: Falls back to `"default"` if no namespace is detected
- You can explicitly override by calling `init(namespace="my-namespace")` before creating any Compute instances

This uses Kubernetes server-side apply, so it will:
- Create the resource if it doesn't exist
- Update the resource if it already exists
- Preserve fields managed by other controllers

### Watching Events

You can monitor Kubernetes Events for your Compute resource by providing an event callback:

```python
from skytorch.client import Compute, log_event

# Option 1: Use the default logging callback
compute = Compute(
    name="my-compute",
    image="localhost:5001/skytorch.torch-server:latest",
    on_events=log_event  # Automatically logs all events
)

# Option 2: Provide a custom callback
def my_event_handler(event):
    print(f"Event: {event.reason} - {event.message}")

compute = Compute(
    name="my-compute",
    image="localhost:5001/skytorch.torch-server:latest",
    on_events=my_event_handler
)
```

The event callback receives a `V1Event` object from the Kubernetes API for each event related to the Compute resource.

### Waiting for Ready

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

### Tensor Streaming

#### Send Tensors (Client-to-Server Streaming)

```python
tensor1 = torch.randn(100, 100)
tensor2 = torch.randn(50, 50)

response = await compute.send_tensors(tensor1, tensor2)
print(response.message)
```

#### Receive Tensors (Server-to-Client Streaming)

```python
tensors = await compute.receive_tensors(
    count=2,
    parameters={'shape': '20,20'}
)

for i, tensor in enumerate(tensors):
    print(f"Tensor {i}: {tensor.shape}")
```

#### Bidirectional Streaming

```python
# Send tensors and receive processed results
input_tensors = [torch.randn(10, 10), torch.randn(20, 20)]
output_tensors = await compute.stream_tensors(*input_tensors)
```

### Resource Management

#### Suspend and Resume

```python
# Suspend the Compute (scale to zero)
await compute.suspend()

# Resume a suspended Compute
await compute.resume()
```

#### Delete

```python
# Delete the Compute resource
await compute.delete()

# Custom grace period
await compute.delete(grace_period_seconds=60)
```

#### Using Context Manager

The context manager automatically handles the lifecycle:

```python
async with Compute(name="my-compute", ...) as compute:
    # Use the compute
    await compute.send_tensors(tensor)

# Compute is automatically deleted when exiting
```

### External Access

When running outside the Kubernetes cluster, you can override the host:

```python
from skytorch.client import init, Compute

# Optional: Set namespace if different from your kubeconfig context
init(namespace="default")

# Use port-forwarding: kubectl port-forward svc/my-compute 50051:50051
compute = Compute(
    name="my-compute",
    image="localhost:5001/skytorch.torch-server:latest",
    host="localhost",  # Override host for external access
    port=50051,
)
```

### Status and Properties

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

## Architecture

### Server-Side Apply

The Compute API uses Kubernetes server-side apply for declarative resource management:

- **Field Manager**: `skytorch-python-client`
- **Force Apply**: Enabled to take ownership of fields
- **Idempotent**: Safe to call multiple times

This means you can:
1. Create a Compute by instantiating the Python object
2. Update a Compute by instantiating with the same name
3. Share management with other controllers (e.g., the SkyTorch operator manages status)

### gRPC Connection

The gRPC client is automatically initialized when the Compute becomes ready:

1. When `ready()` returns, the gRPC connection is established
2. Tensor streaming methods delegate to the underlying `TensorClient`
3. Connection is automatically closed on suspend/delete

### Default Host Resolution

- **In-cluster**: Uses Kubernetes service DNS: `{name}.{namespace}.svc.cluster.local`
- **External**: Override with `host` parameter for port-forwarding or external access

## API Reference

### `init(client_config=None, namespace=None)`

Initialize Kubernetes client configuration globally (optional).

If not called, the client will auto-initialize on first use by trying default
kubeconfig first, then falling back to in-cluster config. The namespace will also
be auto-detected from kubeconfig context or service account.

**Important:** This function can only be called once, and must be called before creating
any Compute instances. Calling it after initialization will raise a `RuntimeError`.

**Parameters:**
- `client_config` (kubernetes.client.Configuration, optional): Custom Kubernetes client configuration.
  Use this for advanced settings like custom API server URLs, authentication, timeouts, etc.
- `namespace` (str, optional): Namespace to use for all Compute resources. If not provided,
  it will be auto-detected from kubeconfig context or service account.

**Raises:**
- `RuntimeError`: If the client has already been initialized.

**Example:**
```python
from kubernetes import client
from skytorch.client import init

# Override namespace only
init(namespace="my-namespace")

# Advanced configuration with custom namespace
config = client.Configuration()
config.host = "https://my-cluster.example.com"
init(client_config=config, namespace="my-namespace")

# Or let everything auto-detect
# No need to call init() at all
```

### `Compute(name, **kwargs)`

Create or update a Compute resource.

The namespace is determined by the global `init()` configuration and auto-detects
from kubeconfig context or in-cluster service account.

**Parameters:**
- `name` (str): Name of the Compute resource
- `image` (str, optional): Container image for the Compute runtime
- `command` (List[str], optional): Entrypoint command override
- `args` (List[str], optional): Arguments for the entrypoint
- `env` (Dict[str, str], optional): Environment variables
- `labels` (Dict[str, str], optional): Labels to apply
- `annotations` (Dict[str, str], optional): Annotations to apply
- `suspend` (bool): Whether to create in suspended state (default: False)
- `host` (str, optional): Override gRPC host
- `port` (int): gRPC port (default: 50051)
- `on_events` (Callable[[V1Event], None], optional): Callback function to receive Events for this Compute resource

**Methods:**
- `ready(timeout=300)`: Wait for Compute to be ready using watch
- `is_ready()`: Check if the Compute is ready (fetches current status from Kubernetes)
- `send_tensors(*tensors)`: Send tensors to server
- `receive_tensors(count=1, parameters=None)`: Receive tensors from server
- `stream_tensors(*tensors)`: Bidirectional streaming
- `suspend()`: Suspend the Compute
- `resume()`: Resume a suspended Compute
- `delete(grace_period_seconds=30)`: Delete the Compute

**Properties:**
- `resource` (ComputeV1alpha1Compute): Underlying Kubernetes resource

### `Cluster(*computes)`

Manage multiple Compute resources in parallel.

The Cluster class provides an async context manager that manages the lifecycle
of multiple Compute instances simultaneously, waiting for all of them to be
ready in parallel and cleaning up all resources on exit.

**Parameters:**
- `*computes` (Compute): Variable number of Compute instances to manage

**Methods:**
- `ready(timeout=None)`: Wait for all Compute instances to become ready in parallel
- `is_ready()`: Check if all Compute instances are ready
- `delete()`: Delete all Compute resources in parallel

**Special Methods:**
- `__iter__()`: Iterate over the managed Compute instances
- `__len__()`: Get the number of managed Compute instances
- `__getitem__(index)`: Access a Compute instance by index

**Example:**
```python
from skytorch.client import Compute, Cluster

async with Cluster(
    Compute(name="compute-1", image="my-image:latest"),
    Compute(name="compute-2", image="my-image:latest"),
) as (compute1, compute2):
    # All computes are ready in parallel
    response1, response2 = await asyncio.gather(
        compute1.send_tensors(tensor),
        compute2.send_tensors(tensor),
    )

# All computes are deleted in parallel
```

**Error Handling:**

If multiple Compute instances fail during context exit, the errors are grouped
together in an `ExceptionGroup` for comprehensive error reporting:

```python
try:
    async with Cluster(...) as cluster:
        # Use cluster
        pass
except ExceptionGroup as eg:
    # Handle multiple failures
    for exc in eg.exceptions:
        print(f"Error: {exc}")
```

### `log_event(event)`

Default event callback that logs Compute events.

This callback can be passed to the Compute constructor's `on_events` parameter
to automatically log all events related to the Compute resource.

**Parameters:**
- `event` (V1Event): Kubernetes Event object

**Example:**
```python
from skytorch.client import Compute, log_event

compute = Compute(
    name="my-compute",
    image="localhost:5001/skytorch.torch-server:latest",
    on_events=log_event  # Logs all events
)
```

The function logs events with different log levels based on event type:
- Normal events: INFO level
- Warning events: WARNING level
- Error events: ERROR level

## Requirements

- Python 3.11+
- kubernetes
- torch
- grpcio
- Access to a Kubernetes cluster with SkyTorch operator installed
