# gRPC Health Checking Service

A reusable Python implementation of the [gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md).

## Overview

This package provides a standard gRPC health checking service that can be integrated into any gRPC server.
It implements both the `Check` (unary) and `Watch` (streaming) RPCs for health status monitoring.

## Features

- **Standard Protocol**: Implements the official gRPC Health Checking Protocol
- **Service-Level Health**: Track health status for individual services
- **Overall Server Health**: Monitor overall server status
- **Watch Support**: Stream health status updates in real-time
- **Async/Await**: Fully asynchronous implementation using Python's `asyncio`

## Quick Start

### Adding Health Service to Your gRPC Server

```python
import grpc
from skytorch.server.health import HealthServicer
from skytorch.server.health import health_pb2
from skytorch.server.health import health_pb2_grpc

async def serve():
    server = grpc.aio.server()

    # Add your application services
    # my_service_pb2_grpc.add_MyServiceServicer_to_server(my_servicer, server)

    # Add health service
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Set health status for your services
    health_servicer.set_service_status(
        "my.application.MyService",
        health_pb2.HealthCheckResponse.SERVING
    )

    # Set overall server health
    health_servicer.set_service_status(
        "",
        health_pb2.HealthCheckResponse.SERVING
    )

    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
```

### Checking Health from a Client

```python
import grpc
from skytorch.server.health import health_pb2
from skytorch.server.health import health_pb2_grpc

async def check_health():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = health_pb2_grpc.HealthStub(channel)

        # Check overall server health
        request = health_pb2.HealthCheckRequest(service="")
        response = await stub.Check(request)
        print(f"Server health: {response.status}")

        # Check specific service health
        request = health_pb2.HealthCheckRequest(service="my.application.MyService")
        response = await stub.Check(request)
        print(f"Service health: {response.status}")
```

### Watching Health Status

```python
async def watch_health():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = health_pb2_grpc.HealthStub(channel)

        # Watch for health status updates
        request = health_pb2.HealthCheckRequest(service="")
        async for response in stub.Watch(request):
            print(f"Health status update: {response.status}")
```

## API Reference

### HealthServicer

The main health service implementation.

#### Methods

**`set_service_status(service: str, status: ServingStatus) -> None`**

Set the health status for a service.

- **Parameters**:
  - `service`: Service name (use empty string `""` for overall server health)
  - `status`: One of:
    - `health_pb2.HealthCheckResponse.SERVING`
    - `health_pb2.HealthCheckResponse.NOT_SERVING`
    - `health_pb2.HealthCheckResponse.SERVICE_UNKNOWN`

**`get_service_status(service: str) -> ServingStatus`**

Get the current health status for a service.

- **Parameters**:
  - `service`: Service name
- **Returns**: Current serving status

#### gRPC Methods

**`Check(request, context) -> HealthCheckResponse`**

Unary RPC to check the health status of a service.

**`Watch(request, context) -> AsyncIterator[HealthCheckResponse]`**

Streaming RPC to watch for health status changes.

## Health Status Values

The health service supports the following status values:

- **`SERVING`** (1): The service is healthy and serving requests
- **`NOT_SERVING`** (2): The service is not healthy and not serving requests
- **`SERVICE_UNKNOWN`** (3): The service is not registered (returned automatically for unknown services)
- **`UNKNOWN`** (0): Status is unknown (not typically used)

## Integration Examples

### Kubernetes Readiness/Liveness Probes

Use the [grpc-health-probe](https://github.com/grpc-ecosystem/grpc-health-probe) tool in Kubernetes:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: app
    image: my-app:latest
    ports:
    - containerPort: 50051
    livenessProbe:
      exec:
        command: ["/bin/grpc_health_probe", "-addr=:50051"]
      initialDelaySeconds: 5
    readinessProbe:
      exec:
        command: ["/bin/grpc_health_probe", "-addr=:50051", "-service=my.application.MyService"]
      initialDelaySeconds: 10
```

### Dynamic Health Updates

Update health status based on your application state:

```python
from skytorch.server.health import HealthServicer
from skytorch.server.health import health_pb2

class MyApplication:
    def __init__(self, health_servicer: HealthServicer):
        self.health = health_servicer
        self.db_connected = False

    async def connect_database(self):
        try:
            await self.db.connect()
            self.db_connected = True
            self.health.set_service_status(
                "my.app.Database",
                health_pb2.HealthCheckResponse.SERVING
            )
        except Exception:
            self.health.set_service_status(
                "my.app.Database",
                health_pb2.HealthCheckResponse.NOT_SERVING
            )

    async def on_shutdown(self):
        # Mark service as not serving during shutdown
        self.health.set_service_status(
            "",
            health_pb2.HealthCheckResponse.NOT_SERVING
        )
```

## Protocol Details

The health checking protocol uses the following proto definition:

```protobuf
syntax = "proto3";

package grpc.health.v1;

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
    SERVICE_UNKNOWN = 3;
  }
  ServingStatus status = 1;
}

service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}
```

## Regenerating Protocol Buffers

If you modify `health.proto`:

```bash
./hack/gen-grpc-proto.sh
```

This generates:
- `health_pb2.py`: Protocol buffer messages
- `health_pb2_grpc.py`: gRPC service stubs
- `health_pb2.pyi`: Type hints

## Best Practices

1. **Overall Server Health**: Always set the overall server health using an empty service name (`""`). This allows health checks without knowing specific service names.

2. **Service-Specific Health**: Register individual service health for fine-grained monitoring. Use fully qualified service names (e.g., `"package.ServiceName"`).

3. **Update on State Changes**: Update health status whenever your application state changes (database connections, dependencies, graceful shutdown, etc.).

4. **Use Watch for Monitoring**: Implement health monitoring using the `Watch` RPC to receive real-time updates instead of polling with `Check`.

5. **Graceful Shutdown**: Set health status to `NOT_SERVING` during graceful shutdown to stop receiving new requests.
