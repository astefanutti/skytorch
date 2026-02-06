# SkyTorch gRPC Server

gRPC server that executes PyTorch tensor operations on behalf of the SkyTorch backend.

## Quick Start

### Starting the Server

```bash
# Start server on default port (50051), all interfaces
python -m skytorch.torch.server

# Custom port, host, chunk size, and log level
python -m skytorch.torch.server --port 50052 --host localhost --chunk-size 524288 --log-level DEBUG

# With NVIDIA GPU metrics enabled
python -m skytorch.torch.server --metrics-sources nvidia-gpu
```

### Environment Variables

All CLI flags can be set via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SKYTORCH_PORT` | `50051` | Port to listen on |
| `SKYTORCH_HOST` | `[::]` | Host address to bind to |
| `SKYTORCH_CHUNK_SIZE` | `1048576` (1 MB) | Chunk size for streaming tensors (bytes) |
| `SKYTORCH_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `SKYTORCH_METRICS_SOURCES` | | Comma-separated list of metrics sources to enable |

### Programmatic Usage

```python
import asyncio
import grpc
from skytorch.torch.server import serve

async def main():
    server = grpc.aio.server()
    await serve(server, host="[::]", port=50051, chunk_size=1024*1024)

asyncio.run(main())
```

## RPC Modes

The server exposes two modes for executing operations, defined in `service.proto`:

### Bidirectional Streaming (`StreamOperations`)

Primary path used by the SkyTorch backend for low-latency operation dispatch.
All operations flow through a single bidirectional stream to maintain ordering.
Supports chunking for large payloads (tensor uploads/downloads).

Operations multiplexed over the stream:
- `ExecuteAtenRequest` &mdash; execute an ATen operation
- `UpdateTensorRequest` &mdash; upload tensor data to the server
- `GetTensorRequest` &mdash; download tensor data from the server
- `CopyTensorRequest` &mdash; copy tensor data on the server
- `DeleteTensorsRequest` &mdash; delete tensors by ID

### Unidirectional RPCs

Individual RPCs useful for synchronous operation and debugging:

| RPC | Direction | Description |
|---|---|---|
| `UpdateTensor` | Client &rarr; Server (stream) | Upload tensor data in chunks |
| `GetTensor` | Server &rarr; Client (stream) | Download tensor data in chunks |
| `CopyTensor` | Unary | Copy tensor storage on the server |
| `DeleteTensors` | Unary | Delete tensors by ID |
| `ExecuteAtenOperation` | Unary | Execute a single ATen operation |

## Metrics

The server supports pluggable metrics sources, enabled via the `--metrics-sources` flag.

Available sources:
- `nvidia-gpu` &mdash; NVIDIA GPU metrics (utilization, memory, temperature, etc.)

When metrics sources are enabled, the server registers the `Metrics` gRPC service with two RPCs:
- `GetMetrics` &mdash; unary RPC that returns a snapshot of all metrics
- `StreamMetrics` &mdash; server-streaming RPC that pushes metrics snapshots at a configured interval

Example:

```bash
# Start the server with GPU metrics
python -m skytorch.torch.server --metrics-sources nvidia-gpu

# Poll metrics with grpcurl
grpcurl -plaintext localhost:50051 skytorch.server.Metrics/GetMetrics
```

## Health Checks

The server implements the [gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md), reporting status for:
- Overall server health (empty service name)
- Tensor service (`skytorch.torch.Service`)
- Metrics service (`skytorch.server.Metrics`), when metrics sources are enabled

See the [health service documentation](../../server/health/README.md) for details.

## Development

### Regenerating gRPC Code

After modifying `service.proto`:

```bash
hack/gen-grpc-proto.sh
```

Generated files:
- `service_pb2.py` &mdash; Protocol buffer message classes
- `service_pb2_grpc.py` &mdash; gRPC service stubs and servicer base classes
- `service_pb2.pyi` &mdash; Type stubs

### Running Tests

Integration tests run the gRPC server in-process (no Kubernetes cluster needed):

```bash
pytest tests/it -m it
```
