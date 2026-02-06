# PyTorch Tensor Streaming gRPC Server

A Python gRPC server for streaming PyTorch tensors using async methods.
This implementation follows PyTorch best practices for tensor serialization and provides efficient stream-based communication.

## Features

- **Async Streaming**: Fully asynchronous gRPC server using Python's `asyncio`
- **Three Streaming Modes**:
  - Client-to-server streaming (`ReceiveTensors`)
  - Server-to-client streaming (`SendTensors`)
  - Bidirectional streaming (`StreamTensors`)
- **Efficient Serialization**: Stream-based tensor serialization using `torch.save()`/`torch.load()`
- **Chunked Transfer**: Large tensors are automatically split into configurable chunks
- **Metadata Support**: Include tensor metadata (shape, dtype, device) with each transfer
- **Health Checking**: Implements gRPC Health Checking Protocol for service monitoring

## Quick Start

### Starting the Server

Run the server using the command-line interface:

```bash
# Start server on default port (50051)
python -m skytorch.torch.server

# Customize port, host, and chunk size
python -m skytorch.torch.server --port 8080 --host localhost --chunk-size 524288

# Set log level
python -m skytorch.torch.server --log-level DEBUG
```

Or programmatically:

```python
import asyncio
import grpc
from skytorch.torch.server import serve

async def main():
    server = grpc.aio.server()
    await serve(server, host="[::]", port=50051, chunk_size=1024*1024)

asyncio.run(main())
```

### Using the Client

```python
import asyncio
import torch
from skytorch.torch.client.service import TensorClient


async def main():
    tensor = torch.randn(100, 100)

    async with TensorClient(host='localhost', port=50051) as client:
        # Send tensors to server
        response = await client.send_tensors(tensor)
        print(f"Server received: {response.message}")

        # Receive tensors from server
        tensors = await client.receive_tensors(count=2, parameters={'shape': '50,50'})
        print(f"Received {len(tensors)} tensors")

        # Bidirectional streaming
        processed = await client.stream_tensors(tensor)
        print(f"Received {len(processed)} processed tensors")


asyncio.run(main())
```

## Architecture

### Serialization (`serialization.py`)

The serialization module provides stream-based tensor serialization:

- **`serialize_tensor_to_chunks()`**: Converts a PyTorch tensor into an iterator of chunks
  - Uses `torch.save()` for serialization (PyTorch best practice)
  - Automatically splits large tensors into configurable chunks (default: 1MB)
  - Includes metadata (shape, dtype, device) in first chunk
  - Generates unique tensor IDs for tracking

- **`TensorAssembler`**: Reassembles chunks back into complete tensors
  - Buffers incoming chunks by tensor ID
  - Handles out-of-order chunk delivery
  - Uses `torch.load()` for deserialization
  - Automatically cleans up after tensor assembly

### Server (`server.py`)

The server implements the `Service` gRPC service with three methods:

1. **ReceiveTensors** (Client → Server streaming)
   - Receives tensor chunks from client
   - Assembles complete tensors
   - Returns summary response

2. **SendTensors** (Server → Client streaming)
   - Generates or retrieves tensors
   - Streams chunks to client
   - Supports request parameters

3. **StreamTensors** (Bidirectional streaming)
   - Receives tensors from client
   - Processes tensors (override `_process_tensor()`)
   - Streams processed tensors back

### Protocol Buffers (`service.proto`)

The proto file defines:
- `Service`: The gRPC service with three streaming methods
- `TensorChunk`: Message for streaming tensor data in chunks
- `TensorRequest`/`TensorResponse`: Request and response messages

### Health Service

The server includes a [gRPC Health Checking service](../../server/health/README.md) that implements the standard health checking protocol. The health service is automatically configured to report:
- Overall server health (empty service name)
- Tensor service health (`skytorch.torch.Service`)

This allows clients and orchestration systems (like Kubernetes) to monitor the server's health using tools like [grpc-health-probe](https://github.com/grpc-ecosystem/grpc-health-probe).

For complete documentation on the health service, see [`skytorch/server/health/README.md`](../../server/health/README.md).

## Customization

### Custom Tensor Processing

Override the `_process_tensor()` method in `TensorServicer`:

```python
from skytorch.torch.server.server import TensorServicer
import torch

class MyTensorServicer(TensorServicer):
    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # Your custom processing logic
        return tensor * 2 + 1
```

### Custom Tensor Source

Override the `_get_tensor_to_send()` method:

```python
class MyTensorServicer(TensorServicer):
    def _get_tensor_to_send(self, index: int, parameters: dict) -> torch.Tensor:
        # Your custom logic to retrieve/generate tensors
        # Could load from database, file system, etc.
        return my_tensor_database.get(index)
```

### Adjust Chunk Size

**Command-line:**
```bash
# 2MB chunks
python -m skytorch.torch.server --chunk-size 2097152
```

**Programmatic:**
```python
import asyncio
import grpc
from skytorch.torch.server import serve

async def main():
    server = grpc.aio.server()
    await serve(server, chunk_size=2*1024*1024)  # 2MB chunks

asyncio.run(main())
```

**Serialization:**
```python
from skytorch.torch.server import serialize_tensor_to_chunks

for chunk in serialize_tensor_to_chunks(tensor, chunk_size=512*1024):
    # Process 512KB chunks
    pass
```

## Protocol Details

### TensorChunk Message

```protobuf
message TensorChunk {
  string tensor_id = 1;        // Unique identifier
  uint32 chunk_number = 2;     // Sequence number
  bytes data = 3;              // Binary chunk data
  uint32 total_chunks = 4;     // Total number of chunks
  bool is_last = 5;            // Last chunk indicator
  map<string, string> metadata = 6;  // Optional metadata
}
```

Metadata fields (included in first chunk):
- `shape`: Tensor shape as string (e.g., "[100, 100]")
- `dtype`: PyTorch dtype (e.g., "torch.float32")
- `device`: Device type (e.g., "cpu" or "cuda:0")
- `size_bytes`: Total serialized size in bytes

## Performance Considerations

1. **Chunk Size**: Default 1MB balances memory usage and network efficiency
   - Larger chunks: Fewer network round-trips, more memory usage
   - Smaller chunks: More overhead, better for memory-constrained environments

2. **Serialization**: Uses PyTorch's native `torch.save()`/`torch.load()`
   - Preserves tensor properties (device, dtype, etc.)
   - Handles complex tensor structures
   - Not optimized for maximum speed (use custom formats if needed)

3. **Async I/O**: Fully asynchronous server handles concurrent connections efficiently
   - Non-blocking I/O for network operations
   - Can handle multiple clients simultaneously

## Development

### Testing the Server

```bash
# Start the server
python -m skytorch.torch.server

# In another terminal, create a simple client script to test
# (See the client example in "Using the Client" section above)
```

For comprehensive end-to-end testing, see the test suite in the `test/` directory at the project root.

### Regenerating gRPC Code

After modifying `service.proto` or `skytorch/server/health/health.proto`:

```bash
./hack/gen-grpc-proto.sh
```

This generates:
- `service_pb2.py`: Protocol buffer messages for tensor service
- `service_pb2_grpc.py`: gRPC service stubs for tensor service
- `service_pb2.pyi`: Type hints for tensor service
