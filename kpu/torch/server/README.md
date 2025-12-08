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

## Quick Start

### Starting the Server

```python
from kpu.torch.server import serve
import asyncio

# Start server on default port (50051)
asyncio.run(serve())

# Or customize port and chunk size
asyncio.run(serve(port=8080, chunk_size=512*1024))
```

Or run directly:
```bash
python -m kpu.torch.server
```

### Using the Client

See `example_client.py` for complete examples:

```python
from kpu.torch.client.example_client import TensorClient
import torch
import asyncio


async def main():
    tensor = torch.randn(100, 100)

    async with TensorClient() as client:
        # Send tensors to server
        response = await client.send_tensors(tensor)
        print(f"Server received: {response.message}")

        # Receive tensors from server
        tensors = await client.receive_tensors(count=2)
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

## Customization

### Custom Tensor Processing

Override the `_process_tensor()` method in `TensorServicer`:

```python
from kpu.torch.server.server import TensorServicer
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

```python
# Server
asyncio.run(serve(chunk_size=2*1024*1024))  # 2MB chunks

# Serialization
from kpu.torch.server import serialize_tensor_to_chunks
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

### Running Tests

```bash
# Start the server in one terminal
python -m kpu.torch.server

# Run the example client in another terminal
python -m kpu.torch.client.example_client
```

### Regenerating gRPC Code

After modifying `service.proto`:

```bash
./generate_proto.sh
```

This generates:
- `service_pb2.py`: Protocol buffer messages
- `service_pb2_grpc.py`: gRPC service stubs
- `service_pb2.pyi`: Type hints

## License

This is part of the KPU project.
