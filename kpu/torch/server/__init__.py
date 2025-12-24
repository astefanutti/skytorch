"""
PyTorch Tensor Streaming gRPC Server

This package provides an async gRPC server for streaming PyTorch tensors
using best practices for serialization and streaming.
"""

from kpu.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler,
    deserialize_tensor_from_bytes,
    DEFAULT_CHUNK_SIZE
)

try:
    from kpu.torch.server.server import (
        TensorServicer,
        serve
    )
except ImportError:
    # Generated gRPC code not available yet
    TensorServicer = None
    serve = None


__all__ = [
    'serialize_tensor_to_chunks',
    'TensorAssembler',
    'deserialize_tensor_from_bytes',
    'DEFAULT_CHUNK_SIZE',
    'TensorServicer',
    'serve'
]
