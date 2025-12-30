"""
PyTorch Streaming gRPC Server

This package provides an async gRPC server for streaming PyTorch tensors
using best practices for serialization and streaming.
"""

from kpu.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler,
    deserialize_tensor_from_bytes,
    DEFAULT_CHUNK_SIZE
)

from kpu.torch.server.service import TensorServicer
from kpu.torch.server.server import serve

__all__ = [
    'serialize_tensor_to_chunks',
    'TensorAssembler',
    'deserialize_tensor_from_bytes',
    'DEFAULT_CHUNK_SIZE',
    'TensorServicer',
    'serve'
]
