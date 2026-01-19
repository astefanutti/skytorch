"""
KPU PyTorch Server.

This package provides:
- serve: Async gRPC server for tensor operations
- Compute: Lightweight compute for local testing without Kubernetes
- compute: Decorator for automatic Compute lifecycle management
"""

from kpu.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler,
    DEFAULT_CHUNK_SIZE
)

from kpu.torch.server.service import TensorServicer
from kpu.torch.server.server import serve
from kpu.torch.server.compute import Compute, compute

__all__ = ["serve", "Compute", "compute"]
