"""
SkyTorch Server.

This package provides:
- serve: Async gRPC server for tensor operations
- Compute: Lightweight compute for local testing without Kubernetes
- compute: Decorator for automatic Compute lifecycle management
"""

from skytorch.torch.server.service import TensorServicer
from skytorch.torch.server.server import serve
from skytorch.torch.server.compute import Compute, compute

__all__ = ["serve", "Compute", "compute"]
