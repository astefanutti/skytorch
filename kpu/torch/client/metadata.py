"""
TensorMetadata - Metadata class for KPU tensors.

This module defines the TensorMetadata dataclass used for creating and
referencing tensors on the server.
"""

from dataclasses import dataclass
from typing import Optional, Self

import torch


@dataclass
class TensorMetadata:
    """Metadata for creating/referencing a tensor on the server."""

    tensor_id: int  # Unique identifier (metadata hash from C++ extension)
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int  # Total storage size in bytes
    device_type: str  # Device type (e.g., "kpu", "cuda", "cpu")
    stride: Optional[tuple[int, ...]] = None
    storage_offset: int = 0
    device_index: int = 0

    def to_proto_dict(self) -> dict[str, str]:
        """Convert to proto-compatible string dict."""
        return {
            "tensor_id": str(self.tensor_id),
            "shape": str(list(self.shape)),
            "dtype": str(self.dtype),
            "nbytes": str(self.nbytes),
            "device_type": self.device_type,
            "stride": str(list(self.stride)) if self.stride else "",
            "storage_offset": str(self.storage_offset),
            "device_index": str(self.device_index),
        }

    @classmethod
    def from_proto_dict(cls, d: dict[str, str]) -> Self:
        """Create from proto string dict."""
        import ast

        return cls(
            tensor_id=int(d["tensor_id"]),
            shape=tuple(ast.literal_eval(d["shape"])),
            dtype=eval(d["dtype"]),  # e.g., "torch.float32"
            nbytes=int(d["nbytes"]),
            device_type=d["device_type"],
            stride=tuple(ast.literal_eval(d["stride"])) if d.get("stride") else None,
            storage_offset=int(d.get("storage_offset", "0")),
            device_index=int(d.get("device_index", "0")),
        )
