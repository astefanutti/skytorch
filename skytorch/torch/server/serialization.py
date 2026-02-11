"""
Utilities for serializing and deserializing PyTorch tensors for streaming.
"""

import ctypes
from typing import Iterator, Optional

from skytorch.torch.server.service_pb2 import TensorChunk

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")


# Dtype string to torch.dtype lookup for fast parsing (replaces eval())
_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.int16": torch.int16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
    "torch.complex64": torch.complex64,
    "torch.complex128": torch.complex128,
}


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    if dtype_str in _DTYPE_MAP:
        return _DTYPE_MAP[dtype_str]
    # Fallback for unknown dtypes
    return eval(dtype_str)


# dtypes that numpy doesn't support
_NUMPY_UNSUPPORTED_DTYPES = {torch.bfloat16}


# Default chunk size for streaming (1MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024


def tensor_from_bytes(data: bytes, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    """Create tensor from bytes, handling numpy-unsupported dtypes like bfloat16."""
    # torch.frombuffer fails with 0-byte buffers, so handle empty tensors directly
    if len(data) == 0:
        return torch.empty(shape, dtype=dtype)
    if dtype in _NUMPY_UNSUPPORTED_DTYPES:
        # Use UntypedStorage for dtypes numpy doesn't support
        storage = torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
        tensor = torch.empty(shape, dtype=dtype)
        tensor.untyped_storage().copy_(storage)
        return tensor
    else:
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes, handling numpy-unsupported dtypes like bfloat16."""
    tensor_contig = tensor.detach().contiguous().cpu()
    if tensor.dtype in _NUMPY_UNSUPPORTED_DTYPES:
        storage = tensor_contig.untyped_storage()
        return ctypes.string_at(storage.data_ptr(), storage.nbytes())
    else:
        return tensor_contig.numpy().tobytes()


def serialize_tensor_to_chunks(
    tensor_id: int,
    tensor: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[TensorChunk]:
    """
    Serialize a PyTorch tensor into chunks for streaming.

    Uses raw storage bytes for efficient serialization.

    Args:
        tensor_id: Unique identifier for the tensor
        tensor: PyTorch tensor to serialize
        chunk_size: Size of each chunk in bytes

    Yields:
        TensorChunk proto messages
    """
    # Make tensor contiguous and serialize efficiently
    # Detach is needed for tensors that require grad (e.g., nn.Parameter)
    tensor_contig = tensor.detach().contiguous()

    if tensor.dtype in _NUMPY_UNSUPPORTED_DTYPES:
        # Use ctypes for dtypes numpy doesn't support (e.g., bfloat16)
        storage = tensor_contig.untyped_storage()
        serialized_data = ctypes.string_at(storage.data_ptr(), storage.nbytes())
    else:
        # Use numpy for fast memcpy-based serialization
        serialized_data = tensor_contig.numpy().tobytes()

    total_size = len(serialized_data)

    # Calculate total number of chunks (at least 1 for empty tensors)
    total_chunks = max(1, (total_size + chunk_size - 1) // chunk_size)

    # Handle empty tensors: yield a single metadata-only chunk
    if total_size == 0:
        chunk = TensorChunk(
            tensor_id=tensor_id,
            chunk_number=0,
            data=b"",
            total_chunks=1,
            total_bytes=0,
            dtype=str(tensor_contig.dtype),
        )
        chunk.shape.extend(tensor_contig.shape)
        chunk.stride.extend(tensor_contig.stride())
        chunk.storage_offset = tensor_contig.storage_offset()
        yield chunk
        return

    # Use memoryview for zero-copy slicing
    mv = memoryview(serialized_data)

    # Stream chunks
    offset = 0
    chunk_number = 0
    first_chunk = True

    while offset < total_size:
        end_offset = min(offset + chunk_size, total_size)
        # Zero-copy slice via memoryview, then copy to bytes for protobuf
        chunk_data = bytes(mv[offset:end_offset])

        chunk = TensorChunk(
            tensor_id=tensor_id,
            chunk_number=chunk_number,
            data=chunk_data,
            total_chunks=total_chunks,
        )

        # Set tensor metadata on first chunk
        if first_chunk:
            chunk.total_bytes = total_size
            chunk.shape.extend(tensor_contig.shape)
            chunk.stride.extend(tensor_contig.stride())
            chunk.storage_offset = tensor_contig.storage_offset()
            chunk.dtype = str(tensor_contig.dtype)
            first_chunk = False

        yield chunk

        offset = end_offset
        chunk_number += 1


class TensorAssembler:
    """Assembles tensor chunks back into a complete tensor."""

    def __init__(self):
        self._buffer: bytearray | None = None
        self._chunks_received: int = 0
        self._chunk_size: int = 0
        self.total_chunks: int | None = None
        self.shape: list[int] | None = None
        self.stride: list[int] | None = None
        self.storage_offset: int = 0
        self.dtype: str | None = None

    def add_chunk(self, chunk: TensorChunk) -> Optional[torch.Tensor]:
        """
        Add a chunk to the assembler.

        Args:
            chunk: TensorChunk proto message

        Returns:
            Complete tensor if all chunks received, None otherwise
        """
        if self._buffer is None:
            # Pre-allocate buffer on first chunk
            self._buffer = bytearray(chunk.total_bytes)
            self._chunk_size = len(chunk.data)
            self.total_chunks = chunk.total_chunks
            self.shape = list(chunk.shape)
            self.stride = list(chunk.stride)
            self.storage_offset = chunk.storage_offset
            self.dtype = chunk.dtype

        # Write directly into buffer at correct offset
        start = chunk.chunk_number * self._chunk_size
        end = start + len(chunk.data)
        self._buffer[start:end] = chunk.data

        self._chunks_received += 1
        if self._chunks_received == self.total_chunks:
            return self._assemble_tensor()

        return None

    def _assemble_tensor(self) -> torch.Tensor:
        """Assemble all chunks into a complete tensor."""
        # Get metadata
        dtype = parse_dtype(self.dtype)

        # torch.frombuffer fails with 0-byte buffers, so handle empty tensors directly
        if len(self._buffer) == 0:
            return torch.empty(self.shape, dtype=dtype)

        # Zero-copy tensor from pre-allocated buffer
        tensor = torch.frombuffer(self._buffer, dtype=dtype)

        # Apply shape, stride, storage_offset via as_strided
        # Note: Use `is not None` instead of truthiness check because empty
        # stride [] is valid for scalar tensors
        if self.stride is not None:
            tensor = torch.as_strided(tensor, self.shape, self.stride, self.storage_offset)
        else:
            tensor = tensor.view(self.shape)

        return tensor
