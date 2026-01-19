"""
Utilities for serializing and deserializing PyTorch tensors for streaming.
"""

from typing import Iterator, Optional

from kpu.torch.server.service_pb2 import TensorChunk

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")


# Default chunk size for streaming (1MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024


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
    # Get raw bytes from storage
    storage = tensor.untyped_storage()
    serialized_data = bytes(storage)
    total_size = len(serialized_data)

    # Calculate total number of chunks
    total_chunks = (total_size + chunk_size - 1) // chunk_size

    # Stream chunks
    offset = 0
    chunk_number = 0
    first_chunk = True

    while offset < total_size:
        end_offset = min(offset + chunk_size, total_size)
        chunk_data = serialized_data[offset:end_offset]

        chunk = TensorChunk(
            tensor_id=tensor_id,
            chunk_number=chunk_number,
            data=chunk_data,
            total_chunks=total_chunks,
        )

        # Set tensor metadata on first chunk
        if first_chunk:
            chunk.shape.extend(tensor.shape)
            chunk.stride.extend(tensor.stride())
            chunk.storage_offset = tensor.storage_offset()
            chunk.dtype = str(tensor.dtype)
            first_chunk = False

        yield chunk

        offset = end_offset
        chunk_number += 1


class TensorAssembler:
    """Assembles tensor chunks back into a complete tensor."""

    def __init__(self):
        self.chunks: dict[int, bytes] = {}
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
        # On first chunk, capture metadata
        if self.total_chunks is None:
            self.total_chunks = chunk.total_chunks
            self.shape = list(chunk.shape)
            self.stride = list(chunk.stride)
            self.storage_offset = chunk.storage_offset
            self.dtype = chunk.dtype

        self.chunks[chunk.chunk_number] = chunk.data

        if len(self.chunks) == self.total_chunks:
            return self._assemble_tensor()

        return None

    def _assemble_tensor(self) -> torch.Tensor:
        """Assemble all chunks into a complete tensor."""
        # Reconstruct serialized data in correct order
        data = b''.join(self.chunks[i] for i in sorted(self.chunks.keys()))

        # Get metadata
        dtype = eval(self.dtype)  # "torch.float32" -> torch.float32

        # Create tensor from raw bytes
        tensor = torch.frombuffer(bytearray(data), dtype=dtype)

        # Apply shape, stride, storage_offset via as_strided
        if self.stride:
            tensor = torch.as_strided(tensor, self.shape, self.stride, self.storage_offset)
        else:
            tensor = tensor.view(self.shape)

        return tensor
