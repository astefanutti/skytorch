"""
Utilities for serializing and deserializing PyTorch tensors for streaming.

This module provides stream-based serialization following PyTorch best practices.
"""

import io
from typing import Iterator, Optional

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")


# Default chunk size for streaming (1MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024


def serialize_tensor_to_chunks(
    tensor: torch.Tensor,
    tensor_id: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    metadata: Optional[dict[str, str]] = None,
) -> Iterator[tuple[int, int, bytes, int, bool, Optional[dict[str, str]]]]:
    """
    Serialize a PyTorch tensor into chunks for streaming.

    Uses torch.save() with BytesIO to follow PyTorch best practices.

    Args:
        tensor: PyTorch tensor to serialize
        chunk_size: Size of each chunk in bytes
        tensor_id: Unique identifier for the tensor
        metadata: Optional metadata to include with the tensor

    Yields:
        Tuples of (tensor_id, chunk_number, data, total_chunks, is_last, metadata)
    """
    # Serialize tensor to bytes using PyTorch's save method
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)

    # Read the serialized data
    serialized_data = buffer.read()
    total_size = len(serialized_data)

    # Calculate total number of chunks
    total_chunks = (total_size + chunk_size - 1) // chunk_size

    # Prepare metadata (include tensor info in first chunk)
    if metadata is None:
        metadata = {}

    first_chunk_metadata = {
        **metadata,
        "shape": str(list(tensor.shape)),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "size_bytes": str(total_size)
    }

    # Stream chunks
    offset = 0
    chunk_number = 0

    while offset < total_size:
        end_offset = min(offset + chunk_size, total_size)
        chunk_data = serialized_data[offset:end_offset]
        is_last = (end_offset >= total_size)

        # Include metadata only in first chunk
        chunk_metadata = first_chunk_metadata if chunk_number == 0 else None

        yield tensor_id, chunk_number, chunk_data, total_chunks, is_last, chunk_metadata

        offset = end_offset
        chunk_number += 1


class TensorAssembler:
    """
    Assembles tensor chunks back into complete tensors.
    """

    def __init__(self):
        self.buffers: dict[int, dict] = {}

    def add_chunk(
        self,
        tensor_id: int,
        chunk_number: int,
        data: bytes,
        total_chunks: int,
        is_last: bool,
        metadata: Optional[dict[str, str]] = None
    ) -> Optional[torch.Tensor]:
        """
        Add a chunk to the assembler.

        Args:
            tensor_id: Unique identifier for the tensor
            chunk_number: Sequence number of this chunk
            data: Chunk data
            total_chunks: Total number of chunks expected
            is_last: Whether this is the last chunk
            metadata: Optional metadata (typically only in first chunk)

        Returns:
            Complete tensor if all chunks received, None otherwise
        """
        if tensor_id not in self.buffers:
            self.buffers[tensor_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'metadata': metadata or {}
            }

        # Store metadata from first chunk
        if metadata:
            self.buffers[tensor_id]['metadata'].update(metadata)

        # Add chunk
        self.buffers[tensor_id]['chunks'][chunk_number] = data

        # Check if all chunks received
        if len(self.buffers[tensor_id]['chunks']) == total_chunks:
            return self._assemble_tensor(tensor_id)

        return None

    def _assemble_tensor(self, tensor_id: int) -> torch.Tensor:
        """
        Assemble all chunks into a complete tensor.

        Args:
            tensor_id: Unique identifier for the tensor

        Returns:
            Reconstructed PyTorch tensor
        """
        buffer_info = self.buffers[tensor_id]
        chunks = buffer_info['chunks']

        # Reconstruct serialized data in correct order
        serialized_data = b''.join(
            chunks[i] for i in sorted(chunks.keys())
        )

        # Deserialize using PyTorch
        buffer = io.BytesIO(serialized_data)
        tensor = torch.load(buffer, weights_only=False)

        # Clean up
        del self.buffers[tensor_id]

        return tensor

    def get_metadata(self, tensor_id: int) -> Optional[dict[str, str]]:
        """
        Get metadata for a tensor being assembled.

        Args:
            tensor_id: Unique identifier for the tensor

        Returns:
            Metadata dictionary if available, None otherwise
        """
        if tensor_id in self.buffers:
            return self.buffers[tensor_id]['metadata']
        return None


def deserialize_tensor_from_bytes(data: bytes) -> torch.Tensor:
    """
    Deserialize a complete tensor from bytes.

    Args:
        data: Serialized tensor data

    Returns:
        Reconstructed PyTorch tensor
    """
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=False)
