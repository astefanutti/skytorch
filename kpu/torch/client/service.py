"""
Client for the KPU PyTorch tensor management gRPC service.

This module provides the TensorClient class with methods for tensor lifecycle
management and ATen operation execution on the remote server.
"""

import logging
from typing import Optional

try:
    import grpc
    import torch
except ImportError as e:
    raise ImportError(f"Required dependency not found: {e}")

try:
    from kpu.torch.server import service_pb2
    from kpu.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first."
    )

from kpu.torch.server.serialization import serialize_tensor_to_chunks, TensorAssembler
from kpu.torch.common.metadata import TensorMetadata
from grpc.aio._typing import MetadataType

logger = logging.getLogger(__name__)


class TensorClient:
    """
    Async gRPC client for tensor management and ATen operation execution.

    Provides methods for:
    - Creating tensors on the server
    - Uploading/downloading tensor data
    - Server-side tensor copy
    - Remote ATen operation execution

    Uses a shared gRPC channel provided by the caller (typically GRPCClient).
    """

    def __init__(
        self, channel: grpc.aio.Channel, metadata: Optional[MetadataType] = None
    ):
        """
        Initialize the client.

        Args:
            channel: gRPC channel to use for communication
            metadata: Optional metadata to include in requests
        """
        self.channel = channel
        self.metadata = metadata
        self.stub = service_pb2_grpc.ServiceStub(self.channel)

    async def create_tensor(self, metadata: TensorMetadata) -> bool:
        """
        Create a tensor on the server.

        Args:
            metadata: TensorMetadata with tensor configuration

        Returns:
            True if successful

        Raises:
            RuntimeError: If tensor creation fails
        """
        request = service_pb2.CreateTensorRequest(
            tensor_id=metadata.tensor_id,
            shape=list(metadata.shape),
            dtype=str(metadata.dtype),
            nbytes=metadata.nbytes,
            device_type=metadata.device_type,
            stride=list(metadata.stride) if metadata.stride else [],
            storage_offset=metadata.storage_offset,
            device_index=metadata.device_index,
        )

        response = await self.stub.CreateTensor(request, metadata=self.metadata)

        if not response.success:
            raise RuntimeError(f"Failed to create tensor: {response.message}")

        logger.info(f"Created tensor {metadata.tensor_id} on server")
        return True

    async def update_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: int,
        storage_offset: int = 0,
    ) -> bool:
        """
        Upload tensor data to server storage.

        The server will auto-create storage if it doesn't exist (implicit creation).

        Args:
            tensor: CPU tensor to upload
            tensor_id: Target tensor ID on the server
            storage_offset: Element offset in the storage

        Returns:
            True if successful
        """

        async def tensor_generator():
            logger.debug(
                f"Uploading tensor to {tensor_id} "
                f"(shape={tensor.shape}, offset={storage_offset})"
            )
            for chunk_data in serialize_tensor_to_chunks(tensor):
                t_id, c_num, data, total, is_last, meta = chunk_data

                # Add storage targeting metadata
                if meta is None:
                    meta = {}
                meta["target_tensor_id"] = str(tensor_id)
                meta["target_storage_offset"] = str(storage_offset)

                chunk = service_pb2.TensorChunk(
                    tensor_id=t_id,
                    chunk_number=c_num,
                    data=data,
                    total_chunks=total,
                    is_last=is_last,
                )
                chunk.metadata.update(meta)
                yield chunk

        response = await self.stub.UpdateTensor(
            tensor_generator(), metadata=self.metadata
        )

        if not response.success:
            raise RuntimeError(f"Failed to update tensor: {response.message}")

        logger.info(f"Updated tensor {tensor_id} on server")
        return True

    async def get_storage_data(
        self,
        tensor_id: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        stride: Optional[tuple[int, ...]] = None,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """
        Download tensor data from server storage.

        Args:
            tensor_id: Source tensor ID on the server
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            stride: Optional stride (default: contiguous)
            storage_offset: Element offset in the storage

        Returns:
            CPU tensor with data from server storage

        Raises:
            RuntimeError: If tensor retrieval fails
        """
        request = service_pb2.GetStorageDataRequest(
            tensor_id=tensor_id,
            shape=list(shape),
            dtype=str(dtype),
            stride=list(stride) if stride else [],
            storage_offset=storage_offset,
        )

        logger.debug(
            f"Downloading tensor {tensor_id} (shape={shape}, dtype={dtype})"
        )

        assembler = TensorAssembler()

        async for chunk in self.stub.GetStorageData(request, metadata=self.metadata):
            logger.debug(
                f"Received chunk {chunk.chunk_number}/{chunk.total_chunks} "
                f"for tensor {chunk.tensor_id}"
            )

            chunk_metadata = dict(chunk.metadata) if chunk.metadata else None

            tensor = assembler.add_chunk(
                tensor_id=chunk.tensor_id,
                chunk_number=chunk.chunk_number,
                data=chunk.data,
                total_chunks=chunk.total_chunks,
                is_last=chunk.is_last,
                metadata=chunk_metadata,
            )

            if tensor is not None:
                logger.info(
                    f"Downloaded tensor {tensor_id} with shape {tensor.shape}"
                )
                return tensor

        raise RuntimeError(f"Failed to receive tensor from storage {tensor_id}")

    async def copy_tensor(
        self,
        src_tensor_id: int,
        dst_tensor_id: int,
        src_offset: int = 0,
        dst_offset: int = 0,
        num_bytes: int = -1,
    ) -> bool:
        """
        Copy data between tensors on the server.

        Args:
            src_tensor_id: Source tensor ID
            dst_tensor_id: Destination tensor ID
            src_offset: Byte offset in source storage
            dst_offset: Byte offset in destination storage
            num_bytes: Number of bytes to copy (-1 for all)

        Returns:
            True if successful

        Raises:
            RuntimeError: If copy fails
        """
        request = service_pb2.CopyTensorRequest(
            src_tensor_id=src_tensor_id,
            dst_tensor_id=dst_tensor_id,
            src_offset=src_offset,
            dst_offset=dst_offset,
            num_bytes=num_bytes,
        )

        response = await self.stub.CopyTensor(request, metadata=self.metadata)

        if not response.success:
            raise RuntimeError(f"Failed to copy tensor: {response.message}")

        logger.info(f"Copied tensor {src_tensor_id} -> {dst_tensor_id}")
        return True

    async def execute_aten_operation(
        self,
        op_name: str,
        input_refs: list[TensorMetadata],
        output_refs: list[TensorMetadata],
        kwargs: Optional[dict] = None,
    ) -> bool:
        """
        Execute an ATen operation on the server.

        Args:
            op_name: ATen operation name (e.g., "aten::add.Tensor")
            input_refs: List of input tensor metadata
            output_refs: List of output tensor metadata
            kwargs: Optional keyword arguments for the operation

        Returns:
            True if successful

        Raises:
            RuntimeError: If operation execution fails
        """
        request = service_pb2.ExecuteAtenRequest(
            op_name=op_name,
            inputs=[self._to_aten_arg(m) for m in input_refs],
            outputs=[self._to_tensor_ref(m) for m in output_refs],
        )

        if kwargs:
            for k, v in kwargs.items():
                request.kwargs[k].CopyFrom(self._to_aten_arg(v))

        response = await self.stub.ExecuteAtenOperation(
            request, metadata=self.metadata
        )

        if not response.success:
            raise RuntimeError(f"ATen operation failed: {response.message}")

        logger.info(f"Executed {op_name} on server")
        return True

    def _to_tensor_ref(
        self, meta: TensorMetadata
    ) -> service_pb2.TensorReference:
        """Convert TensorMetadata to TensorReference proto."""
        return service_pb2.TensorReference(
            tensor_id=meta.tensor_id,
            shape=list(meta.shape),
            dtype=str(meta.dtype),
            nbytes=meta.nbytes,
            device_type=meta.device_type,
            stride=list(meta.stride) if meta.stride else [],
            storage_offset=meta.storage_offset,
            device_index=meta.device_index,
        )

    def _to_aten_arg(self, value) -> service_pb2.AtenArgument:
        """Convert a value to AtenArgument proto."""
        arg = service_pb2.AtenArgument()

        if isinstance(value, TensorMetadata):
            arg.tensor.CopyFrom(self._to_tensor_ref(value))
        elif isinstance(value, bool):
            # Must check bool before int since bool is subclass of int
            arg.scalar_bool = value
        elif isinstance(value, int):
            arg.scalar_int = value
        elif isinstance(value, float):
            arg.scalar_float = value
        elif isinstance(value, str):
            arg.scalar_string = value
        elif isinstance(value, (list, tuple)):
            # Handle lists of scalars
            if all(isinstance(x, float) for x in value):
                arg.list_value.float_values.extend(value)
            elif all(isinstance(x, bool) for x in value):
                arg.list_value.bool_values.extend(value)
            elif all(isinstance(x, int) for x in value):
                arg.list_value.int_values.extend(value)
            else:
                raise ValueError(f"Unsupported list type in ATen argument: {value}")
        else:
            raise ValueError(f"Unsupported ATen argument type: {type(value)}")

        return arg
