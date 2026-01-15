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

from kpu.torch.client.tensor import get_tensor_metadata
from kpu.torch.server.serialization import serialize_tensor_to_chunks, TensorAssembler
from kpu.torch.client.metadata import TensorMetadata
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
            for chunk_data in serialize_tensor_to_chunks(tensor, tensor_id):
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
        args: tuple,
        kwargs: dict,
        output_tensors: list[torch.Tensor] | None,
    ) -> list[TensorMetadata] | None:
        """
        Execute an ATen operation on the server.

        Supports two modes:
        - Pre-allocated outputs: output_tensors provided, writes to them, returns None
        - Server-created outputs: output_tensors is None, returns list[TensorMetadata]

        Device mapping (KPU → remote) is handled by _to_aten_arg.

        Args:
            op_name: ATen operation name (e.g., "aten::add.Tensor")
            args: Positional arguments (may contain KPU tensors)
            kwargs: Keyword arguments (may contain KPU tensors)
            output_tensors: Pre-allocated output tensors, or None for server-created

        Returns:
            None if output_tensors provided, list[TensorMetadata] if server created outputs

        Raises:
            RuntimeError: If operation execution fails
        """
        from kpu.torch.backend._device import device_manager

        request = service_pb2.ExecuteAtenRequest(
            op_name=op_name,
            args=[self._to_aten_arg(arg) for arg in args],
        )

        for k, v in kwargs.items():
            request.kwargs[k].CopyFrom(self._to_aten_arg(v))

        # Add output references with mapped devices
        if output_tensors is not None:
            for t in output_tensors:
                if t is not None:
                    info = device_manager.get_remote_device_info(t.device.index)
                    request.outputs.append(
                        self._tensor_to_ref_with_device(
                            t, info.device_type, info.device_index
                        )
                    )

        response = await self.stub.ExecuteAtenOperation(
            request, metadata=self.metadata
        )

        if not response.success:
            raise RuntimeError(f"ATen operation failed: {response.message}")

        logger.info(f"Executed {op_name} on server")

        # Return output metadata if server created outputs
        if output_tensors is None and response.output_tensors:
            return [self._ref_to_metadata(ref) for ref in response.output_tensors]

        return None

    def _tensor_to_ref_with_device(
        self,
        tensor: torch.Tensor,
        device_type: str,
        device_index: int,
    ) -> service_pb2.TensorReference:
        """Convert a torch.Tensor to TensorReference proto with mapped device."""
        meta = get_tensor_metadata(tensor)
        meta.device_type = device_type
        meta.device_index = device_index
        return self._to_tensor_ref(meta)

    def _ref_to_metadata(self, ref: service_pb2.TensorReference) -> TensorMetadata:
        """Convert TensorReference proto to TensorMetadata."""
        return TensorMetadata(
            tensor_id=ref.tensor_id,
            shape=tuple(ref.shape),
            dtype=eval(ref.dtype),  # "torch.float32" -> torch.float32
            nbytes=ref.nbytes,
            device_type=ref.device_type,
            stride=tuple(ref.stride) if ref.stride else None,
            storage_offset=ref.storage_offset,
            device_index=ref.device_index,
        )

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
        """Convert a value to AtenArgument proto with device mapping.

        Handles:
        - None → none_value
        - torch.Tensor (KPU) → TensorReference with mapped remote device
        - torch.Tensor (CPU, scalar) → scalar value
        - TensorMetadata → TensorReference
        - bool/int/float/str → scalar values
        - torch.device (KPU) → mapped remote device string
        - torch.device/torch.dtype → string
        - list/tuple → recursive AtenArgumentList
        """
        from kpu.torch.backend._device import device_manager

        arg = service_pb2.AtenArgument()

        if value is None:
            arg.none_value = True
        elif isinstance(value, torch.Tensor):
            if value.device.type == "kpu":
                # KPU tensor → get metadata and map device
                info = device_manager.get_remote_device_info(value.device.index)
                meta = get_tensor_metadata(value)
                meta.device_type = info.device_type
                meta.device_index = info.device_index
                arg.tensor.CopyFrom(self._to_tensor_ref(meta))
            elif value.device.type == "cpu" and value.dim() == 0:
                # CPU scalar tensor → convert to Python scalar
                scalar = value.item()
                if isinstance(scalar, bool):
                    arg.scalar_bool = scalar
                elif isinstance(scalar, int):
                    arg.scalar_int = scalar
                elif isinstance(scalar, float):
                    arg.scalar_float = scalar
                else:
                    arg.scalar_string = str(scalar)
            else:
                raise ValueError(
                    f"Unsupported tensor device: {value.device.type}. "
                    f"Only KPU tensors and 0-dim CPU scalars are allowed."
                )
        elif isinstance(value, TensorMetadata):
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
        elif isinstance(value, torch.device):
            if value.type == "kpu":
                # Map KPU device to remote device
                info = device_manager.get_remote_device_info(value.index or 0)
                arg.scalar_string = str(
                    torch.device(info.device_type, info.device_index)
                )
            else:
                arg.scalar_string = str(value)
        elif isinstance(value, torch.dtype):
            arg.scalar_string = str(value)
        elif isinstance(value, (list, tuple)):
            # Handle nested lists/tuples recursively
            list_arg = service_pb2.AtenArgumentList()
            list_arg.is_tuple = isinstance(value, tuple)
            for item in value:
                list_arg.values.append(self._to_aten_arg(item))
            arg.list_value.CopyFrom(list_arg)
        else:
            raise ValueError(f"Unsupported ATen argument type: {type(value)}")

        return arg
