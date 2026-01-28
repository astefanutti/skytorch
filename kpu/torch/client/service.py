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

from kpu.torch.client.tensor import get_tensor_id
from kpu.torch.client.metadata import TensorMetadata
from kpu.torch.server.serialization import serialize_tensor_to_chunks, TensorAssembler
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

    async def create_tensor(
        self, metadata: TensorMetadata, tensor_ref: Optional[int] = None
    ) -> None:
        """
        Create a tensor on the server.

        Args:
            metadata: TensorMetadata with tensor configuration
            tensor_ref: Optional reference to base tensor for view creation

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
        if tensor_ref is not None:
            request.tensor_ref = tensor_ref

        response = await self.stub.CreateTensor(request, metadata=self.metadata)

        if not response.success:
            raise RuntimeError(f"Failed to create tensor: {response.message}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Created tensor {metadata.tensor_id} on server")

    async def update_tensor(
        self,
        src: torch.Tensor,
        tensor_id: int,
    ) -> None:
        """
        Upload tensor data to server storage.

        Args:
            src: Source CPU tensor with data to upload
            tensor_id: Destination tensor ID on the server

        Raises:
            RuntimeError: If update fails
        """

        async def stream_tensor():
            for chunk in serialize_tensor_to_chunks(tensor_id, src):
                yield chunk

        response = await self.stub.UpdateTensor(
            stream_tensor(), metadata=self.metadata
        )

        if not response.success:
            raise RuntimeError(f"Failed to update tensor: {response.message}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Updated tensor {tensor_id} on server")

    async def get_tensor(
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
        request = service_pb2.GetTensorRequest(
            tensor_id=tensor_id,
            shape=list(shape),
            dtype=str(dtype),
            stride=list(stride) if stride else [],
            storage_offset=storage_offset,
        )

        assembler = TensorAssembler()

        async for chunk in self.stub.GetTensor(request, metadata=self.metadata):
            tensor = assembler.add_chunk(chunk)
            if tensor is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Got tensor tensor_id={tensor_id} shape={tensor.shape} "
                        f"data={tensor.flatten()[:8].tolist()}... from server"
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
    ) -> None:
        """
        Copy data between tensors on the server.

        Args:
            src_tensor_id: Source tensor ID
            dst_tensor_id: Destination tensor ID
            src_offset: Byte offset in source storage
            dst_offset: Byte offset in destination storage
            num_bytes: Number of bytes to copy (-1 for all)

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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Copied tensor {src_tensor_id} -> {dst_tensor_id}")

    async def execute_aten_operation(
        self,
        op_name: str,
        args: tuple,
        kwargs: dict,
        output_tensors: list[torch.Tensor] | None,
    ) -> list[int] | None:
        """
        Execute an ATen operation on the server.

        Supports two modes:
        - Pre-allocated outputs: output_tensors provided, writes to them, returns None
        - Server-created outputs: output_tensors is None, returns list[int] (tensor_ids)

        Args:
            op_name: ATen operation name (e.g., "aten::add.Tensor")
            args: Positional arguments (may contain KPU tensors)
            kwargs: Keyword arguments (may contain KPU tensors)
            output_tensors: Pre-allocated output tensors, or None for server-created

        Returns:
            None if output_tensors provided, list[int] of tensor_ids if server created outputs

        Raises:
            RuntimeError: If operation execution fails
        """
        request = service_pb2.ExecuteAtenRequest(
            op_name=op_name,
            args=[self._to_aten_arg(arg) for arg in args],
        )

        for k, v in kwargs.items():
            request.kwargs[k].CopyFrom(self._to_aten_arg(v))

        if output_tensors is not None:
            for t in output_tensors:
                if t is not None:
                    request.outputs.append(
                        service_pb2.TensorReference(tensor_id=get_tensor_id(t))
                    )

        if logger.isEnabledFor(logging.DEBUG):
            input_tensor_ids = [
                get_tensor_id(arg)
                for arg in args
                if isinstance(arg, torch.Tensor) and arg.device.type == "kpu"
            ]
            output_tensor_ids = [
                get_tensor_id(t) for t in (output_tensors or []) if t is not None
            ]
            logger.debug(
                f"Executing {op_name} | "
                f"inputs={input_tensor_ids} | outputs={output_tensor_ids}"
            )

        response = await self.stub.ExecuteAtenOperation(
            request, metadata=self.metadata
        )

        if not response.success:
            raise RuntimeError(f"ATen operation failed: {response.message}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Executed {op_name}")

        if output_tensors is None and response.output_tensors:
            return [ref.tensor_id for ref in response.output_tensors]

        return None

    def _to_aten_arg(self, value) -> service_pb2.AtenArgument:
        """Convert a value to AtenArgument proto.

        Handles:
        - None → none_value
        - torch.Tensor (KPU) → TensorReference with tensor_id
        - torch.Tensor (CPU, scalar) → scalar value
        - bool/int/float/str → scalar values
        - torch.device → string
        - torch.dtype → scalar_dtype
        - list/tuple → recursive AtenArgumentList
        """
        arg = service_pb2.AtenArgument()

        if value is None:
            arg.none_value = True
        elif isinstance(value, torch.Tensor):
            if value.device.type == "kpu":
                arg.tensor.CopyFrom(
                    service_pb2.TensorReference(tensor_id=get_tensor_id(value))
                )
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
            arg.scalar_string = str(value)
        elif isinstance(value, torch.dtype):
            arg.scalar_dtype = str(value)
        elif isinstance(value, torch.memory_format):
            arg.scalar_memory_format = str(value)
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
