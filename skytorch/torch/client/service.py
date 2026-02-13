"""
Client for the SkyTorch tensor management gRPC service.

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
    from skytorch.torch.server import service_pb2
    from skytorch.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first."
    )

from skytorch.torch.client.tensor import get_tensor_id
from skytorch.torch.client.metadata import TensorMetadata
from skytorch.torch.client.request import (
    tensor_metadata_to_proto,
    build_copy_tensor_request,
    build_execute_aten_request,
)
from skytorch.torch.server.serialization import serialize_tensor_to_chunks, TensorAssembler
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

    async def update_tensor(
        self,
        src: torch.Tensor,
        tensor_id: int,
        tensor_metadata: Optional[TensorMetadata] = None,
    ) -> None:
        """
        Upload tensor data to server.

        If tensor_metadata is provided, the server will auto-create the tensor
        if it doesn't exist, making this a single operation.

        Args:
            src: Source cpu tensor with data to upload
            tensor_id: Destination tensor ID on the server
            tensor_metadata: Optional metadata for auto-creating the tensor

        Raises:
            RuntimeError: If update fails
        """

        async def stream_tensor():
            first_chunk = True
            for chunk in serialize_tensor_to_chunks(tensor_id, src):
                # Attach sequence number and metadata to first chunk
                if first_chunk:
                    if tensor_metadata is not None:
                        chunk.metadata.CopyFrom(
                            tensor_metadata_to_proto(tensor_metadata)
                        )
                    first_chunk = False
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
        tensor_metadata: Optional[TensorMetadata] = None,
    ) -> torch.Tensor:
        """
        Download tensor data from server storage.

        Args:
            tensor_id: Source tensor ID on the server
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            stride: Optional stride (default: contiguous)
            storage_offset: Element offset in the storage
            tensor_metadata: Optional metadata for auto-creating the tensor

        Returns:
            cpu tensor with data from server storage

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
        if tensor_metadata is not None:
            request.metadata.CopyFrom(tensor_metadata_to_proto(tensor_metadata))

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

    async def delete_tensors(self, tensor_ids: list[int]) -> None:
        """
        Delete tensors on the server.

        Args:
            tensor_ids: List of tensor IDs to delete

        Raises:
            RuntimeError: If deletion fails
        """
        request = service_pb2.DeleteTensorsRequest(tensor_ids=tensor_ids)
        response = await self.stub.DeleteTensors(request, metadata=self.metadata)

        if not response.success:
            raise RuntimeError(f"Failed to delete tensors: {response.message}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Deleted {len(tensor_ids)} tensors on server")

    async def copy_tensor(
        self,
        src_tensor_id: int,
        dst_tensor_id: int,
        src_offset: int = 0,
        dst_offset: int = 0,
        num_bytes: int = -1,
        src_metadata: Optional[TensorMetadata] = None,
        dst_metadata: Optional[TensorMetadata] = None,
    ) -> None:
        """
        Copy data between tensors on the server.

        If metadata is provided for source/destination, the server will
        auto-create those tensors if they don't exist, making this atomic.

        Args:
            src_tensor_id: Source tensor ID
            dst_tensor_id: Destination tensor ID
            src_offset: Byte offset in source storage
            dst_offset: Byte offset in destination storage
            num_bytes: Number of bytes to copy (-1 for all)
            src_metadata: Optional metadata for auto-creating source tensor
            dst_metadata: Optional metadata for auto-creating destination tensor

        Raises:
            RuntimeError: If copy fails
        """
        request = build_copy_tensor_request(
            src_tensor_id=src_tensor_id,
            dst_tensor_id=dst_tensor_id,
            src_offset=src_offset,
            dst_offset=dst_offset,
            num_bytes=num_bytes,
            src_metadata=src_metadata,
            dst_metadata=dst_metadata,
        )

        response = await self.stub.CopyTensor(request, metadata=self.metadata)

        if not response.success:
            raise RuntimeError(f"Failed to copy tensor: {response.message}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Copied tensor {src_tensor_id} -> {dst_tensor_id}")

    async def execute_function(
        self,
        callable_bytes: bytes,
        args_bytes: bytes,
        kwargs_bytes: bytes,
        callable_source: str = "",
        callable_name: str = "",
    ) -> service_pb2.ExecuteFunctionResponse:
        """
        Execute a pickled function on the server.

        Args:
            callable_bytes: cloudpickle'd callable (empty when using source)
            args_bytes: pickle'd args tuple
            kwargs_bytes: pickle'd kwargs dict
            callable_source: function source code from inspect.getsource()
            callable_name: function name (fn.__name__)

        Returns:
            ExecuteFunctionResponse with tensor metadata

        Raises:
            RuntimeError: If execution fails
        """
        request = service_pb2.ExecuteFunctionRequest(
            callable=callable_bytes,
            args=args_bytes,
            kwargs=kwargs_bytes,
            callable_source=callable_source,
            callable_name=callable_name,
        )
        return await self.stub.ExecuteFunction(request, metadata=self.metadata)

    async def execute_aten_operation(
        self,
        op_name: str,
        args: tuple,
        kwargs: dict,
        output_tensors: list[torch.Tensor] | None,
        tensor_metadata: Optional[list[TensorMetadata]] = None,
        output_metadata: Optional[list[TensorMetadata]] = None,
    ) -> list[int] | None:
        """
        Execute an ATen operation on the server.

        Supports two modes:
        - Pre-allocated outputs: output_tensors provided, writes to them, returns None
        - Server-created outputs: output_tensors is None, returns list[int] (tensor_ids)

        If metadata lists are provided, the server will auto-create those tensors
        if they don't exist, making this a single atomic operation.

        Args:
            op_name: ATen operation name (e.g., "aten::add.Tensor")
            args: Positional arguments (may contain sky tensors)
            kwargs: Keyword arguments (may contain sky tensors)
            output_tensors: Pre-allocated output tensors, or None for server-created
            tensor_metadata: Optional list of input tensor metadata for auto-creation
            output_metadata: Optional list of output tensor metadata for auto-creation

        Returns:
            None if output_tensors provided, list[int] of tensor_ids if server created outputs

        Raises:
            RuntimeError: If operation execution fails
        """
        request = build_execute_aten_request(
            op_name=op_name,
            args=args,
            kwargs=kwargs,
            output_tensors=output_tensors,
            tensor_metadata=tensor_metadata,
            output_metadata=output_metadata,
        )

        if logger.isEnabledFor(logging.DEBUG):
            input_tensor_ids = [
                get_tensor_id(arg)
                for arg in args
                if isinstance(arg, torch.Tensor) and arg.device.type == "sky"
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
