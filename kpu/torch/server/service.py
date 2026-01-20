"""
Tensor gRPC service implementation for KPU PyTorch backend.

This module implements the gRPC service for tensor management and
ATen operation execution.
"""

import logging
from typing import AsyncIterator

try:
    import grpc
    import torch
except ImportError as e:
    raise ImportError(
        f"Required dependency not found: {e}. Install with: pip install grpcio torch"
    )

try:
    from kpu.torch.server import service_pb2
    from kpu.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first.\n"
        "Make sure to install grpcio-tools: pip install grpcio-tools"
    )

from kpu.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler,
    DEFAULT_CHUNK_SIZE,
)
from kpu.torch.server.tensor import TensorManager

logger = logging.getLogger(__name__)


class TensorServicer(service_pb2_grpc.ServiceServicer):
    """
    Async gRPC servicer for tensor management and ATen operations.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize the tensor servicer.

        Args:
            chunk_size: Size of chunks for streaming tensors
        """
        self.chunk_size = chunk_size
        self.tensor_manager = TensorManager()

    async def CreateTensor(
        self,
        request: service_pb2.CreateTensorRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.TensorResponse:
        """
        Create a tensor on the server.

        Args:
            request: CreateTensorRequest with tensor metadata
            context: gRPC context

        Returns:
            TensorResponse with success status
        """
        try:
            dtype = eval(request.dtype)  # "torch.float32" -> torch.float32
            shape = list(request.shape)
            stride = list(request.stride) if request.stride else None
            storage_offset = request.storage_offset

            if request.HasField("tensor_ref"):
                # Create view from existing tensor's storage
                base_tensor = self.tensor_manager.get(request.tensor_ref)
                storage = base_tensor.untyped_storage()
                tensor = torch.empty(0, dtype=dtype, device=base_tensor.device).set_(
                    storage, storage_offset, shape, stride
                )
            else:
                # Create new tensor with fresh storage
                device = torch.device(request.device_type, request.device_index)
                storage = torch.UntypedStorage(request.nbytes, device=device)
                tensor = torch.empty(0, dtype=dtype, device=device).set_(
                    storage, storage_offset, shape, stride
                )

            self.tensor_manager.register(request.tensor_id, tensor)

            logger.info(
                f"Created tensor {request.tensor_id} "
                f"(nbytes={request.nbytes}, dtype={dtype})"
            )

            return service_pb2.TensorResponse(
                success=True,
                message=f"Created tensor {request.tensor_id}",
            )
        except Exception as e:
            logger.error(f"Failed to create tensor: {e}")
            return service_pb2.TensorResponse(
                success=False,
                message=str(e),
            )

    async def UpdateTensor(
        self,
        request_iterator: AsyncIterator[service_pb2.TensorChunk],
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.TensorResponse:
        """
        Receive tensor data and update storage.

        Args:
            request_iterator: Async iterator of tensor chunks from client
            context: gRPC context

        Returns:
            TensorResponse with success status
        """
        assembler = TensorAssembler()
        tensor_id = None

        try:
            async for chunk in request_iterator:
                if tensor_id is None:
                    tensor_id = chunk.tensor_id

                logger.debug(
                    f"Received chunk {chunk.chunk_number}/{chunk.total_chunks} "
                    f"for tensor {tensor_id}"
                )

                tensor = assembler.add_chunk(chunk)
                if tensor is None:
                    continue

                target = self.tensor_manager.get(tensor_id)
                target.copy_(tensor.to(target.device))

                logger.info(f"Updated tensor {tensor_id} with shape {tensor.shape}")

            return service_pb2.TensorResponse(
                success=True,
                message=f"Updated tensor {tensor_id}",
            )

        except Exception as e:
            logger.error(f"Error updating tensor {tensor_id}: {e}")
            return service_pb2.TensorResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    async def GetTensor(
        self,
        request: service_pb2.GetTensorRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[service_pb2.TensorChunk]:
        """
        Stream tensor data from storage.

        Args:
            request: GetStorageDataRequest with tensor ID and shape info
            context: gRPC context

        Yields:
            TensorChunk messages containing the tensor data
        """
        tensor_id = request.tensor_id
        shape = tuple(request.shape)
        dtype = eval(request.dtype)
        stride = tuple(request.stride) if request.stride else None
        offset = request.storage_offset

        try:
            tensor = self.tensor_manager.get(tensor_id)

            # Create view with requested shape/stride
            if stride:
                tensor = tensor.as_strided(shape, stride, offset)
            elif not shape:
                # Scalar tensor: access element at offset
                tensor = tensor.flatten()[offset]
            else:
                numel = torch.Size(shape).numel()
                tensor = tensor[offset : offset + numel].view(shape)

            logger.info(f"Sending tensor {tensor_id} with shape {tensor.shape}")

            # Stream the tensor data
            for chunk in serialize_tensor_to_chunks(
                tensor_id, tensor, self.chunk_size
            ):
                logger.debug(
                    f"Sending chunk {chunk.chunk_number}/{chunk.total_chunks} "
                    f"for tensor {chunk.tensor_id}"
                )
                yield chunk

        except ValueError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"Tensor {tensor_id} not found"
            )
        except Exception as e:
            logger.error(f"Error sending tensor: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error: {e}")

    async def CopyTensor(
        self,
        request: service_pb2.CopyTensorRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.TensorResponse:
        """
        Copy data between tensors on the server.

        Args:
            request: CopyTensorRequest with source and destination info
            context: gRPC context

        Returns:
            TensorResponse with success status
        """
        try:
            src_tensor = self.tensor_manager.get(request.src_tensor_id)
        except ValueError:
            return service_pb2.TensorResponse(
                success=False,
                message=f"Source tensor {request.src_tensor_id} not found",
            )

        try:
            dst_tensor = self.tensor_manager.get(request.dst_tensor_id)
        except ValueError:
            return service_pb2.TensorResponse(
                success=False,
                message=f"Destination tensor {request.dst_tensor_id} not found",
            )

        try:
            # Copy bytes
            src_bytes = src_tensor.view(torch.uint8)
            dst_bytes = dst_tensor.view(torch.uint8)
            num_bytes = (
                request.num_bytes if request.num_bytes > 0 else src_bytes.numel()
            )
            dst_bytes[request.dst_offset : request.dst_offset + num_bytes].copy_(
                src_bytes[request.src_offset : request.src_offset + num_bytes]
            )

            logger.info(
                f"Copied {num_bytes} bytes from tensor {request.src_tensor_id} "
                f"to tensor {request.dst_tensor_id}"
            )

            return service_pb2.TensorResponse(success=True)

        except Exception as e:
            logger.error(f"Error copying tensor: {e}")
            return service_pb2.TensorResponse(
                success=False,
                message=str(e),
            )

    async def ExecuteAtenOperation(
        self,
        request: service_pb2.ExecuteAtenRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.ExecuteAtenResponse:
        """
        Execute an ATen operation on server tensors.

        Supports two modes:
        - Pre-allocated outputs: request.outputs provided, writes to them
        - Server-created outputs: request.outputs empty, returns result metadata

        Args:
            request: ExecuteAtenRequest with operation name and arguments
            context: gRPC context

        Returns:
            ExecuteAtenResponse with success status and optionally output metadata
        """
        try:
            # Resolve args - replace tensor refs with actual tensors
            args = tuple(self._resolve_aten_arg(arg) for arg in request.args)
            kwargs = self._resolve_kwargs(dict(request.kwargs))

            # Get the ATen op
            op = self._get_aten_op(request.op_name)

            logger.info(
                f"Executing {request.op_name} with {len(args)} args, "
                f"{len(kwargs)} kwargs"
            )
            result = op(*args, **kwargs)

            # Normalize result to list
            if isinstance(result, torch.Tensor):
                result_tensors = [result]
            elif isinstance(result, (tuple, list)):
                result_tensors = [t for t in result if isinstance(t, torch.Tensor)]
            else:
                result_tensors = []

            if request.outputs:
                # Pre-allocated outputs mode: register results with IDs from request.outputs
                # for ref, tensor in zip(request.outputs, result_tensors):
                #     if tensor is not None:
                #         self.tensor_manager.register(ref.tensor_id, tensor)
                return service_pb2.ExecuteAtenResponse(success=True)
            else:
                # Server-created outputs mode: register and return references
                output_refs = []
                for tensor in result_tensors:
                    output_refs.append(self._tensor_to_ref(tensor))
                return service_pb2.ExecuteAtenResponse(
                    success=True,
                    output_tensors=output_refs,
                )

        except Exception as e:
            logger.error(f"Error executing ATen operation {request.op_name}: {e}")
            return service_pb2.ExecuteAtenResponse(
                success=False,
                message=str(e),
            )

    def _tensor_to_ref(self, tensor: torch.Tensor) -> service_pb2.TensorReference:
        """Convert a tensor to TensorReference proto."""
        storage_id = tensor.untyped_storage().data_ptr()
        self.tensor_manager.register(storage_id, tensor)
        return service_pb2.TensorReference(tensor_id=storage_id)

    def _resolve_aten_arg(self, arg: service_pb2.AtenArgument):
        """Resolve an AtenArgument to a Python value, replacing tensor refs."""
        which = arg.WhichOneof("value")

        if which == "tensor":
            return self.tensor_manager.get(arg.tensor.tensor_id)
        elif which == "scalar_float":
            return arg.scalar_float
        elif which == "scalar_int":
            return arg.scalar_int
        elif which == "scalar_bool":
            return arg.scalar_bool
        elif which == "scalar_string":
            return arg.scalar_string
        elif which == "none_value":
            return None
        elif which == "list_value":
            values = [self._resolve_aten_arg(v) for v in arg.list_value.values]
            if arg.list_value.is_tuple:
                return tuple(values)
            return values
        else:
            raise ValueError(f"Unknown AtenArgument type: {which}")

    def _resolve_kwargs(
        self, kwargs: dict[str, service_pb2.AtenArgument]
    ) -> dict:
        """Resolve kwargs from proto format to Python values."""
        return {key: self._resolve_aten_arg(arg) for key, arg in kwargs.items()}

    def _get_aten_op(self, op_name: str):
        """Get ATen operator by name.

        Args:
            op_name: Operation name (e.g., "aten.add.Tensor")

        Returns:
            The ATen operator callable
        """
        parts = op_name.split(".")
        op = torch.ops
        for part in parts:
            op = getattr(op, part)
        return op
