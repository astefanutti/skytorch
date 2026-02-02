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

    def _ensure_tensor_exists(
        self, metadata: service_pb2.TensorMetadata
    ) -> torch.Tensor:
        """
        Create tensor from metadata if it doesn't exist, or return existing.

        If the tensor already exists in the tensor_manager, returns it.
        Otherwise, creates a new tensor from the metadata (either a view
        if tensor_ref is set, or a fresh storage tensor).

        Args:
            metadata: TensorMetadata proto with tensor configuration

        Returns:
            The existing or newly created tensor
        """
        tensor_id = metadata.tensor_id
        if tensor_id in self.tensor_manager:
            return self.tensor_manager.get(tensor_id)

        dtype = eval(metadata.dtype)  # "torch.float32" -> torch.float32
        shape = list(metadata.shape)
        stride = list(metadata.stride) if metadata.stride else None
        storage_offset = metadata.storage_offset

        if metadata.HasField("tensor_ref"):
            # Create view from existing tensor's storage
            base_tensor = self.tensor_manager.get(metadata.tensor_ref)
            storage = base_tensor.untyped_storage()
            tensor = torch.empty(0, dtype=dtype, device=base_tensor.device).set_(
                storage, storage_offset, shape, stride
            )
        else:
            # Create new tensor with fresh storage
            device = torch.device(metadata.device_type, metadata.device_index)
            storage = torch.UntypedStorage(metadata.nbytes, device=device)
            tensor = torch.empty(0, dtype=dtype, device=device).set_(
                storage, storage_offset, shape, stride
            )

        self.tensor_manager.register(tensor_id, tensor)

        if logger.isEnabledFor(logging.DEBUG):
            if metadata.HasField("tensor_ref"):
                logger.debug(
                    f"Auto-created tensor {tensor_id} "
                    f"(view of {metadata.tensor_ref}, shape={shape}, dtype={dtype})"
                )
            else:
                logger.debug(
                    f"Auto-created tensor {tensor_id} "
                    f"(nbytes={metadata.nbytes}, dtype={dtype})"
                )

        return tensor

    async def UpdateTensor(
        self,
        request_iterator: AsyncIterator[service_pb2.TensorChunk],
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.TensorResponse:
        """
        Update tensor data and storage.

        Args:
            request_iterator: Async iterator of tensor chunks from client
            context: gRPC context

        Returns:
            TensorResponse with success status
        """
        assembler = TensorAssembler()
        tensor_id = None
        metadata_processed = False

        try:
            async for chunk in request_iterator:
                if tensor_id is None:
                    tensor_id = chunk.tensor_id

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Updating tensor {tensor_id}")

                # Auto-create tensor from metadata on first chunk if provided
                if not metadata_processed and chunk.HasField("metadata"):
                    self._ensure_tensor_exists(chunk.metadata)
                    metadata_processed = True

                tensor = assembler.add_chunk(chunk)
                if tensor is None:
                    continue

                target = self.tensor_manager.get(tensor_id)
                target.copy_(tensor.to(target.device))

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Updated tensor {tensor_id} with shape {tensor.shape}")

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
        Get tensor data from storage.

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
        except ValueError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"Tensor {tensor_id} not found"
            )

        try:
            # Stream the tensor data
            for chunk in serialize_tensor_to_chunks(
                tensor_id, tensor.cpu().detach(), self.chunk_size
            ):
                yield chunk

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Sent tensor tensor_id={tensor_id} shape={tensor.shape} "
                    f"data={tensor.cpu().flatten()[:8].tolist()}..."
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

        Auto-creates source and/or destination tensors from metadata if provided.

        Args:
            request: CopyTensorRequest with source and destination info
            context: gRPC context

        Returns:
            TensorResponse with success status
        """
        try:
            # Auto-create tensors from metadata if provided
            if request.HasField("src_metadata"):
                self._ensure_tensor_exists(request.src_metadata)
            if request.HasField("dst_metadata"):
                self._ensure_tensor_exists(request.dst_metadata)

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
            dst_tensor.copy_(src_tensor)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Copied tensor {request.src_tensor_id} "
                    f"to tensor {request.dst_tensor_id}"
                )

            return service_pb2.TensorResponse(success=True)

        except Exception as e:
            logger.error(f"Error copying tensor: {e}")
            return service_pb2.TensorResponse(
                success=False,
                message=str(e),
            )

    async def DeleteTensors(
        self,
        request: service_pb2.DeleteTensorsRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.TensorResponse:
        """
        Delete tensors by ID.

        Args:
            request: DeleteTensorsRequest with tensor IDs to delete
            context: gRPC context

        Returns:
            TensorResponse with success status
        """
        deleted = 0
        for tensor_id in request.tensor_ids:
            if self.tensor_manager.delete(tensor_id):
                deleted += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Deleted {deleted} tensors")

        return service_pb2.TensorResponse(
            success=True,
            message=f"Deleted {deleted} tensors",
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
            # Auto-create input tensors from metadata if provided
            for metadata in request.tensor_metadata:
                self._ensure_tensor_exists(metadata)

            # Auto-create output tensors from metadata if provided
            for metadata in request.output_metadata:
                self._ensure_tensor_exists(metadata)

            # Resolve args - replace tensor refs with actual tensors
            args = tuple(self._resolve_aten_arg(arg) for arg in request.args)
            kwargs = self._resolve_kwargs(dict(request.kwargs))

            # Get the ATen op
            op = self._get_aten_op(request.op_name)

            input_tensor_ids = []
            output_tensor_ids = []

            if logger.isEnabledFor(logging.DEBUG):
                # Extract input tensor IDs for logging
                input_tensor_ids = [
                    arg.tensor.tensor_id
                    for arg in request.args
                    if arg.WhichOneof("value") == "tensor"
                ]
                output_tensor_ids = [ref.tensor_id for ref in request.outputs]
                logger.debug(
                    f"Executing {request.op_name} | "
                    f"inputs={input_tensor_ids} | outputs={output_tensor_ids}"
                )

                # Log input tensor data for debugging
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        logger.debug(
                            f"Input arg[{i}] shape={arg.shape} "
                            f"data={arg.cpu().flatten()[:8].tolist()}..."
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
                for i, (ref, tensor) in enumerate(zip(request.outputs, result_tensors)):
                    # TODO: check whether it's also an input tensor
                    if tensor is not None:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Output[{i}] tensor_id={ref.tensor_id} shape={tensor.shape} "
                                f"data={tensor.cpu().flatten()[:8].tolist()}..."
                            )
                        self.tensor_manager.register(ref.tensor_id, tensor)

                logger.debug(
                    f"Executed {request.op_name} | "
                    f"inputs={input_tensor_ids} | outputs={output_tensor_ids}"
                )

                return service_pb2.ExecuteAtenResponse(success=True)
            else:
                # Server-created outputs mode: register and return references
                output_refs = []
                for tensor in result_tensors:
                    output_refs.append(self._tensor_to_ref(tensor))

                logger.debug(
                    f"Executed {request.op_name} | "
                    f"inputs={input_tensor_ids} | outputs={output_tensor_ids}"
                )

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
        elif which == "scalar_dtype":
            # Convert dtype string (e.g., "torch.float32") to torch.dtype
            dtype_str = arg.scalar_dtype
            if dtype_str.startswith("torch."):
                dtype_name = dtype_str[6:]  # Remove "torch." prefix
                return getattr(torch, dtype_name)
            raise ValueError(f"Invalid dtype string: {dtype_str}")
        elif which == "scalar_memory_format":
            # Convert memory_format string (e.g., "torch.contiguous_format") to torch.memory_format
            format_str = arg.scalar_memory_format
            if format_str.startswith("torch."):
                format_name = format_str[6:]  # Remove "torch." prefix
                return getattr(torch, format_name)
            raise ValueError(f"Invalid memory_format string: {format_str}")
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
