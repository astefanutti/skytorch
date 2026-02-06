"""
Tensor gRPC service implementation for KPU PyTorch backend.

This module implements the gRPC service for tensor management and
ATen operation execution.
"""

import functools
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
    tensor_from_bytes,
    tensor_to_bytes,
    parse_dtype,
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

        dtype = parse_dtype(metadata.dtype)
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
                if tensor.device == target.device:
                    target.copy_(tensor)
                else:
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
        dtype = parse_dtype(request.dtype)
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
            tensor_for_serialization = tensor if tensor.device.type == 'cpu' else tensor.cpu()
            for chunk in serialize_tensor_to_chunks(
                tensor_id, tensor_for_serialization.detach(), self.chunk_size
            ):
                yield chunk

            if logger.isEnabledFor(logging.DEBUG):
                data_preview = tensor.cpu().flatten()[:8].tolist()
                logger.debug(
                    f"Sent tensor tensor_id={tensor_id} shape={tensor.shape} "
                    f"data={data_preview}..."
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
                        data_preview = arg.cpu().flatten()[:8].tolist()
                        logger.debug(
                            f"Input arg[{i}] shape={arg.shape} "
                            f"data={data_preview}..."
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
                            data_preview = tensor.cpu().flatten()[:8].tolist()
                            logger.debug(
                                f"Output[{i}] tensor_id={ref.tensor_id} shape={tensor.shape} "
                                f"data={data_preview}..."
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

    @functools.lru_cache(maxsize=1024)
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

    async def _handle_update_tensor(
        self, request: service_pb2.UpdateTensorRequest
    ) -> service_pb2.TensorResponse:
        """Handle update_tensor request in streaming context.

        Deserializes tensor data and stores it on the server.

        Args:
            request: UpdateTensorRequest with tensor data

        Returns:
            TensorResponse with success status
        """
        try:
            # Get or create tensor
            if request.HasField("metadata"):
                tensor = self._ensure_tensor_exists(request.metadata)
            else:
                tensor = self.tensor_manager.get(request.tensor_id)

            # Deserialize tensor data
            dtype = parse_dtype(request.dtype)
            shape = list(request.shape)

            # Copy data from bytes to tensor
            src_tensor = tensor_from_bytes(request.data, dtype, shape)
            tensor.copy_(src_tensor)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Updated tensor {request.tensor_id} via stream")

            return service_pb2.TensorResponse(success=True)

        except Exception as e:
            logger.error(f"Error updating tensor via stream: {e}")
            return service_pb2.TensorResponse(success=False, message=str(e))

    async def _handle_get_tensor(
        self, request: service_pb2.GetTensorRequest
    ) -> service_pb2.GetTensorResponse:
        """Handle get_tensor request in streaming context.

        Serializes tensor data for download.

        Args:
            request: GetTensorRequest with tensor ID

        Returns:
            GetTensorResponse with serialized tensor data
        """
        try:
            tensor = self.tensor_manager.get(request.tensor_id)

            # Apply view parameters if provided
            shape = list(request.shape) if request.shape else list(tensor.shape)
            stride = list(request.stride) if request.stride else list(tensor.stride())
            storage_offset = request.storage_offset

            # Create view with requested parameters
            view_tensor = tensor.as_strided(shape, stride, storage_offset)

            # Serialize tensor data to bytes
            data = tensor_to_bytes(view_tensor)

            if logger.isEnabledFor(logging.DEBUG):
                data_preview = view_tensor.contiguous().cpu().flatten()[:8].tolist()
                logger.debug(f"Sent tensor {request.tensor_id} via stream data={data_preview}...")

            return service_pb2.GetTensorResponse(
                success=True,
                data=data,
                shape=shape,
                dtype=str(tensor.dtype),
                stride=stride,
                storage_offset=storage_offset,
            )

        except Exception as e:
            logger.error(f"Error getting tensor via stream: {e}")
            return service_pb2.GetTensorResponse(success=False, message=str(e))

    async def StreamOperations(
        self,
        request_iterator: AsyncIterator[service_pb2.StreamRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[service_pb2.StreamResponse]:
        """
        Bidirectional streaming handler for low-latency operations.

        Processes StreamRequest messages and yields StreamResponse messages
        in FIFO order. All operations go through this single channel to
        ensure proper sequencing.

        Supports chunked requests for large payloads (>1MB). Intermediate
        chunks are buffered; only the final chunk triggers processing and
        generates a response.

        Args:
            request_iterator: Async iterator of StreamRequest messages
            context: gRPC context

        Yields:
            StreamResponse messages with operation results (in request order)
        """
        # Per-stream state for the current chunked request being assembled
        # Only one chunked request can be in-flight at a time per stream (FIFO)
        current_chunk_state: list = [None]  # [buffer, first_request, chunk_size] or None

        async for request in request_iterator:
            total_chunks = request.total_chunks

            if total_chunks <= 1:
                # Single message (existing behavior)
                response = await self._handle_single_request(request, context)
                yield response
            else:
                # Chunked request
                response = await self._handle_chunked_request(request, context, current_chunk_state)
                if response is not None:
                    yield response

    async def _handle_single_request(
        self,
        request: service_pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.StreamResponse:
        """Handle a single (non-chunked) stream request."""
        response = service_pb2.StreamResponse()

        try:
            request_type = request.WhichOneof("request")

            if request_type == "execute_aten":
                result = await self.ExecuteAtenOperation(
                    request.execute_aten, context
                )
                response.success = result.success
                if not result.success:
                    response.error_message = result.message
                response.execute_aten.CopyFrom(result)

            elif request_type == "delete_tensors":
                result = await self.DeleteTensors(request.delete_tensors, context)
                response.success = result.success
                if not result.success:
                    response.error_message = result.message
                response.delete_tensors.CopyFrom(result)

            elif request_type == "copy_tensor":
                result = await self.CopyTensor(request.copy_tensor, context)
                response.success = result.success
                if not result.success:
                    response.error_message = result.message
                response.copy_tensor.CopyFrom(result)

            elif request_type == "update_tensor":
                result = await self._handle_update_tensor(request.update_tensor)
                response.success = result.success
                if not result.success:
                    response.error_message = result.message
                response.update_tensor.CopyFrom(result)

            elif request_type == "get_tensor":
                result = await self._handle_get_tensor(request.get_tensor)
                response.success = result.success
                if not result.success:
                    response.error_message = result.message
                response.get_tensor.CopyFrom(result)

            else:
                response.success = False
                response.error_message = f"Unknown request type: {request_type}"

        except Exception as e:
            logger.error(f"Error in StreamOperations: {e}")
            response.success = False
            response.error_message = str(e)

        return response

    async def _handle_chunked_request(
        self,
        request: service_pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
        chunk_state: list,
    ) -> service_pb2.StreamResponse | None:
        """
        Handle a chunked stream request.

        Buffers intermediate chunks and returns None. When the final chunk
        is received, processes the assembled request and returns the response.

        Only one chunked request can be in-flight at a time per stream (FIFO).

        Args:
            request: StreamRequest with chunking metadata
            context: gRPC context
            chunk_state: Mutable list holding [buffer, first_request, chunk_size] or [None]

        Returns:
            StreamResponse for final chunk, None for intermediate chunks
        """
        chunk_number = request.chunk_number
        total_chunks = request.total_chunks
        update_req = request.update_tensor

        if chunk_number == 0:
            # First chunk: initialize buffer
            buffer = bytearray(request.total_bytes)
            chunk_size = len(update_req.data)
            chunk_state[0] = (buffer, request, chunk_size)

        if chunk_state[0] is None:
            return service_pb2.StreamResponse(
                success=False,
                error_message="Missing first chunk for chunked request"
            )

        buffer, first_request, chunk_size = chunk_state[0]

        # Write chunk to buffer at correct offset
        start = chunk_number * chunk_size
        buffer[start:start + len(update_req.data)] = update_req.data

        if chunk_number < total_chunks - 1:
            # Intermediate chunk: buffer and continue (no response)
            return None

        # Last chunk: clear state and process complete request
        chunk_state[0] = None

        return await self._process_assembled_update_tensor(first_request, buffer, context)

    async def _process_assembled_update_tensor(
        self,
        first_request: service_pb2.StreamRequest,
        buffer: bytearray,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.StreamResponse:
        """
        Process an assembled update_tensor request from chunked data.

        Args:
            first_request: The first StreamRequest containing metadata
            buffer: Assembled data buffer from all chunks
            context: gRPC context

        Returns:
            StreamResponse with operation result
        """
        response = service_pb2.StreamResponse()
        try:
            update_req = first_request.update_tensor

            # Auto-create tensor from metadata if provided
            if update_req.HasField("metadata"):
                self._ensure_tensor_exists(update_req.metadata)

            # Deserialize from assembled buffer
            dtype = parse_dtype(update_req.dtype)
            shape = list(update_req.shape)

            src_tensor = tensor_from_bytes(bytes(buffer), dtype, shape)

            # Copy to target tensor
            target = self.tensor_manager.get(update_req.tensor_id)
            if src_tensor.device == target.device:
                target.copy_(src_tensor)
            else:
                target.copy_(src_tensor.to(target.device))

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Updated tensor {update_req.tensor_id} from chunked stream "
                    f"(shape={shape}, {len(buffer)} bytes)"
                )

            response.success = True
            response.update_tensor.CopyFrom(service_pb2.TensorResponse(success=True))

        except Exception as e:
            logger.error(f"Error processing chunked update_tensor: {e}")
            response.success = False
            response.error_message = str(e)

        return response
