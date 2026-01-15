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
from kpu.torch.server.storage import StorageManager

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
        self.storage = StorageManager()

    async def CreateTensor(
        self,
        request: service_pb2.CreateTensorRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.CreateTensorResponse:
        """
        Create a tensor on the server (explicit creation).

        Args:
            request: CreateTensorRequest with tensor metadata
            context: gRPC context

        Returns:
            CreateTensorResponse with success status
        """
        try:
            dtype = eval(request.dtype)  # "torch.float32" -> torch.float32

            self.storage.create(
                tensor_id=request.tensor_id,
                nbytes=request.nbytes,
                dtype=dtype,
                device_type=request.device_type,
                device_index=request.device_index,
            )

            logger.info(
                f"Created tensor {request.tensor_id} "
                f"(nbytes={request.nbytes}, dtype={dtype})"
            )

            return service_pb2.CreateTensorResponse(
                success=True,
                tensor_id=request.tensor_id,
            )
        except Exception as e:
            logger.error(f"Failed to create tensor: {e}")
            return service_pb2.CreateTensorResponse(
                success=False,
                message=str(e),
            )

    async def UpdateTensor(
        self,
        request_iterator: AsyncIterator[service_pb2.TensorChunk],
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.TensorResponse:
        """
        Receive tensor data and update storage (auto-creates if needed).

        Args:
            request_iterator: Async iterator of tensor chunks from client
            context: gRPC context

        Returns:
            TensorResponse with success status and received tensor IDs
        """
        assembler = TensorAssembler()
        received_ids = []

        try:
            async for chunk in request_iterator:
                metadata = dict(chunk.metadata) if chunk.metadata else {}
                tensor_id = int(metadata.get("target_tensor_id", "0"))

                logger.debug(
                    f"Received chunk {chunk.chunk_number}/{chunk.total_chunks} "
                    f"for tensor {tensor_id}"
                )

                # Convert metadata from proto map to dict
                chunk_metadata = dict(chunk.metadata) if chunk.metadata else None

                # Add chunk to assembler
                tensor = assembler.add_chunk(
                    tensor_id=chunk.tensor_id,
                    chunk_number=chunk.chunk_number,
                    data=chunk.data,
                    total_chunks=chunk.total_chunks,
                    is_last=chunk.is_last,
                    metadata=chunk_metadata,
                )

                if tensor is None:
                    continue

                # Get existing storage or create if it doesn't exist
                info = self.storage.get(tensor_id)
                if info is None:
                    info = self.storage.create(
                        tensor_id=tensor_id,
                        nbytes=tensor.untyped_storage().nbytes(),
                        dtype=tensor.dtype,
                    )
                    created = True
                else:
                    created = False

                # Copy data into storage
                offset = int(metadata.get("target_storage_offset", "0"))
                if offset == 0 and info.tensor.numel() == tensor.numel():
                    # Full tensor update
                    info.tensor.copy_(tensor.view(-1))
                else:
                    # Partial update with offset
                    info.tensor.view(-1)[
                        offset : offset + tensor.numel()
                    ].copy_(tensor.view(-1))

                received_ids.append(str(tensor_id))
                if created:
                    logger.info(
                        f"Created tensor {tensor_id} with shape {tensor.shape}"
                    )
                else:
                    logger.info(
                        f"Updated tensor {tensor_id} with shape {tensor.shape}"
                    )

            return service_pb2.TensorResponse(
                success=True,
                message=f"Received {len(received_ids)} tensor(s)",
                received_tensor_ids=received_ids,
            )

        except Exception as e:
            logger.error(f"Error receiving tensors: {e}")
            return service_pb2.TensorResponse(
                success=False,
                message=f"Error: {str(e)}",
                received_tensor_ids=received_ids,
            )

    async def GetStorageData(
        self,
        request: service_pb2.GetStorageDataRequest,
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

        info = self.storage.get(tensor_id)
        if info is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"Tensor {tensor_id} not found"
            )
            return

        try:
            # Create view with requested shape/stride
            if stride:
                tensor = info.tensor.as_strided(shape, stride, offset)
            else:
                numel = torch.Size(shape).numel()
                tensor = info.tensor[offset : offset + numel].view(shape)

            logger.info(f"Sending tensor {tensor_id} with shape {tensor.shape}")

            # Stream the tensor data
            for chunk_data in serialize_tensor_to_chunks(
                tensor.contiguous(), tensor_id, self.chunk_size
            ):
                t_id, c_num, data, total, is_last, meta = chunk_data

                chunk = service_pb2.TensorChunk(
                    tensor_id=t_id,
                    chunk_number=c_num,
                    data=data,
                    total_chunks=total,
                    is_last=is_last,
                )

                if meta:
                    chunk.metadata.update(meta)

                logger.debug(f"Sending chunk {c_num}/{total} for tensor {t_id}")
                yield chunk

        except Exception as e:
            logger.error(f"Error sending tensor data: {e}")
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
        src_info = self.storage.get(request.src_tensor_id)
        dst_info = self.storage.get(request.dst_tensor_id)

        if src_info is None:
            return service_pb2.TensorResponse(
                success=False,
                message=f"Source tensor {request.src_tensor_id} not found",
            )

        if dst_info is None:
            return service_pb2.TensorResponse(
                success=False,
                message=f"Destination tensor {request.dst_tensor_id} not found",
            )

        try:
            # Copy bytes
            src_bytes = src_info.tensor.view(torch.uint8)
            dst_bytes = dst_info.tensor.view(torch.uint8)
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

            if request.outputs:
                # Pre-allocated outputs mode
                outputs = [self._resolve_tensor_ref(ref) for ref in request.outputs]

                logger.info(
                    f"Executing {request.op_name} with {len(args)} args, "
                    f"{len(outputs)} pre-allocated outputs, {len(kwargs)} kwargs"
                )

                if len(outputs) == 1:
                    op(*args, out=outputs[0], **kwargs)
                else:
                    op(*args, out=tuple(outputs), **kwargs)

                return service_pb2.ExecuteAtenResponse(success=True)
            else:
                # Server creates outputs mode
                logger.info(
                    f"Executing {request.op_name} with {len(args)} args, "
                    f"server-created outputs, {len(kwargs)} kwargs"
                )

                result = op(*args, **kwargs)

                # Convert result to TensorReference list
                output_refs = []
                if isinstance(result, torch.Tensor):
                    output_refs.append(self._tensor_to_ref(result))
                elif isinstance(result, (tuple, list)):
                    for t in result:
                        if isinstance(t, torch.Tensor):
                            output_refs.append(self._tensor_to_ref(t))

                return service_pb2.ExecuteAtenResponse(
                    success=True,
                    output_tensors=output_refs,
                )

        except Exception as e:
            logger.error(f"Error executing ATen operation: {e}")
            return service_pb2.ExecuteAtenResponse(
                success=False,
                message=str(e),
            )

    def _tensor_to_ref(self, tensor: torch.Tensor) -> service_pb2.TensorReference:
        """Convert a tensor to TensorReference proto."""
        storage_id = self.storage.register_tensor(tensor)
        return service_pb2.TensorReference(
            tensor_id=storage_id,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=tensor.untyped_storage().nbytes(),
            device_type=tensor.device.type,
            stride=list(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            device_index=tensor.device.index or 0,
        )

    def _resolve_tensor_ref(
        self, ref: service_pb2.TensorReference
    ) -> torch.Tensor:
        """Resolve TensorReference to actual tensor."""
        info = self.storage.get(ref.tensor_id)
        if info is None:
            raise ValueError(f"Tensor {ref.tensor_id} not found")

        shape = tuple(ref.shape)
        stride = tuple(ref.stride) if ref.stride else None

        if stride:
            return info.tensor.as_strided(shape, stride, ref.storage_offset)

        numel = torch.Size(shape).numel()
        return info.tensor[: numel].view(shape)

    def _resolve_aten_arg(self, arg: service_pb2.AtenArgument):
        """Resolve an AtenArgument to a Python value, replacing tensor refs."""
        which = arg.WhichOneof("value")

        if which == "tensor":
            return self._resolve_tensor_ref(arg.tensor)
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
            op_name: Operation name (e.g., "aten::add.Tensor")

        Returns:
            The ATen operator callable
        """
        # op_name format: "aten::add.Tensor"
        parts = op_name.replace("aten::", "").split(".")
        op = getattr(torch.ops.aten, parts[0])
        if len(parts) > 1:
            op = getattr(op, parts[1])
        return op
