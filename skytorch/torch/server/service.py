"""
Tensor gRPC service implementation for SkyTorch PyTorch backend.

This module implements the gRPC service for tensor management and
ATen operation execution.
"""

import asyncio
import functools
import logging
import pickle
import struct
import sys
import time
from typing import AsyncIterator

try:
    import cloudpickle
    import grpc
    import torch
except ImportError as e:
    raise ImportError(
        f"Required dependency not found: {e}. Install with: pip install grpcio torch cloudpickle"
    )

try:
    from skytorch.torch.server import service_pb2
    from skytorch.torch.server import service_pb2_grpc
except ImportError:
    raise ImportError(
        "Generated gRPC code not found. Run hack/gen-grpc-proto.sh first.\n"
        "Make sure to install grpcio-tools: pip install grpcio-tools"
    )

from skytorch.torch.server.serialization import (
    serialize_tensor_to_chunks,
    TensorAssembler,
    DEFAULT_CHUNK_SIZE,
    tensor_from_bytes,
    tensor_to_bytes,
    parse_dtype,
)
from skytorch.torch.profiler import PROFILING_ENABLED
from skytorch.torch.server.tensor import TensorManager

try:
    from skytorch.torch.server._C import (
        execute_raw_aten_inline as _cpp_execute_raw_aten_inline,
        execute_raw_batched_aten_inline as _cpp_execute_raw_batched_aten_inline,
    )

    _USE_CPP_PARSER = True
except ImportError:
    _USE_CPP_PARSER = False

logger = logging.getLogger(__name__)

# Pre-compiled struct formats for binary parsing (avoids per-call format string parsing)
_STRUCT_Q = struct.Struct("<Q")  # uint64
_STRUCT_q = struct.Struct("<q")  # int64
_STRUCT_d = struct.Struct("<d")  # float64
_STRUCT_H = struct.Struct("<H")  # uint16
_STRUCT_I = struct.Struct("<I")  # uint32
_STRUCT_i = struct.Struct("<i")  # int32
# Pre-compiled shape/stride formats for common dimensions (0-dim to 8-dim)
_STRUCT_SHAPE: dict[int, struct.Struct] = {i: struct.Struct(f"<{i}q") for i in range(9)}


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
        # Pending tensors from ExecuteFunction, keyed by server-assigned storage_id
        self._pending_tensors: dict[int, torch.Tensor] = {}
        # Counter for server-assigned storage IDs (starts above client range)
        self._next_remote_storage_id = 2**32
        # Deferred error from fire-and-forget operations, reported on next get_tensor
        self._deferred_error: str | None = None

    def _ensure_tensor_exists(self, metadata: service_pb2.TensorMetadata) -> torch.Tensor:
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
        try:
            return self.tensor_manager._tensors[tensor_id]
        except KeyError:
            pass

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
                    f"Auto-created tensor {tensor_id} " f"(nbytes={metadata.nbytes}, dtype={dtype})"
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
            # Auto-create tensor from metadata if provided
            if request.HasField("metadata"):
                self._ensure_tensor_exists(request.metadata)

            tensor = self.tensor_manager.get(tensor_id)
        except ValueError:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Tensor {tensor_id} not found")

        try:
            # Stream the tensor data
            tensor_for_serialization = tensor if tensor.device.type == "cpu" else tensor.cpu()
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
                    f"Copied tensor {request.src_tensor_id} " f"to tensor {request.dst_tensor_id}"
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

    async def ExecuteFunction(
        self,
        request: service_pb2.ExecuteFunctionRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.ExecuteFunctionResponse:
        """
        Execute a pickled function on the server and return tensor metadata.

        The function is deserialized via cloudpickle, executed, and any tensors
        in the result are assigned server-side storage IDs. The actual tensor data
        stays on the GPU; only metadata is returned to the client.

        The entire operation runs in a thread to keep the asyncio event loop free
        for health checks during long-running functions (e.g. model downloads).

        Args:
            request: ExecuteFunctionRequest with pickled callable, args, kwargs
            context: gRPC context

        Returns:
            ExecuteFunctionResponse with tensor metadata
        """
        if request.callable_source:
            logger.info(
                f"ExecuteFunction: received request "
                f"(source={len(request.callable_source)}B, name={request.callable_name!r}, "
                f"args={len(request.args)}B, kwargs={len(request.kwargs)}B)"
            )
        else:
            logger.info(
                f"ExecuteFunction: received request "
                f"(callable={len(request.callable)}B, "
                f"args={len(request.args)}B, kwargs={len(request.kwargs)}B)"
            )
        try:
            return await asyncio.to_thread(self._execute_function_sync, request)
        except Exception as e:
            logger.error(f"Error executing function: {e}")
            return service_pb2.ExecuteFunctionResponse(success=False, error_message=str(e))

    def _execute_function_sync(
        self,
        request: service_pb2.ExecuteFunctionRequest,
    ) -> service_pb2.ExecuteFunctionResponse:
        if request.callable_source:
            namespace = {}
            exec(request.callable_source, namespace)
            fn = namespace[request.callable_name]
        else:
            fn = cloudpickle.loads(request.callable)
        args = pickle.loads(request.args) if request.args else ()
        kwargs = pickle.loads(request.kwargs) if request.kwargs else {}

        logger.info(f"ExecuteFunction: calling {fn!r}")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            sys.stdout.flush()
            raise
        sys.stdout.flush()

        # Extract tensors from result
        if result is None:
            tensors = {}
        elif isinstance(result, torch.nn.Module):
            tensors = result.state_dict()
            # Include non-persistent buffers (e.g., inv_freq in rotary embeddings)
            for name, buffer in result.named_buffers():
                if name not in tensors and buffer is not None:
                    tensors[name] = buffer
        elif isinstance(result, dict):
            tensors = {k: v for k, v in result.items() if isinstance(v, torch.Tensor)}
        elif isinstance(result, torch.Tensor):
            tensors = {"tensor": result}
        else:
            return service_pb2.ExecuteFunctionResponse(
                success=False,
                error_message=f"Unsupported result type: {type(result).__name__}. "
                "Expected nn.Module, dict of tensors, tensor, or None.",
            )

        # Assign storage_ids, detect shared storage
        storage_map: dict[int, int] = {}  # GPU data_ptr → storage_id
        tensor_infos = []
        for name, tensor in tensors.items():
            data_ptr = tensor.untyped_storage().data_ptr()
            if data_ptr not in storage_map:
                storage_map[data_ptr] = self._next_remote_storage_id
                self._next_remote_storage_id += 1
            sid = storage_map[data_ptr]
            self._pending_tensors[sid] = tensor
            tensor_infos.append(
                service_pb2.RemoteTensorInfo(
                    name=name,
                    storage_id=sid,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    stride=list(tensor.stride()),
                    storage_offset=tensor.storage_offset(),
                    storage_nbytes=tensor.untyped_storage().nbytes(),
                    device_type=tensor.device.type,
                    device_index=tensor.device.index or 0,
                )
            )

        logger.info(
            f"ExecuteFunction: {len(tensors)} tensors, " f"{len(storage_map)} unique storages"
        )

        return service_pb2.ExecuteFunctionResponse(success=True, tensors=tensor_infos)

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
                        logger.debug(f"Input arg[{i}] shape={arg.shape} " f"data={data_preview}...")

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
        elif which == "scalar_layout":
            # Convert layout string (e.g., "torch.strided") to torch.layout
            layout_str = arg.scalar_layout
            if layout_str.startswith("torch."):
                layout_name = layout_str[6:]  # Remove "torch." prefix
                return getattr(torch, layout_name)
            raise ValueError(f"Invalid layout string: {layout_str}")
        elif which == "none_value":
            return None
        elif which == "list_value":
            values = [self._resolve_aten_arg(v) for v in arg.list_value.values]
            if arg.list_value.is_tuple:
                return tuple(values)
            return values
        else:
            raise ValueError(f"Unknown AtenArgument type: {which}")

    def _resolve_kwargs(self, kwargs: dict[str, service_pb2.AtenArgument]) -> dict:
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

    def _parse_raw_execute_aten(self, data: bytes):
        """Parse binary execute_aten format (v2: metadata-first) from C++ RequestBuilder.

        Binary format order: header, op_name, metadata, outputs, args, kwargs.
        Single-pass parsing — metadata is parsed first so tensors exist when
        args reference them by ID.

        Returns:
            Tuple of (op_name, args, kwargs_dict, output_tensor_ids)
            where args are resolved Python values with tensor refs resolved.
        """
        pos = 0

        # Header (4 bytes)
        num_args = data[pos]
        num_kwargs = data[pos + 1]
        num_outputs = data[pos + 2]
        num_metadata = data[pos + 3]
        pos = 4

        # Op name
        op_name_len = _STRUCT_H.unpack_from(data, pos)[0]
        pos += 2
        op_name = data[pos : pos + op_name_len].decode("utf-8")
        pos += op_name_len

        # Parse metadata and auto-create tensors (now immediately after op_name)
        tensors = self.tensor_manager._tensors
        for _ in range(num_metadata):
            # Peek at tensor_id to skip already-registered tensors
            tensor_id = _STRUCT_Q.unpack_from(data, pos)[0]
            if tensor_id in tensors:
                pos = self._skip_raw_tensor_metadata(data, pos)
            else:
                pos = self._parse_and_create_tensor(data, pos)

        # Parse output tensor IDs
        output_tensor_ids = []
        for _ in range(num_outputs):
            tid = _STRUCT_Q.unpack_from(data, pos)[0]
            pos += 8
            output_tensor_ids.append(tid)

        # Parse args (tensors already exist from metadata above)
        args = []
        for _ in range(num_args):
            val, pos = self._parse_raw_arg(data, pos)
            args.append(val)

        # Parse kwargs
        kwargs = {}
        for _ in range(num_kwargs):
            name_len = data[pos]
            pos += 1
            name = data[pos : pos + name_len].decode("utf-8")
            pos += name_len
            val, pos = self._parse_raw_arg(data, pos)
            kwargs[name] = val

        return op_name, tuple(args), kwargs, output_tensor_ids

    def _parse_raw_arg(self, data: bytes, pos: int):
        """Parse a single argument from binary format. Returns (value, new_pos)."""
        arg_type = data[pos]
        pos += 1

        if arg_type == 0x01:  # TENSOR_ID (most common — checked first)
            tid = _STRUCT_Q.unpack_from(data, pos)[0]
            return self.tensor_manager.get(tid), pos + 8
        elif arg_type == 0x00:  # NONE
            return None, pos
        elif arg_type == 0x02:  # INT64
            return _STRUCT_q.unpack_from(data, pos)[0], pos + 8
        elif arg_type == 0x03:  # FLOAT64
            return _STRUCT_d.unpack_from(data, pos)[0], pos + 8
        elif arg_type == 0x04:  # BOOL
            return data[pos] != 0, pos + 1
        elif arg_type == 0x05:  # DTYPE
            slen = data[pos]
            s = data[pos + 1 : pos + 1 + slen].decode("utf-8")
            if s.startswith("torch."):
                return getattr(torch, s[6:]), pos + 1 + slen
            raise ValueError(f"Invalid dtype string: {s}")
        elif arg_type == 0x06:  # MEMORY_FORMAT
            slen = data[pos]
            s = data[pos + 1 : pos + 1 + slen].decode("utf-8")
            if s.startswith("torch."):
                return getattr(torch, s[6:]), pos + 1 + slen
            raise ValueError(f"Invalid memory_format string: {s}")
        elif arg_type == 0x07:  # LAYOUT
            slen = data[pos]
            s = data[pos + 1 : pos + 1 + slen].decode("utf-8")
            if s.startswith("torch."):
                return getattr(torch, s[6:]), pos + 1 + slen
            raise ValueError(f"Invalid layout string: {s}")
        elif arg_type == 0x08:  # STRING
            slen = _STRUCT_H.unpack_from(data, pos)[0]
            s = data[pos + 2 : pos + 2 + slen].decode("utf-8")
            return s, pos + 2 + slen
        elif arg_type == 0x09:  # LIST
            count = _STRUCT_H.unpack_from(data, pos)[0]
            pos += 2
            items = []
            for _ in range(count):
                val, pos = self._parse_raw_arg(data, pos)
                items.append(val)
            return items, pos
        elif arg_type == 0x0A:  # TUPLE
            count = _STRUCT_H.unpack_from(data, pos)[0]
            pos += 2
            items = []
            for _ in range(count):
                val, pos = self._parse_raw_arg(data, pos)
                items.append(val)
            return tuple(items), pos
        else:
            raise ValueError(f"Unknown arg type: 0x{arg_type:02x}")

    def _skip_raw_tensor_metadata(self, data: bytes, pos: int) -> int:
        """Skip past tensor metadata in binary format without parsing. Returns new position."""
        pos += 8  # tensor_id (uint64)
        ndim = data[pos]
        pos += 1  # ndim (uint8)
        pos += ndim * 16  # shape + stride (ndim * 2 * int64)
        dtype_len = data[pos]
        pos += 1 + dtype_len  # dtype_str_len + dtype_str
        pos += 16  # storage_offset + nbytes (2 * int64)
        dt_len = data[pos]
        pos += 1 + dt_len  # device_type_len + device_type
        pos += 4  # device_index (int32)
        has_ref = data[pos]
        pos += 1  # has_tensor_ref (uint8)
        if has_ref:
            pos += 8  # tensor_ref (uint64)
        return pos

    def _parse_and_create_tensor(self, data: bytes, pos: int) -> int:
        """Parse tensor metadata from binary and create tensor directly. Returns new position.

        Bypasses protobuf TensorMetadata construction for faster tensor creation.
        """
        tensor_id = _STRUCT_Q.unpack_from(data, pos)[0]
        pos += 8
        ndim = data[pos]
        pos += 1

        fmt = _STRUCT_SHAPE.get(ndim)
        if fmt is not None:
            shape = list(fmt.unpack_from(data, pos))
            pos += ndim * 8
            stride = list(fmt.unpack_from(data, pos))
            pos += ndim * 8
        else:
            shape = list(struct.unpack_from(f"<{ndim}q", data, pos))
            pos += ndim * 8
            stride = list(struct.unpack_from(f"<{ndim}q", data, pos))
            pos += ndim * 8

        # dtype string (uint8 len + bytes)
        dtype_len = data[pos]
        pos += 1
        dtype_str = data[pos : pos + dtype_len].decode("utf-8")
        pos += dtype_len

        storage_offset = _STRUCT_q.unpack_from(data, pos)[0]
        pos += 8
        nbytes = _STRUCT_q.unpack_from(data, pos)[0]
        pos += 8

        # device_type string (uint8 len + bytes)
        dt_len = data[pos]
        pos += 1
        device_type = data[pos : pos + dt_len].decode("utf-8")
        pos += dt_len

        device_index = _STRUCT_i.unpack_from(data, pos)[0]
        pos += 4

        # tensor_ref (optional)
        has_tensor_ref = data[pos]
        pos += 1
        tensor_ref = None
        if has_tensor_ref:
            tensor_ref = _STRUCT_Q.unpack_from(data, pos)[0]
            pos += 8

        # Create tensor directly (no protobuf intermediary)
        dtype = parse_dtype(dtype_str)
        if tensor_ref is not None:
            base_tensor = self.tensor_manager.get(tensor_ref)
            storage = base_tensor.untyped_storage()
            tensor = torch.empty(0, dtype=dtype, device=base_tensor.device).set_(
                storage, storage_offset, shape, stride
            )
        else:
            device = torch.device(device_type, device_index)
            storage = torch.UntypedStorage(nbytes, device=device)
            tensor = torch.empty(0, dtype=dtype, device=device).set_(
                storage, storage_offset, shape, stride
            )

        self.tensor_manager.register(tensor_id, tensor)

        if logger.isEnabledFor(logging.DEBUG):
            if tensor_ref is not None:
                logger.debug(
                    f"Auto-created tensor {tensor_id} "
                    f"(view of {tensor_ref}, shape={shape}, dtype={dtype})"
                )
            else:
                logger.debug(
                    f"Auto-created tensor {tensor_id} (nbytes={nbytes}, dtype={dtype})"
                )

        return pos

    def _execute_raw_aten_inline(self, data: bytes) -> None:
        """Execute a raw binary execute_aten request inline (no response construction).

        Used for fire-and-forget operations. Raises on error.
        """
        op_name, args, kwargs, output_tensor_ids = self._parse_raw_execute_aten(data)

        # Get the ATen op (metadata already auto-created during parse)
        op = self._get_aten_op(op_name)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Executing raw {op_name} | outputs={output_tensor_ids}")

        # Execute
        result = op(*args, **kwargs)

        # Register outputs
        if output_tensor_ids:
            if isinstance(result, torch.Tensor):
                result_tensors = [result]
            elif isinstance(result, (tuple, list)):
                result_tensors = [t for t in result if isinstance(t, torch.Tensor)]
            else:
                result_tensors = []

            for tid, tensor in zip(output_tensor_ids, result_tensors, strict=False):
                if tensor is not None:
                    self.tensor_manager.register(tid, tensor)

    def _execute_batch_inline(self, operations) -> None:
        """Execute a batch of ExecuteAtenRequest operations inline.

        Avoids per-op response construction overhead.
        Raises on first error.
        """
        for op_request in operations:
            # Auto-create tensors from metadata
            for metadata in op_request.tensor_metadata:
                self._ensure_tensor_exists(metadata)
            for metadata in op_request.output_metadata:
                self._ensure_tensor_exists(metadata)

            args = tuple(self._resolve_aten_arg(arg) for arg in op_request.args)
            kwargs = self._resolve_kwargs(dict(op_request.kwargs))
            op = self._get_aten_op(op_request.op_name)

            result = op(*args, **kwargs)

            # Register outputs directly
            if op_request.outputs:
                if isinstance(result, torch.Tensor):
                    result_tensors = [result]
                elif isinstance(result, (tuple, list)):
                    result_tensors = [t for t in result if isinstance(t, torch.Tensor)]
                else:
                    result_tensors = []

                for ref, tensor in zip(op_request.outputs, result_tensors, strict=False):
                    if tensor is not None:
                        self.tensor_manager.register(ref.tensor_id, tensor)

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
            # Auto-create tensor from metadata if provided
            if request.HasField("metadata"):
                self._ensure_tensor_exists(request.metadata)

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

    async def _handle_get_scalar(
        self,
        request: service_pb2.GetScalarRequest,
        server_profiler=None,
    ) -> service_pb2.GetScalarResponse:
        """Handle get_scalar request in streaming context.

        Resolves the tensor, calls .item(), and returns the typed scalar value.

        Args:
            request: GetScalarRequest with tensor ID
            server_profiler: Optional ServerProfiler for timing

        Returns:
            GetScalarResponse with the scalar value
        """
        try:
            for metadata in request.tensor_metadata:
                self._ensure_tensor_exists(metadata)

            if server_profiler is not None:
                _t0 = time.perf_counter_ns()

            tensor = self.tensor_manager.get(request.tensor_id)

            if server_profiler is not None:
                _t1 = time.perf_counter_ns()
                server_profiler.scalar_lookup.add(_t1 - _t0)

            value = tensor.item()

            if server_profiler is not None:
                _t2 = time.perf_counter_ns()
                server_profiler.scalar_gpu_sync.add(_t2 - _t1)

            if isinstance(value, bool):
                return service_pb2.GetScalarResponse(bool_value=value)
            elif isinstance(value, int):
                return service_pb2.GetScalarResponse(int_value=value)
            elif isinstance(value, float):
                return service_pb2.GetScalarResponse(float_value=value)
            else:
                return service_pb2.GetScalarResponse(float_value=float(value))

        except Exception as e:
            logger.error(f"Error getting scalar via stream: {e}")
            raise

    # Request types that are fire-and-forget (no response sent to client)
    _FIRE_AND_FORGET_TYPES = frozenset(
        {
            "execute_aten",
            "delete_tensors",
            "copy_tensor",
            "update_tensor",
            "register_tensors",
            "batched_execute_aten",
            "raw_execute_aten",
            "raw_batched_execute_aten",
        }
    )

    async def StreamOperations(
        self,
        request_iterator: AsyncIterator[service_pb2.StreamRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[service_pb2.StreamResponse]:
        """
        Bidirectional streaming handler for low-latency operations.

        Fire-and-forget operations (execute_aten, delete_tensors, copy_tensor,
        update_tensor, register_tensors) are processed without yielding a
        response. Errors from these operations are deferred and reported on
        the next get_tensor response.

        Only get_tensor operations yield a response.

        Args:
            request_iterator: Async iterator of StreamRequest messages
            context: gRPC context

        Yields:
            StreamResponse messages for get_tensor operations only
        """
        # Per-stream state for the current chunked request being assembled
        # Only one chunked request can be in-flight at a time per stream (FIFO)
        current_chunk_state: list = [None]  # [buffer, first_request, chunk_size] or None

        # Per-stream profiler (created only when profiling is enabled)
        _sprof = None
        if PROFILING_ENABLED:
            from skytorch.torch.profiler import ServerProfiler

            _sprof = ServerProfiler()
            _sprof.stream_start_ns = time.perf_counter_ns()
            _t_prev_end = _sprof.stream_start_ns

        try:
            async for request in request_iterator:
                if PROFILING_ENABLED:
                    _t_recv = time.perf_counter_ns()
                    _sprof.idle_time.add(_t_recv - _t_prev_end)

                total_chunks = request.total_chunks

                if total_chunks > 1:
                    # Chunked request (always fire-and-forget update_tensor)
                    await self._handle_chunked_request(request, context, current_chunk_state)
                    if PROFILING_ENABLED:
                        _t_prev_end = time.perf_counter_ns()
                    continue

                # Inline hot-path: check fire-and-forget types with HasField
                # instead of WhichOneof + frozenset lookup
                if request.HasField("raw_execute_aten"):
                    try:
                        if PROFILING_ENABLED:
                            _t0 = time.perf_counter_ns()

                        if _USE_CPP_PARSER:
                            _cpp_execute_raw_aten_inline(
                                request.raw_execute_aten,
                                self.tensor_manager._tensors,
                            )
                        else:
                            self._execute_raw_aten_inline(request.raw_execute_aten)

                        if PROFILING_ENABLED:
                            _t1 = time.perf_counter_ns()
                            _sprof.raw_execute.add(_t1 - _t0)
                            _sprof.total_ops += 1
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("raw_batched_execute_aten"):
                    try:
                        if PROFILING_ENABLED:
                            _t0 = time.perf_counter_ns()

                        if _USE_CPP_PARSER:
                            _cpp_execute_raw_batched_aten_inline(
                                request.raw_batched_execute_aten,
                                self.tensor_manager._tensors,
                            )
                        else:
                            raw_data = request.raw_batched_execute_aten
                            pos = 0
                            while pos < len(raw_data):
                                op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
                                pos += 4
                                op_data = raw_data[pos : pos + op_len]
                                pos += op_len
                                self._execute_raw_aten_inline(op_data)

                        if PROFILING_ENABLED:
                            _t1 = time.perf_counter_ns()
                            _sprof.raw_batched_execute.add(_t1 - _t0)
                            # Count ops by scanning length prefixes
                            raw_data = request.raw_batched_execute_aten
                            _n_ops = 0
                            _pos = 0
                            while _pos < len(raw_data):
                                _op_len = _STRUCT_I.unpack_from(raw_data, _pos)[0]
                                _pos += 4 + _op_len
                                _n_ops += 1
                            _sprof.total_ops += _n_ops
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("batched_execute_aten"):
                    try:
                        self._execute_batch_inline(
                            request.batched_execute_aten.operations
                        )
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("execute_aten"):
                    try:
                        result = await self.ExecuteAtenOperation(
                            request.execute_aten, context
                        )
                        if not result.success:
                            self._deferred_error = result.message
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("delete_tensors"):
                    try:
                        result = await self.DeleteTensors(
                            request.delete_tensors, context
                        )
                        if not result.success:
                            self._deferred_error = result.message
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("copy_tensor"):
                    try:
                        result = await self.CopyTensor(request.copy_tensor, context)
                        if not result.success:
                            self._deferred_error = result.message
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("update_tensor"):
                    try:
                        result = await self._handle_update_tensor(request.update_tensor)
                        if not result.success:
                            self._deferred_error = result.message
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                elif request.HasField("register_tensors"):
                    try:
                        for reg in request.register_tensors.registrations:
                            tensor = self._pending_tensors.pop(reg.storage_id, None)
                            if tensor is not None:
                                self.tensor_manager.register(reg.tensor_id, tensor)
                    except Exception as e:
                        logger.error(f"Error in fire-and-forget operation: {e}")
                        self._deferred_error = str(e)
                else:
                    # Sync operation (get_tensor, get_scalar): yield response
                    if PROFILING_ENABLED:
                        _t0 = time.perf_counter_ns()

                    response = await self._handle_single_request(
                        request, context, server_profiler=_sprof
                    )

                    if PROFILING_ENABLED:
                        _t1 = time.perf_counter_ns()
                        _sprof.sync_handle.add(_t1 - _t0)

                    yield response

                if PROFILING_ENABLED:
                    _t_prev_end = time.perf_counter_ns()
        finally:
            if PROFILING_ENABLED:
                _sprof.stream_end_ns = time.perf_counter_ns()
                _sprof.print_summary()

            # Release all tensor memory when the client disconnects
            self.tensor_manager.clear()
            self._pending_tensors.clear()

    async def _handle_fire_and_forget(
        self,
        request: service_pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> None:
        """Handle a fire-and-forget request. Errors are deferred."""
        try:
            request_type = request.WhichOneof("request")

            if request_type == "execute_aten":
                result = await self.ExecuteAtenOperation(request.execute_aten, context)
                if not result.success:
                    self._deferred_error = result.message

            elif request_type == "delete_tensors":
                result = await self.DeleteTensors(request.delete_tensors, context)
                if not result.success:
                    self._deferred_error = result.message

            elif request_type == "copy_tensor":
                result = await self.CopyTensor(request.copy_tensor, context)
                if not result.success:
                    self._deferred_error = result.message

            elif request_type == "update_tensor":
                result = await self._handle_update_tensor(request.update_tensor)
                if not result.success:
                    self._deferred_error = result.message

            elif request_type == "register_tensors":
                for reg in request.register_tensors.registrations:
                    tensor = self._pending_tensors.pop(reg.storage_id, None)
                    if tensor is not None:
                        self.tensor_manager.register(reg.tensor_id, tensor)

            elif request_type == "batched_execute_aten":
                self._execute_batch_inline(request.batched_execute_aten.operations)

            elif request_type == "raw_execute_aten":
                self._execute_raw_aten_inline(request.raw_execute_aten)

            elif request_type == "raw_batched_execute_aten":
                # Multiple binary ops concatenated, each prefixed with uint32 length
                raw_data = request.raw_batched_execute_aten
                pos = 0
                while pos < len(raw_data):
                    op_len = _STRUCT_I.unpack_from(raw_data, pos)[0]
                    pos += 4
                    op_data = raw_data[pos : pos + op_len]
                    pos += op_len
                    self._execute_raw_aten_inline(op_data)

        except Exception as e:
            logger.error(f"Error in fire-and-forget operation: {e}")
            self._deferred_error = str(e)

    async def _handle_single_request(
        self,
        request: service_pb2.StreamRequest,
        context: grpc.aio.ServicerContext,
        server_profiler=None,
    ) -> service_pb2.StreamResponse:
        """Handle a single sync stream request (get_tensor).

        Checks for deferred errors from prior fire-and-forget operations
        and reports them instead of the normal response.
        """
        response = service_pb2.StreamResponse()

        # Report deferred error from fire-and-forget operations
        if self._deferred_error is not None:
            response.success = False
            response.error_message = self._deferred_error
            self._deferred_error = None
            return response

        try:
            request_type = request.WhichOneof("request")

            if request_type == "get_tensor":
                result = await self._handle_get_tensor(request.get_tensor)
                response.success = result.success
                if not result.success:
                    response.error_message = result.message
                response.get_tensor.CopyFrom(result)

            elif request_type == "get_scalar":
                result = await self._handle_get_scalar(
                    request.get_scalar, server_profiler=server_profiler
                )
                response.success = True
                response.get_scalar.CopyFrom(result)

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
    ) -> None:
        """
        Handle a chunked stream request (fire-and-forget).

        Buffers intermediate chunks. When the final chunk is received,
        processes the assembled request. Errors are deferred.

        Only one chunked request can be in-flight at a time per stream (FIFO).

        Args:
            request: StreamRequest with chunking metadata
            context: gRPC context
            chunk_state: Mutable list holding [buffer, first_request, chunk_size] or [None]
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
            self._deferred_error = "Missing first chunk for chunked request"
            return

        buffer, first_request, chunk_size = chunk_state[0]

        # Write chunk to buffer at correct offset
        start = chunk_number * chunk_size
        buffer[start : start + len(update_req.data)] = update_req.data

        if chunk_number < total_chunks - 1:
            # Intermediate chunk: buffer and continue
            return

        # Last chunk: clear state and process complete request
        chunk_state[0] = None

        response = await self._process_assembled_update_tensor(first_request, buffer, context)
        if not response.success:
            self._deferred_error = response.error_message

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
