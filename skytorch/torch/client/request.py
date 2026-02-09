"""
Request building utilities for SkyTorch gRPC operations.

This module provides functions to build protobuf request messages
for tensor operations and ATen execution.
"""

from typing import Optional

import torch

from skytorch.torch.server import service_pb2
from skytorch.torch.client.tensor import get_tensor_id
from skytorch.torch.client.metadata import TensorMetadata


def tensor_metadata_to_proto(
    metadata: TensorMetadata,
) -> service_pb2.TensorMetadata:
    """Convert TensorMetadata to proto message."""
    proto = service_pb2.TensorMetadata(
        tensor_id=metadata.tensor_id,
        shape=list(metadata.shape),
        dtype=str(metadata.dtype),
        nbytes=metadata.nbytes,
        device_type=metadata.device_type,
        stride=list(metadata.stride) if metadata.stride else [],
        storage_offset=metadata.storage_offset,
        device_index=metadata.device_index,
    )
    if metadata.tensor_ref is not None:
        proto.tensor_ref = metadata.tensor_ref
    return proto


def to_aten_arg(value) -> service_pb2.AtenArgument:
    """Convert a value to AtenArgument proto.

    Handles:
    - None → none_value
    - torch.Tensor (sky) → TensorReference with tensor_id
    - torch.Tensor (cpu, scalar) → scalar value
    - bool/int/float/str → scalar values
    - torch.device → string
    - torch.dtype → scalar_dtype
    - list/tuple → recursive AtenArgumentList
    """
    arg = service_pb2.AtenArgument()

    if value is None:
        arg.none_value = True
    elif isinstance(value, torch.Tensor):
        if value.device.type == "sky":
            arg.tensor.CopyFrom(
                service_pb2.TensorReference(tensor_id=get_tensor_id(value))
            )
        elif value.device.type == "cpu" and value.dim() == 0:
            # cpu scalar tensor → convert to python scalar
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
                f"Only sky tensors and 0-dim cpu scalars are allowed."
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
    elif isinstance(value, torch.layout):
        arg.scalar_layout = str(value)
    elif isinstance(value, (list, tuple)):
        # Handle nested lists/tuples recursively
        list_arg = service_pb2.AtenArgumentList()
        list_arg.is_tuple = isinstance(value, tuple)
        for item in value:
            list_arg.values.append(to_aten_arg(item))
        arg.list_value.CopyFrom(list_arg)
    else:
        raise ValueError(f"Unsupported ATen argument type: {type(value)}")

    return arg


def build_copy_tensor_request(
    src_tensor_id: int,
    dst_tensor_id: int,
    src_offset: int = 0,
    dst_offset: int = 0,
    num_bytes: int = -1,
    src_metadata: Optional[TensorMetadata] = None,
    dst_metadata: Optional[TensorMetadata] = None,
) -> service_pb2.CopyTensorRequest:
    """
    Build a CopyTensorRequest proto message.

    Args:
        src_tensor_id: Source tensor ID
        dst_tensor_id: Destination tensor ID
        src_offset: Byte offset in source storage
        dst_offset: Byte offset in destination storage
        num_bytes: Number of bytes to copy (-1 for all)
        src_metadata: Optional metadata for auto-creating source tensor
        dst_metadata: Optional metadata for auto-creating destination tensor

    Returns:
        CopyTensorRequest proto message
    """
    request = service_pb2.CopyTensorRequest(
        src_tensor_id=src_tensor_id,
        dst_tensor_id=dst_tensor_id,
        src_offset=src_offset,
        dst_offset=dst_offset,
        num_bytes=num_bytes,
    )

    if src_metadata is not None:
        request.src_metadata.CopyFrom(tensor_metadata_to_proto(src_metadata))
    if dst_metadata is not None:
        request.dst_metadata.CopyFrom(tensor_metadata_to_proto(dst_metadata))

    return request


def build_execute_aten_request(
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_tensors: list[torch.Tensor] | None,
    tensor_metadata: Optional[list[TensorMetadata]] = None,
    output_metadata: Optional[list[TensorMetadata]] = None,
) -> service_pb2.ExecuteAtenRequest:
    """
    Build an ExecuteAtenRequest proto message.

    Args:
        op_name: ATen operation name
        args: Positional arguments
        kwargs: Keyword arguments
        output_tensors: Pre-allocated output tensors
        tensor_metadata: Input tensor metadata for auto-creation
        output_metadata: Output tensor metadata for auto-creation

    Returns:
        ExecuteAtenRequest proto message
    """
    request = service_pb2.ExecuteAtenRequest(
        op_name=op_name,
        args=[to_aten_arg(arg) for arg in args],
    )

    for k, v in kwargs.items():
        request.kwargs[k].CopyFrom(to_aten_arg(v))

    if output_tensors is not None:
        for t in output_tensors:
            if t is not None:
                request.outputs.append(
                    service_pb2.TensorReference(tensor_id=get_tensor_id(t))
                )

    if tensor_metadata is not None:
        for meta in tensor_metadata:
            request.tensor_metadata.append(tensor_metadata_to_proto(meta))
    if output_metadata is not None:
        for meta in output_metadata:
            request.output_metadata.append(tensor_metadata_to_proto(meta))

    return request
