"""
KPU ATen Copy Operations.

This module implements copy operations between KPU and other devices.
Copy operations need explicit implementation because they involve
data transfer between devices.
"""

import torch

from kpu.torch.backend._driver import driver


def _copy_from_device(tensor: torch.Tensor) -> torch.Tensor:
    """Copy data from KPU tensor to CPU tensor.

    Args:
        tensor: Source KPU tensor

    Returns:
        CPU tensor with copied data
    """
    if tensor.device.type != "kpu":
        raise ValueError("_copy_from_device requires a KPU tensor")

    # Get storage ID from tensor
    storage_id = _get_storage_id(tensor)

    # TODO: Fetch data from remote via driver when gRPC integration is ready
    # For now, create an empty CPU tensor with the same shape
    # This will work for testing the structure but won't have actual data
    #
    # data = driver.fetch_tensor_data(storage_id, tensor.shape, tensor.dtype)
    # return torch.frombuffer(data, dtype=tensor.dtype).reshape(tensor.shape)

    # Placeholder: create uninitialized CPU tensor
    cpu_tensor = torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device="cpu",
    )
    return cpu_tensor


def _copy_to_device(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to KPU tensor.

    Args:
        src: Source CPU tensor
        dst: Destination KPU tensor

    Returns:
        Destination tensor (same as dst)
    """
    if dst.device.type != "kpu":
        raise ValueError("_copy_to_device requires a KPU target tensor")
    if src.device.type != "cpu":
        raise ValueError("_copy_to_device requires a CPU source tensor")

    # Get storage ID from destination tensor
    storage_id = _get_storage_id(dst)

    # TODO: Upload data to remote via driver when gRPC integration is ready
    #
    # driver.upload_tensor_data(storage_id, src.numpy().tobytes())

    return dst


def _copy_kpu_to_kpu(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy data between KPU tensors.

    Args:
        src: Source KPU tensor
        dst: Destination KPU tensor
    """
    if src.device.type != "kpu" or dst.device.type != "kpu":
        raise ValueError("_copy_kpu_to_kpu requires KPU tensors")

    src_storage_id = _get_storage_id(src)
    dst_storage_id = _get_storage_id(dst)

    # TODO: Remote copy via driver when gRPC integration is ready
    #
    # driver.copy_storage(src_storage_id, dst_storage_id, src.numel() * src.element_size())


def _get_storage_id(tensor: torch.Tensor) -> int:
    """Get the storage ID from a KPU tensor.

    The storage ID is stored as the data pointer in KPU tensors.
    """
    # The data pointer is actually the storage ID cast to void*
    data_ptr = tensor.data_ptr()
    return data_ptr


def _copy_from(
    from_: torch.Tensor,
    to_: torch.Tensor,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy data from one tensor to another, handling KPU device transfers.

    This function implements the core copy operation for KPU tensors,
    supporting CPU<->KPU transfers and KPU<->KPU copies.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to
        non_blocking: Whether to perform the copy asynchronously (currently ignored)

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting unsupported copy operations
    """
    if from_.device.type == "kpu" and to_.device.type == "cpu":
        # KPU to CPU
        host_mem = _copy_from_device(from_)
        return to_.copy_(host_mem)

    elif from_.device.type == "cpu" and to_.device.type == "kpu":
        # CPU to KPU
        return _copy_to_device(from_, to_)

    elif from_.device.type == "kpu" and to_.device.type == "kpu":
        # KPU to KPU
        _copy_kpu_to_kpu(from_, to_)
        return to_

    else:
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} "
            f"is not supported. Only CPU<->KPU transfers and KPU<->KPU copies "
            f"are allowed."
        )
