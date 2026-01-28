"""
KPU ATen Dynamic Operations - Handlers for data-dependent output shapes.

This module provides implementations for ATen operations that have
data-dependent output shapes (e.g., masked_select, nonzero). These
operations cannot use meta tensor execution for shape inference and
require computing the actual output size at runtime.
"""

from typing import Any

import torch

from kpu.torch.backend._async import run_async
from kpu.torch.backend import _client


def _handle_masked_select(
        op: torch._ops.OpOverload,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
) -> Any:
    """Handle masked_select with data-dependent output shape."""
    # Extract self tensor and mask from args
    self_tensor = args[0]
    mask = args[1]

    # Compute output size by counting True values in mask
    # This executes on the KPU server via the normal dispatch path
    output_size = int(mask.sum().cpu().item())

    # Create output tensor with the correct size
    output = torch.empty(
        output_size,
        dtype=self_tensor.dtype,
        device=self_tensor.device,
    )

    # Execute the operation with the pre-allocated output
    run_async(
        _client.execute_aten_operation(
            kpu_device=self_tensor.device,
            op_name=str(op),
            args=args,
            kwargs=kwargs,
            output_tensors=[output],
        )
    )

    return output


def _masked_select(self_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Handle masked_select with data-dependent output shape."""
    return _handle_masked_select(
        torch.ops.aten.masked_select.default,
        (self_tensor, mask),
        {},
    )


def _masked_select_out(
    self_tensor: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Handle masked_select.out with data-dependent output shape."""
    return _handle_masked_select(
        torch.ops.aten.masked_select.out,
        (self_tensor, mask),
        {"out": out},
    )
