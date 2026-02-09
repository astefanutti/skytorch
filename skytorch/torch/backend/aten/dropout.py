"""
SkyTorch ATen Dropout Operations.

Handle deterministic native_dropout edge cases client-side to avoid
server round-trips that may silently fail in streaming mode:

- p=0 or train=False: returns input unchanged with all-True mask
- p=1 with train=True: returns all-zeros with all-False mask

The general case (0 < p < 1) still delegates to the server via the
standard fallback.
"""

from typing import Optional

import torch

from .dispatch import _sky_kernel_fallback


def _native_dropout(
    input: torch.Tensor, p: float, train: Optional[bool]
) -> tuple[torch.Tensor, torch.Tensor]:
    if train is False or p == 0:
        return (input, torch.ones_like(input, dtype=torch.bool))
    if p == 1:
        return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
    return _sky_kernel_fallback(
        torch.ops.aten.native_dropout.default, input, p, train
    )
