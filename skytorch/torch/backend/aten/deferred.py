"""
SkyTorch Deferred Scalar Tensor.

A torch.Tensor subclass that wraps a zero-filled CPU tensor and a
concurrent.futures.Future. Overrides tolist() and item() to block
on the future and fill data before delegating to the real tensor.

Used by async copy to make .cpu() non-blocking for single-element
tensors during LLM generation.
"""

from __future__ import annotations

from concurrent.futures import Future

import torch


class DeferredScalarTensor(torch.Tensor):
    """Tensor subclass that lazily resolves its scalar value from a Future.

    Created via _make_subclass to share storage with a real CPU tensor,
    so .shape, .device, .dtype, .dim() all work natively through C++.
    The actual value is resolved on first access via tolist() or item().

    Uses a class-level dict keyed by id(self) because _make_subclass
    instances don't support __slots__ or reliable per-instance __dict__.
    """

    _futures: dict[int, Future] = {}

    @staticmethod
    def create(shape: tuple | list, dtype: torch.dtype, future: Future) -> DeferredScalarTensor:
        """Create a tensor backed by a zero-filled CPU tensor.

        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            future: Future that resolves to the scalar value

        Returns:
            DeferredScalarTensor wrapping the backing tensor
        """
        backing = torch.zeros(shape, dtype=dtype)
        t = torch.Tensor._make_subclass(DeferredScalarTensor, backing)
        DeferredScalarTensor._futures[id(t)] = future
        return t

    def _resolve(self):
        """Block on the future and fill the backing tensor with the resolved value."""
        future = DeferredScalarTensor._futures.pop(id(self), None)
        if future is not None:
            self.fill_(future.result())

    def tolist(self):
        self._resolve()
        return super().tolist()

    def item(self):
        self._resolve()
        return super().item()

    def __del__(self):
        DeferredScalarTensor._futures.pop(id(self), None)
