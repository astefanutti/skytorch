"""
KV cache utilities for speculative scalar with HuggingFace Transformers.

Provides a DynamicCache subclass that supports cropping after speculation
overshoot, including for models with sliding window attention.
"""

from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer


class _CropSafeLayer(DynamicSlidingWindowLayer):
    """DynamicSlidingWindowLayer that supports crop() after the window fills.

    The base class forbids crop() once cumulative_length >= sliding_window,
    but removing recent tail entries is always safe (evicted head entries
    aren't involved). Also computes masks from actual stored key count
    instead of the fixed ``sliding_window - 1`` formula — identical during
    normal operation (stored = sliding_window - 1), but correct after crop.
    """

    def crop(self, max_length):
        if max_length < 0:
            max_length = self.keys.shape[-2] + max_length
        stored = self.keys.shape[-2]
        if stored <= max_length:
            return
        trimmed = stored - max_length
        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]
        self.cumulative_length = max(0, self.cumulative_length - trimmed)

    def get_mask_sizes(self, cache_position):
        query_length = cache_position.shape[0]
        if not self.is_initialized:
            return query_length, 0
        stored = self.keys.shape[-2]
        kv_length = stored + query_length
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        return kv_length, kv_offset


class SpeculativeCache(DynamicCache):
    """DynamicCache with crop-safe sliding window layers.

    Replaces ``DynamicSlidingWindowLayer`` instances with ``_CropSafeLayer``
    so that speculation overshoot entries can be removed after generation
    via :meth:`crop_overshoot`.

    Usage::

        cache = SpeculativeCache(config=model.config)
        output = model.generate(..., past_key_values=cache)

        overshoot = ...  # number of tokens beyond the valid stopping point
        if overshoot > 0:
            cache.crop_overshoot(overshoot)
    """

    def __init__(self, config):
        super().__init__(config=config)
        for i, layer in enumerate(self.layers):
            if type(layer) is DynamicSlidingWindowLayer:
                safe = _CropSafeLayer(sliding_window=layer.sliding_window)
                self.layers[i] = safe

    def crop_overshoot(self, overshoot):
        """Remove the last ``overshoot`` entries from all cache layers."""
        for layer in self.layers:
            if not layer.is_initialized:
                continue
            stored = layer.keys.shape[-2]
            if overshoot > 0 and stored > overshoot:
                layer.crop(stored - overshoot)
