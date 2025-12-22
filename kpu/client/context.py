"""
Context variable management for KPU Compute resources.

This module provides context variables that allow access to the current Compute
or Cluster instance within an async context, enabling implicit access without
explicit parameter passing.

The context variables are set automatically when entering a Compute or Cluster
async context manager, and reset when exiting.

Example:
    >>> from kpu.client import Compute
    >>> from kpu.client.context import compute_ctx
    >>>
    >>> async with Compute(name="my-compute") as compute:
    ...     # Inside the context, compute_ctx contains the Compute instance
    ...     current = compute_ctx.get()
    ...     assert current is compute
"""

import contextvars

# Context variable that holds the current Compute or Cluster instance.
# This is automatically set when entering a Compute or Cluster async context manager.
# Use compute_ctx.get() to retrieve the current instance, or compute_ctx.get(default)
# to provide a default value if not set.
compute_ctx = contextvars.ContextVar('compute_ctx')
