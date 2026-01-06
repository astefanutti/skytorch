"""
Decorators for KPU Compute resources.

This module provides decorators to simplify working with Compute resources,
automatically managing their lifecycle.
"""

import functools
import inspect
from typing import Callable, Dict, List, Optional

try:
    from kubernetes.client import CoreV1Event
except ImportError as e:
    raise ImportError(
        f"kubernetes package is required: {e}\n"
        "Install with: pip install kubernetes"
    )


def compute(
    name: str,
    *,
    image: Optional[str] = None,
    command: Optional[List[str]] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    resources: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    suspend: bool = False,
    on_events: Optional[Callable[[CoreV1Event], None]] = None,
    on_metrics: Optional[Callable[[object], None]] = None,
):
    """
    Decorator that automatically manages a Compute instance lifecycle.

    Creates a Compute instance with the provided parameters, enters its async
    context (waiting for it to be ready), passes it to the decorated function,
    and ensures cleanup on exit (deletes the Compute resource).

    Args:
        name: Name of the Compute resource
        image: Container image for the Compute runtime
        command: Entrypoint command override
        args: Arguments for the entrypoint
        env: Environment variables as dict
        resources: Resource requirements as dict
        labels: Labels to apply to the Compute resources
        annotations: Annotations to apply to the Compute resources
        suspend: Whether to create the Compute in suspended state
        on_events: Optional callback to receive events for this Compute resource
        on_metrics: Optional callback to receive metrics from this Compute resource

    Returns:
        Decorator function that wraps async functions

    Example:
        >>> @compute(
        ...     name="my-compute",
        ...     image="localhost:5001/kpu-torch-server:latest",
        ...     resources={"nvidia.com/gpu": "1"}
        ... )
        ... async def process_data(compute: Compute):
        ...     tensors = await compute.receive_tensors(count=1)
        ...     # Process tensors...
        ...     result = tensors[0] * 2
        ...     await compute.send_tensors(result)
        >>>
        >>> await process_data()
    """
    def decorator(func):
        # Import here to avoid circular import
        from kpu.client.compute import Compute

        # Inspect the function signature to find a Compute parameter
        sig = inspect.signature(func)
        compute_param_name = None

        for param_name, param in sig.parameters.items():
            if param.annotation is not inspect.Parameter.empty:
                # Check if the annotation is Compute or references Compute
                if param.annotation is Compute or (
                    hasattr(param.annotation, '__name__') and
                    param.annotation.__name__ == 'Compute'
                ):
                    compute_param_name = param_name
                    break

        @functools.wraps(func)
        async def wrapper(*func_args, **func_kwargs):
            compute_instance = Compute(
                name=name,
                image=image,
                command=command,
                args=args,
                env=env,
                resources=resources,
                labels=labels,
                annotations=annotations,
                suspend=suspend,
                on_events=on_events,
                on_metrics=on_metrics,
            )

            # Enter context manager (waits for ready and initializes gRPC)
            # Exit will cleanup (delete the Compute resource)
            async with compute_instance as c:
                if compute_param_name:
                    # Pass as keyword argument to the typed parameter
                    func_kwargs[compute_param_name] = c
                    return await func(*func_args, **func_kwargs)
                else:
                    # Fallback to positional argument (first position)
                    return await func(c, *func_args, **func_kwargs)

        return wrapper
    return decorator
