"""
Utility functions for KPU client operations.
"""

from typing import Any, Awaitable, Callable


def map_args_kwargs(
    func: Callable[[Any], Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Apply func to all elements in args/kwargs, recursing into lists/tuples.

    The func should handle leaf values (non-containers). Recursion into
    lists and tuples is handled by this function.

    Args:
        func: Transformer function for leaf values
        args: Positional arguments to transform
        kwargs: Keyword arguments to transform

    Returns:
        Transformed (args, kwargs) tuple
    """

    def transform(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            return type(obj)(transform(item) for item in obj)
        return func(obj)

    return (
        tuple(transform(arg) for arg in args),
        {k: transform(v) for k, v in kwargs.items()},
    )


async def async_map_args_kwargs(
    func: Callable[[Any], Awaitable[Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Async version of map_args_kwargs.

    Apply an async func to all elements in args/kwargs, recursing into lists/tuples.

    Args:
        func: Async transformer function for leaf values
        args: Positional arguments to transform
        kwargs: Keyword arguments to transform

    Returns:
        Transformed (args, kwargs) tuple
    """

    async def transform(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            transformed = [await transform(item) for item in obj]
            return type(obj)(transformed)
        return await func(obj)

    transformed_args = tuple([await transform(arg) for arg in args])
    transformed_kwargs = {k: await transform(v) for k, v in kwargs.items()}
    return transformed_args, transformed_kwargs
