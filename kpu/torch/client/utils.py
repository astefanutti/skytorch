"""
Utility functions for KPU client operations.
"""

from typing import Any, Callable


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
