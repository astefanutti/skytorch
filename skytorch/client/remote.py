"""Utilities for extracting and preparing callable source code for remote execution."""

import inspect
import textwrap
import types


def generate_imports(fn) -> str:
    """Generate import statements for global references used by a function."""
    lines = []
    seen = set()
    for name in fn.__code__.co_names:
        if name in seen or name not in fn.__globals__:
            continue
        seen.add(name)
        obj = fn.__globals__[name]
        if isinstance(obj, types.ModuleType):
            if obj.__name__ != name:
                lines.append(f"import {obj.__name__} as {name}")
            else:
                lines.append(f"import {obj.__name__}")
        elif hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
            module = getattr(obj, "__module__", None)
            if module and module != "builtins":
                lines.append(f"from {module} import {name}")
    return "\n".join(lines)


def get_callable_source(fn) -> tuple[str, str]:
    """Extract source code and name from a callable.

    Returns:
        A tuple of (source_code, function_name).

    Raises:
        TypeError: If the callable cannot be serialized for remote execution.
    """
    if getattr(fn, "__name__", "") == "<lambda>":
        raise TypeError(
            "Lambdas cannot be serialized for remote execution. " "Use a named function instead."
        )

    if fn.__code__.co_freevars:
        raise TypeError(
            "Closures cannot be serialized for remote execution. "
            "Move captured variables to function arguments."
        )

    try:
        source = inspect.getsource(fn)
        source = textwrap.dedent(source)
    except (OSError, TypeError) as e:
        raise TypeError(
            f"Cannot extract source code for {fn!r}. Functions defined in "
            "interactive sessions or C extensions are not supported."
        ) from e

    imports = generate_imports(fn)
    if imports:
        source = imports + "\n\n" + source

    return source, fn.__name__
