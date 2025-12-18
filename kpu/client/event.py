"""
Event handling utilities for KPU Compute resources.
"""

import logging

try:
    from kubernetes.client import CoreV1Event
except ImportError as e:
    raise ImportError(
        f"kubernetes package is required: {e}\n"
        "Install with: pip install kubernetes"
    )


logger = logging.getLogger(__name__)


def log_event(event: CoreV1Event) -> None:
    """
    Default event callback that logs Compute events.

    This callback can be passed to the Compute constructor's on_events parameter
    to automatically log all events related to the Compute resource.

    Args:
        event: Kubernetes Event object

    Example:
        >>> from kpu.client import Compute, log_event
        >>> compute = Compute(
        ...     name="my-compute",
        ...     image="...",
        ...     on_events=log_event
        ... )
    """
    # Extract event information
    reason = event.reason or "Unknown"
    message = event.message or ""
    event_type = event.type or "Normal"
    timestamp = event.last_timestamp or event.first_timestamp

    # Determine log level based on event type
    if event_type == "Warning":
        log_level = logging.WARNING
    elif event_type == "Error":
        log_level = logging.ERROR
    else:
        log_level = logging.INFO

    # Format the log message
    log_message = f"[{reason}] {message}"
    if timestamp:
        log_message = f"{timestamp} - {log_message}"

    logger.log(log_level, log_message)
