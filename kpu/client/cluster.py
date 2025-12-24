"""
Cluster management for multiple Compute resources in parallel.
"""

import asyncio
import logging
from typing import List, Optional, Self

from kpu.client.compute import Compute
from kpu.client.context import compute_ctx


logger = logging.getLogger(__name__)


class Cluster:
    """
    Cluster of Compute resources managed in parallel.

    This class provides an async context manager that orchestrates multiple
    Compute instances, allowing you to manage a cluster of Compute resources
    simultaneously with parallel operations.

    Example:
        >>> # Create a cluster and use all Computes in parallel.
        >>> async with Cluster(
        ...     Compute(name="compute-1", image="my-image:latest"),
        ...     Compute(name="compute-2", image="my-image:latest"),
        ...     Compute(name="compute-3", image="my-image:latest"),
        ... ) as (compute1, compute2, compute3):
        ...     # All computes in the cluster are now ready in parallel
        ...     response1, response2, response3 = await asyncio.gather(
        ...         compute1.send_tensors(tensor),
        ...         compute2.send_tensors(tensor),
        ...         compute3.send_tensors(tensor),
        ...     )
        ...
        >>> # All computes are deleted in parallel when exiting the context

    Args:
        *computes: Variable number of Compute instances to manage
    """

    def __init__(self, *computes: Compute):
        """
        Initialize the Cluster.

        Args:
            *computes: Variable number of Compute instances to manage
        """
        if not computes:
            raise ValueError("At least one Compute instance is required")

        self._computes: List[Compute] = list(computes)
        self._token = None

        logger.debug(f"Initialized Cluster with {len(self._computes)} compute resources")

    def __iter__(self):
        """Iterate over the managed Compute instances."""
        return iter(self._computes)

    def __len__(self) -> int:
        """Return the number of managed Compute instances."""
        return len(self._computes)

    def __getitem__(self, index: int) -> Compute:
        """Get a Compute instance by index."""
        return self._computes[index]

    async def __aenter__(self) -> Self:
        """
        Enter the async context manager.

        Calls __aenter__ on all Compute instances in parallel and waits
        for all of them to be ready.

        Returns:
            Self for use in the async with statement
        """
        logger.debug(f"Entering context for {len(self._computes)} compute resources")

        self._token = compute_ctx.set(self)

        # Enter all Compute contexts in parallel
        await asyncio.gather(
            *[compute.__aenter__() for compute in self._computes]
        )

        logger.info(f"All {len(self._computes)} compute resources are ready")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the async context manager.

        Calls __aexit__ on all Compute instances in parallel to clean up
        resources.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            None (exceptions are not suppressed)
        """
        logger.debug(f"Exiting context for {len(self._computes)} compute resources")

        # Exit all Compute contexts in parallel
        results = await asyncio.gather(
            *[compute.__aexit__(exc_type, exc_val, exc_tb) for compute in self._computes],
            return_exceptions=True
        )

        compute_ctx.reset(self._token)

        # Check if any exits raised exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            logger.error(f"Errors occurred while exiting {len(exceptions)} compute resources")
            raise ExceptionGroup(
                f"Failed to exit {len(exceptions)} compute resource(s)",
                exceptions
            )

        logger.info(f"All {len(self._computes)} compute resources cleaned up")

    def is_ready(self) -> bool:
        """
        Check if all managed Compute instances are ready.

        Returns:
            True if all Computes are ready, False otherwise
        """
        return all(compute.is_ready() for compute in self._computes)

    async def ready(self, timeout: Optional[int] = None) -> None:
        """
        Wait for all managed Compute instances to become ready in parallel.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            RuntimeError: If any compute fails to become ready
        """
        logger.info(f"Waiting for {len(self._computes)} compute resources to be ready")

        async def wait_with_timeout():
            await asyncio.gather(
                *[compute.ready(timeout=timeout) for compute in self._computes]
            )

        if timeout is not None:
            await asyncio.wait_for(wait_with_timeout(), timeout=timeout)
        else:
            await wait_with_timeout()

        logger.info(f"All {len(self._computes)} compute resources are ready")

    async def delete(self) -> None:
        """
        Delete all managed Compute resources in parallel.

        This method can be called manually if not using the async context manager.
        """
        logger.info(f"Deleting {len(self._computes)} compute resources")

        await asyncio.gather(
            *[compute.delete() for compute in self._computes]
        )

        logger.info(f"All {len(self._computes)} compute resources deleted")
