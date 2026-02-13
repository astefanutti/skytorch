"""
Compute API for creating and managing SkyTorch Compute resources.

This module provides a high-level Python API for creating Compute resources
in Kubernetes and connecting to them via gRPC.
"""

import asyncio
import inspect
import logging
import re
import textwrap
from typing import Callable, Dict, List, Optional, Self

try:
    import torch
except ImportError as e:
    raise ImportError(
        f"torch package is required: {e}\n"
        "Install with: pip install torch"
    )

try:
    from kubernetes import dynamic
    from kubernetes.client import ApiClient, ApiException, Configuration, CoreV1Event, CoreV1Api
    from kubernetes.config import KUBE_CONFIG_DEFAULT_LOCATION
except ImportError as e:
    raise ImportError(
        f"kubernetes package is required: {e}\n"
        "Install with: pip install kubernetes"
    )

from skytorch.client.aio import Stream
from skytorch.client.context import compute_ctx
from skytorch.client.init import init, default_namespace
from skytorch.client.state_dict import SkyStateDict
from skytorch.client.models.compute_v1alpha1_compute import ComputeV1alpha1Compute
from skytorch.client.models.compute_v1alpha1_compute_spec import ComputeV1alpha1ComputeSpec
from skytorch.client.models.io_k8s_apimachinery_pkg_apis_meta_v1_object_meta import (
    IoK8sApimachineryPkgApisMetaV1ObjectMeta,
)
from skytorch.client.models.io_k8s_api_core_v1_env_var import IoK8sApiCoreV1EnvVar
from skytorch.client.models.io_k8s_api_core_v1_persistent_volume_claim_spec import (
    IoK8sApiCoreV1PersistentVolumeClaimSpec,
)
from skytorch.client.models.io_k8s_api_core_v1_persistent_volume_claim_template import (
    IoK8sApiCoreV1PersistentVolumeClaimTemplate,
)
from skytorch.client.models.io_k8s_api_core_v1_volume_resource_requirements import (
    IoK8sApiCoreV1VolumeResourceRequirements,
)
from skytorch.client.models.io_k8s_apimachinery_pkg_api_resource_quantity import (
    IoK8sApimachineryPkgApiResourceQuantity,
)

import grpc

from skytorch.client.grpc import GRPCClient

import skytorch.client.aio as aio

logger = logging.getLogger(__name__)


class Compute:
    """
    High-level client for managing SkyTorch Compute resources.

    This class provides an async API for:
    - Creating/updating Compute resources in Kubernetes (using server-side apply)
    - Waiting for them to become ready
    - Streaming tensors to/from the Compute via gRPC
    - Streaming metrics from the Compute via gRPC
    - Cleaning up resources

    The Compute resource is created or updated in Kubernetes when the object
    is instantiated using server-side apply.

    Configuration is handled globally via the init() function, which auto-configures
    on first use:
    - Tries default kubeconfig first (~/.kube/config)
    - Falls back to in-cluster configuration (when running in a pod)
    - Auto-detects namespace from context or can be explicitly set

    Example:
        >>> def handle_metrics(snapshot):
        ...     for metric in snapshot.metrics:
        ...         print(f"{metric.name}: {metric.value} {metric.unit}")
        >>>
        >>> compute = Compute(
        ...     name="my-compute",
        ...     image="localhost:5001/skytorch-server:latest",
        ...     resources={"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"},
        ...     on_metrics=handle_metrics
        ... )
        >>>
        >>> # Wait for it to be ready and use it
        >>> async with compute:
        ...     tensors = await compute.receive_tensors(count=1)
        ...     await compute.send_tensors(tensor1, tensor2)
        >>>
    """

    # API configuration
    API_GROUP = "compute.skytorch.dev"
    API_VERSION = "v1alpha1"
    PLURAL = "computes"
    FIELD_MANAGER = "skytorch-python-client"

    # Condition types
    CONDITION_READY = "Ready"
    CONDITION_SUSPENDED = "Suspended"

    # Device string parsing pattern
    _DEVICE_PATTERN = re.compile(r"^([a-zA-Z_]+)(?::(\d+))?$")

    def __init__(
        self,
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
        volumes: Optional[List[Dict[str, str]]] = None,
        on_events: Optional[Callable[[CoreV1Event], None]] = None,
        on_metrics: Optional[Callable[[object], None]] = None,
    ):
        """
        Initialize and create/update a Compute resource using server-side apply.

        The namespace is determined by the global init() configuration, which auto-detects
        from kubeconfig context or in-cluster service account, or can be explicitly set.

        Args:
            name: Name of the Compute resource
            image: Container image for the Compute runtime
            command: Entrypoint command override
            args: Arguments for the entrypoint
            env: Environment variables as dict (will be converted to EnvVar list)
            resources: Resource requirements as dict (e.g., {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"})
            labels: Labels to apply to the Compute resources
            annotations: Annotations to apply to the Compute resources
            suspend: Whether to create the Compute in suspended state
            volumes: Simplified volume definitions as list of dicts with keys:
                     name (volume name), storage (e.g. "10Gi"), path (mount path)
            on_events: Optional callback to receive events for this Compute resource
            on_metrics: Optional callback to receive metrics from this Compute resource
        """
        self.name = name
        self._image = image
        self._command = command
        self._args = args
        self._env = env
        self._resources = resources
        self._labels = labels
        self._annotations = annotations
        self._suspend = suspend
        self._volumes = volumes
        self._on_events = on_events
        self._on_metrics = on_metrics
        self._ctx_token = None

        # Initialize Kubernetes client if needed
        if Configuration._default is None:
            init()

        # Global namespace is defaulted in init()
        self.namespace = default_namespace()

        # Kubernetes client APIs
        self._api_client = ApiClient()
        self._dynamic_client = dynamic.DynamicClient(self._api_client)
        self._compute_api = self._dynamic_client.resources.get(
            api_version="compute.skytorch.dev/v1alpha1",
            kind="Compute",
        )

        # State tracking
        self._compute_resource: Optional[ComputeV1alpha1Compute] = None
        self._grpc_client: Optional[GRPCClient] = None
        self._event_watch_task: Optional[asyncio.Task] = None
        self._metrics_stream_task: Optional[asyncio.Task] = None

        # Start event watching if callback is provided
        if self._on_events is not None:
            self._event_watch_task = asyncio.create_task(self._watch_events())

        # Apply the Compute resource using server-side apply
        self._apply_compute()

    @property
    def resource(self) -> Optional[ComputeV1alpha1Compute]:
        """
        Get the current Compute resource.

        Returns the Kubernetes Compute resource object that represents the current
        state of this Compute, including metadata, spec, and status.

        Returns:
            The Compute resource object, or None if not yet fetched from Kubernetes

        Example:
            >>> compute = Compute(name="my-compute", image="my-image:latest")
            >>> resource = compute.resource
            >>> if resource and resource.status:
            ...     for condition in resource.status.conditions:
            ...         print(f"{condition.type}={condition.status}")
        """
        return self._compute_resource

    async def ready(
        self,
        timeout: int = 60,
    ) -> Self:
        """
        Wait for the Compute to become ready using watch.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Self for method chaining

        Raises:
            TimeoutError: If the Compute doesn't become ready within timeout
        """
        logger.info(
            f"Waiting for Compute {self.namespace}/{self.name} to be ready "
            f"(timeout: {timeout}s)"
        )

        async with aio.WsApiClient() as api_client, aio.DynamicClient(api_client) as dynamic_client:
            compute_api = await dynamic_client.resources.get(
                api_version="compute.skytorch.dev/v1alpha1",
                kind="Compute",
            )
            async with Stream(api_client) as watcher:
                async for event in compute_api.watch(
                    namespace=self.namespace,
                    field_selector=f"metadata.name={self.name}",
                    timeout=timeout,
                    watcher=watcher,
                ):
                    event_type = event["type"]
                    obj = event["object"]

                    # Handle deletion
                    if event_type == "DELETED":
                        raise TimeoutError(
                            f"Compute {self.namespace}/{self.name} was deleted"
                        )

                    # Update our cached resource
                    self._compute_resource = ComputeV1alpha1Compute.from_dict(obj.to_dict())

                    # Log current status
                    if self._compute_resource.status and self._compute_resource.status.conditions:
                        conditions = self._compute_resource.status.conditions
                        status_msg = ", ".join(
                            f"{c.type}={c.status}" for c in conditions
                        )
                        logger.debug(
                            f"Compute {self.namespace}/{self.name} event={event_type} "
                            f"status: {status_msg}"
                        )

                    # Check if ready
                    if self._is_ready():
                        logger.info(f"Compute {self.namespace}/{self.name} is ready")
                        break

                # The watch timed out
                if not self.is_ready():
                    raise TimeoutError(
                        f"Compute {self.namespace}/{self.name} did not become ready "
                        f"within {timeout} seconds"
                    )

        # Initialize gRPC client when ready
        await self._init_grpc_client()
        return self

    def is_ready(self) -> bool:
        """
        Check if the Compute is ready by fetching the current status from Kubernetes.

        Returns:
            True if the Compute is ready, False otherwise
        """
        try:
            # Fetch the current resource from Kubernetes
            result = self._compute_api.get(
                name=self.name,
                namespace=self.namespace,
            )

            # Update our cached resource
            self._compute_resource = ComputeV1alpha1Compute.from_dict(result.to_dict())

            # Check if ready
            return self._is_ready()

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"Compute {self.namespace}/{self.name} not found")
                return False
            else:
                logger.error(f"Failed to get Compute status: {e}")
                raise

    async def suspend(self) -> Self:
        """
        Suspend the Compute (scale to zero).

        Returns:
            Self for method chaining
        """
        logger.info(f"Suspending Compute {self.namespace}/{self.name}")

        # Cleanup gRPC client when suspending
        await self._cleanup_grpc_client()

        # Update suspend field and reapply
        self._suspend = True
        self._apply_compute()

        return self

    async def resume(self) -> Self:
        """
        Resume a suspended Compute.

        Returns:
            Self for method chaining
        """
        logger.info(f"Resuming Compute {self.namespace}/{self.name}")

        # Update suspend field and reapply
        self._suspend = False
        self._apply_compute()

        # Wait for it to be ready again and reinitialize gRPC
        await self.ready()
        return self

    async def delete(self, grace_period_seconds: int = 30) -> None:
        """
        Delete the Compute resource from Kubernetes.

        Args:
            grace_period_seconds: Grace period for deletion
        """
        # Cancel event watch task if running
        if self._event_watch_task is not None:
            self._event_watch_task.cancel()
            await self._event_watch_task

        # Cleanup gRPC client
        await self._cleanup_grpc_client()

        # Remove device mappings for this Compute
        from skytorch.torch.backend._device import device_manager

        device_manager.remove_compute_devices(self)

        logger.info(f"Deleting Compute {self.namespace}/{self.name}")

        try:
            self._compute_api.delete(
                name=self.name,
                namespace=self.namespace,
                body={
                    "gracePeriodSeconds": grace_period_seconds,
                },
            )

            logger.info(f"Compute {self.namespace}/{self.name} deleted")
            self._compute_resource = None

        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Compute {self.namespace}/{self.name} not found, "
                    "already deleted"
                )
            else:
                logger.error(f"Failed to delete Compute: {e}")
                raise

    def _apply_compute(self) -> None:
        """
        Create or update the Compute resource using server-side apply.
        """
        logger.info(f"Applying Compute {self.namespace}/{self.name}")

        # Build the Compute resource specification
        # Convert env dict to EnvVar list
        env_vars = None
        if self._env:
            env_vars = [
                IoK8sApiCoreV1EnvVar(name=k, value=v)
                for k, v in self._env.items()
            ]

        # Convert resources dict to ResourceList
        resource_list = None
        if self._resources:
            resource_list = {
                k: IoK8sApimachineryPkgApiResourceQuantity(v)
                for k, v in self._resources.items()
            }

        # Convert volumes to PVC templates and volume mount overrides
        volume_claim_templates = None
        if self._volumes:
            volume_claim_templates = [
                IoK8sApiCoreV1PersistentVolumeClaimTemplate(
                    metadata=IoK8sApimachineryPkgApisMetaV1ObjectMeta(name=v["name"]),
                    spec=IoK8sApiCoreV1PersistentVolumeClaimSpec(
                        access_modes=["ReadWriteOnce"],
                        resources=IoK8sApiCoreV1VolumeResourceRequirements(
                            requests={
                                "storage": IoK8sApimachineryPkgApiResourceQuantity(v["storage"])
                            }
                        ),
                    ),
                )
                for v in self._volumes
            ]
            volume_mounts = [
                {"name": v["name"], "mountPath": v["path"]} for v in self._volumes
            ]
            # Merge volume mounts into override
            override = {}
            containers = override.get("containers", [])
            # Find or create the server container entry
            server_container = None
            for c in containers:
                if c.get("name") == "server":
                    server_container = c
                    break
            if server_container is None:
                server_container = {"name": "server"}
                containers = list(containers) + [server_container]
            existing_mounts = server_container.get("volumeMounts", [])
            server_container["volumeMounts"] = list(existing_mounts) + volume_mounts
            override = {**override, "containers": containers}

        spec = ComputeV1alpha1ComputeSpec(
            image=self._image,
            command=self._command,
            args=self._args,
            env=env_vars,
            resources=resource_list,
            labels=self._labels,
            annotations=self._annotations,
            suspend=self._suspend,
            volume_claim_templates=volume_claim_templates,
            override=override,
        )

        metadata = IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=self.name,
            namespace=self.namespace,
        )

        compute = ComputeV1alpha1Compute(
            api_version=f"{self.API_GROUP}/{self.API_VERSION}",
            kind="Compute",
            metadata=metadata,
            spec=spec,
        )

        body = compute.to_dict()

        try:
            result = self._compute_api.server_side_apply(
                body=body,
                name=self.name,
                namespace=self.namespace,
                field_manager=self.FIELD_MANAGER,
                force=True,
            )

            self._compute_resource = ComputeV1alpha1Compute.from_dict(result.to_dict())
            logger.info(
                f"Compute {self.namespace}/{self.name} applied successfully "
                "(created or updated)"
            )

        except ApiException as e:
            logger.error(f"Failed to apply Compute: {e}")
            raise

    async def _watch_events(self):
        """Watch for Events related to this Compute resource and call the callback."""
        logger.debug(f"Starting event watch for Compute {self.namespace}/{self.name}")

        async with aio.WsApiClient() as api_client:
            core_api = CoreV1Api(api_client)
            async with Stream(api_client).stream(
                core_api.list_namespaced_event,
                namespace=self.namespace,
                field_selector="involvedObject.kind=Pod",
                send_initial_events=False,
                resource_version_match="NotOlderThan",
            ) as stream:
                try:
                    async for event in stream:
                        # FIXME: find a better way to only watch for owned Pods events
                        if event["type"] in ["ADDED", "MODIFIED"]:
                            if event["raw_object"]["metadata"]["name"].startswith(self.name):
                                self._on_events(event["object"])
                except asyncio.CancelledError:
                    pass

    async def _stream_metrics(self):
        """Stream metrics from this Compute resource and call the callback."""
        logger.debug(f"Starting metrics streaming for Compute {self.namespace}/{self.name}")

        try:
            # Stream metrics continuously with 1 second interval
            async for snapshot in self._grpc_client.metrics.stream_metrics(
                metric_names=[
                    "gpu.utilization.compute",
                    "gpu.utilization.memory",
                    "gpu.memory.used",
                    "gpu.power.usage",
                ],
                interval_seconds=1.0,
            ):
                logger.debug(
                    f"Received metrics snapshot from {snapshot.source} "
                    f"with {len(snapshot.metrics)} metrics"
                )
                self._on_metrics(snapshot)
        except asyncio.CancelledError:
            logger.debug(f"Metrics streaming cancelled for Compute {self.namespace}/{self.name}")
        except Exception as e:
            logger.error(f"Error streaming metrics for Compute {self.namespace}/{self.name}: {e}")

    def _is_ready(self) -> bool:
        """Check if the Compute is ready based on conditions."""
        if not self._compute_resource or not self._compute_resource.status:
            return False

        conditions = self._compute_resource.status.conditions
        if not conditions:
            return False

        # Check for Ready=True condition
        for condition in conditions:
            if condition.type == self.CONDITION_READY and condition.status == "True":
                return True

        # Also check that it's not suspended
        for condition in conditions:
            if condition.type == self.CONDITION_SUSPENDED and condition.status == "True":
                return False

        return False

    async def _init_grpc_client(self):
        """Initialize the gRPC client connection."""
        if self._grpc_client is not None:
            # Already initialized
            return

        # Extract host from Compute resource addresses
        host = "localhost"  # Default to localhost
        port = 50051  # Default gRPC port

        if (
            self._compute_resource
            and self._compute_resource.status
            and self._compute_resource.status.addresses
            and len(self._compute_resource.status.addresses) > 0
        ):
            # Use the first address from the Gateway
            address = self._compute_resource.status.addresses[0]
            host = address.value
            logger.debug(
                f"Using address from Compute status: {host} "
                f"(type: {address.type})"
            )
        else:
            logger.info(
                f"No addresses found in Compute {self.namespace}/{self.name} status, "
                f"defaulting to localhost"
            )

        logger.info(
            f"Initializing gRPC connection to Compute {self.namespace}/{self.name} "
            f"at {host}:{port}"
        )

        self._grpc_client = GRPCClient(
            host=host,
            port=port,
            metadata=[("compute-id", f"{self.namespace}/{self.name}")],
        )
        # Enter the async context manager
        await self._grpc_client.__aenter__()

        # Start metrics streaming if callback is provided
        if self._on_metrics is not None:
            # asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
            self._metrics_stream_task = asyncio.create_task(self._stream_metrics())

    async def _cleanup_grpc_client(self):
        """Cleanup the gRPC client connection."""
        # Cancel metrics stream task if running
        if self._metrics_stream_task is not None:
            self._metrics_stream_task.cancel()
            try:
                await self._metrics_stream_task
            except asyncio.CancelledError:
                pass
            self._metrics_stream_task = None

        if self._grpc_client is not None:
            await self._grpc_client.__aexit__(None, None, None)
            self._grpc_client = None

    def device(self, type: str, index: Optional[int] = None) -> torch.device:
        """
        Get a torch.Device instance for this Compute resource.

        Uses the DeviceManager to map this Compute and remote device to a
        local sky device index.

        Args:
            type: Remote device type, optionally with index (e.g., "cuda", "cuda:0", "cpu")
            index: Remote device index (default: 0). Cannot be specified if type
                   already contains an index.

        Returns:
            torch.device instance of type "sky" with mapped local index

        Raises:
            RuntimeError: If type contains an index and index is also passed explicitly,
                          or if the device string format is invalid

        Example:
            >>> compute = Compute(name="my-compute")
            >>> device = compute.device("cuda")      # Same as cuda:0
            >>> device = compute.device("cuda:1")    # Uses index 1
            >>> device = compute.device("cuda", 1)   # Same as cuda:1
            >>> tensor = torch.randn(3, 3, device=device)
        """
        from skytorch.torch.backend._device import device_manager

        # Validate and parse device string
        match = self._DEVICE_PATTERN.match(type)
        if not match:
            raise RuntimeError(
                f"Invalid device string: {type!r}. "
                f"Expected format: 'device_type' or 'device_type:index'"
            )

        device_type = match.group(1)
        parsed_index = match.group(2)

        if parsed_index is not None:
            if index is not None:
                raise RuntimeError(
                    f"type (string) must not include an index because index was "
                    f"passed explicitly: {type}"
                )
            device_index = int(parsed_index)
        else:
            device_index = index if index is not None else 0

        return device_manager.get_sky_device(self, device_type, device_index)

    @staticmethod
    def _generate_imports(fn) -> str:
        """Generate import statements for global references used by a function."""
        import types

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

    @staticmethod
    def _get_callable_source(fn) -> tuple[str, str] | None:
        """Extract source code and name from a callable, or None if not possible."""
        if getattr(fn, "__name__", "") == "<lambda>":
            return None

        if fn.__code__.co_freevars:
            return None  # Has closure variables — can't represent as source

        try:
            source = inspect.getsource(fn)
            source = textwrap.dedent(source)
        except (OSError, TypeError):
            return None

        imports = Compute._generate_imports(fn)
        if imports:
            source = imports + "\n\n" + source

        return source, fn.__name__

    async def execute(self, fn, *args, **kwargs):
        """
        Execute a function on the remote server and return sky tensors.

        For regular functions, the source code is sent as text (version-independent).
        Falls back to cloudpickle for lambdas, closures, or when source is unavailable.
        Any tensors in the result (e.g., model state_dict) stay on the GPU server;
        only metadata is returned. Client-side sky tensors are created to reference
        the remote storage.

        Args:
            fn: Callable to execute on the server
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable

        Returns:
            dict[str, torch.Tensor]: Dictionary of sky tensors referencing
                remote GPU storage

        Raises:
            RuntimeError: If execution fails or gRPC client is not connected
        """
        import cloudpickle
        import pickle

        from skytorch.torch.backend._C import _create_remote_tensor
        from skytorch.torch.backend._storage import storage_manager
        from skytorch.torch.client.tensor import get_tensor_id
        from skytorch.torch.server import service_pb2

        if self._grpc_client is None:
            raise RuntimeError(
                "gRPC client is not connected. " "Use 'async with compute:' or call ready() first."
            )

        # 1. Serialize and call unary RPC
        source_info = self._get_callable_source(fn)
        try:
            if source_info:
                source, name = source_info
                response = await self._grpc_client.torch.execute_function(
                    b"",
                    pickle.dumps(args),
                    pickle.dumps(kwargs),
                    callable_source=source,
                    callable_name=name,
                )
            else:
                response = await self._grpc_client.torch.execute_function(
                    cloudpickle.dumps(fn),
                    pickle.dumps(args),
                    pickle.dumps(kwargs),
                )
        except grpc.aio.AioRpcError as e:
            raise RuntimeError(
                f"Failed to execute function: {e.code().name}: {e.details()}"
            ) from e

        if not response.success:
            raise RuntimeError(f"Remote execution failed: {response.error_message}")

        # 2. Create sky tensors from metadata
        from skytorch.torch.backend._device import device_manager

        sky_state_dict = {}
        registrations = []
        seen_storage: dict[int, torch.Tensor] = {}

        for info in response.tensors:
            sky_device_index = device_manager.get_sky_device(
                self, info.device_type, info.device_index
            ).index

            if info.storage_id in seen_storage:
                # Shared storage (weight tying) — create view from existing sky tensor
                base = seen_storage[info.storage_id]
                sky_tensor = base.as_strided(
                    list(info.shape), list(info.stride), info.storage_offset
                )
            else:
                sky_tensor = _create_remote_tensor(
                    info.storage_id,
                    list(info.shape),
                    info.dtype,
                    list(info.stride),
                    info.storage_offset,
                    info.storage_nbytes,
                    sky_device_index,
                )
                seen_storage[info.storage_id] = sky_tensor

            tensor_id = get_tensor_id(sky_tensor)
            storage_manager.register_storage(
                info.storage_id, info.storage_nbytes, sky_device_index
            )
            storage_manager.register_tensor(sky_tensor)
            registrations.append(
                service_pb2.TensorRegistration(storage_id=info.storage_id, tensor_id=tensor_id)
            )
            sky_state_dict[info.name] = sky_tensor

        # 4. Send mapping to server via stream (fire-and-forget)
        self._grpc_client.stream.submit_register_tensors(
            service_pb2.RegisterTensorsRequest(registrations=registrations)
        )

        logger.info(
            f"Received {len(sky_state_dict)} tensors from remote execution "
            f"({len(seen_storage)} unique storages)"
        )

        return SkyStateDict(sky_state_dict)

    async def __aenter__(self) -> Self:
        """
        Async context manager entry: wait for ready.
        """
        if not compute_ctx.get(None):
            self._ctx_token = compute_ctx.set(self)

        await self.ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit: delete the Compute.
        """
        await self.delete()

        if self._ctx_token is not None:
            compute_ctx.reset(self._ctx_token)
