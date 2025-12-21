"""
Compute API for creating and managing KPU Compute resources.

This module provides a high-level Python API for creating Compute resources
in Kubernetes and connecting to them via gRPC.
"""

import asyncio
import logging
from os.path import exists, expanduser
from typing import Optional, Dict, List, Callable

try:
    import torch
except ImportError as e:
    raise ImportError(
        f"torch package is required: {e}\n"
        "Install with: pip install torch"
    )

try:
    from kubernetes import config as k8s_config
    from kubernetes import dynamic
    from kubernetes.client import ApiClient, ApiException, Configuration, CoreV1Event, CoreV1Api
    from kubernetes.config import KUBE_CONFIG_DEFAULT_LOCATION
except ImportError as e:
    raise ImportError(
        f"kubernetes package is required: {e}\n"
        "Install with: pip install kubernetes"
    )

from kpu.client.models.kpu_v1alpha1_compute import KpuV1alpha1Compute
from kpu.client.models.kpu_v1alpha1_compute_spec import KpuV1alpha1ComputeSpec
from kpu.client.models.io_k8s_apimachinery_pkg_apis_meta_v1_object_meta import (
    IoK8sApimachineryPkgApisMetaV1ObjectMeta
)
from kpu.client.models.io_k8s_api_core_v1_env_var import IoK8sApiCoreV1EnvVar
from kpu.torch.client import TensorClient

from kpu.client.aio import Watch

import kpu.client.aio as aio

logger = logging.getLogger(__name__)


# Global Kubernetes API namespace
_default_namespace: str = "default"


def init(
    *,
    client_config: Optional[Configuration] = None,
    namespace: Optional[str] = None,
):
    """
    Initialize Kubernetes client configuration globally.

    This is optional. If not called, the client will auto-initialize on first use
    by trying default kubeconfig first, then falling back to in-cluster config.

    The default namespace is also auto-detected (unless explicitly provided):
    - From kubeconfig: Uses the namespace from the current context
    - In-cluster: Reads from /var/run/secrets/kubernetes.io/serviceaccount/namespace
    - Falls back to "default" if no namespace is detected

    Args:
        client_config: Kubernetes client Configuration object. Use this for advanced
                      configuration like custom timeouts, SSL settings, or API server URL.
        namespace: Namespace to use for all Compute resources. If not provided, it will
                  be auto-detected from kubeconfig context or service account.

    Raises:
        RuntimeError: If the client has already been initialized.

    Example:
        >>> from kubernetes import client
        >>> from kpu.client import init
        >>>
        >>> # Advanced configuration with custom namespace
        >>> config = client.Configuration()
        >>> config.host = "https://my-k8s-cluster.example.com"
        >>> config.api_key_prefix['authorization'] = 'Bearer'
        >>> config.api_key['authorization'] = 'my-token'
        >>> init(client_config=config, namespace="my-namespace")
        >>>
        >>> # Or just override the namespace
        >>> init(namespace="my-namespace")
        >>>
        >>> # Or let it auto-detect (default kubeconfig -> in-cluster)
        >>> # No need to call init() at all, it will happen lazily
    """
    global _default_namespace

    if Configuration._default is not None:
        raise RuntimeError(
            "Kubernetes client is already initialized. "
            "The init() function can only be called once before creating any Compute instances."
        )

    # If namespace is explicitly provided, use it and skip auto-detection
    if namespace is not None:
        _default_namespace = namespace
        logger.debug(f"Using explicitly provided namespace '{_default_namespace}'")

    if client_config is not None:
        # Use provided configuration
        Configuration.set_default(client_config)
        logger.debug("Kubernetes client initialized with provided configuration")
        return

    # Auto-detect: try default kubeconfig first, then fall back to in-cluster
    if exists(expanduser(KUBE_CONFIG_DEFAULT_LOCATION)):
        # Load kubeconfig and get the current context
        if namespace is None:
            contexts, active_context = k8s_config.list_kube_config_contexts()

            # Extract namespace from active context
            if active_context and "context" in active_context:
                context_namespace = active_context["context"].get("namespace")
                if context_namespace:
                    _default_namespace = context_namespace
                    logger.debug(f"Using namespace '{_default_namespace}' from kubeconfig context")

        k8s_config.load_kube_config()
        logger.debug("Kubernetes client initialized with default kubeconfig")
        return
    else:
        logger.debug("Default kubeconfig not found, trying in-cluster config")
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException as e:
            logger.debug(f"In-cluster configuration failed: {e}")
        else:
            if namespace is None:
                namespace_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
                if not exists(namespace_file):
                    raise k8s_config.ConfigException(
                        f"Running in-cluster but service account namespace file not found: "
                        f"{namespace_file}"
                    )

                with open(namespace_file, 'r') as f:
                    ns = f.read().strip()
                    if not ns:
                        raise k8s_config.ConfigException(
                            f"Service account namespace file is empty: {namespace_file}"
                        )
                    _default_namespace = ns
                    logger.debug(f"Using namespace '{_default_namespace}' from service account")

            logger.debug("Kubernetes client initialized with in-cluster config")
            return

    # Initialization failed
    raise k8s_config.ConfigException(
        "Could not configure Kubernetes client. "
        f"Neither default kubeconfig ({KUBE_CONFIG_DEFAULT_LOCATION}) "
        "nor in-cluster config could be loaded."
    )


class Compute:
    """
    High-level client for managing KPU Compute resources.

    This class provides an async API for:
    - Creating/updating Compute resources in Kubernetes (using server-side apply)
    - Waiting for them to become ready
    - Streaming tensors to/from the Compute via gRPC
    - Cleaning up resources

    The Compute resource is created or updated in Kubernetes when the object
    is instantiated using server-side apply.

    Configuration is handled globally via the init() function, which auto-configures
    on first use:
    - Tries default kubeconfig first (~/.kube/config)
    - Falls back to in-cluster configuration (when running in a pod)
    - Auto-detects namespace from context or can be explicitly set

    Example:
        >>> # Create a Compute resource (auto-configures Kubernetes client and namespace)
        >>> compute = Compute(
        ...     name="my-compute",
        ...     image="localhost:5001/kpu-torch-server:latest"
        ... )
        >>>
        >>> # Wait for it to be ready and use it
        >>> async with compute:
        ...     tensors = await compute.receive_tensors(count=1)
        ...     await compute.send_tensors(tensor1, tensor2)
        >>>
        >>> # Or override the namespace globally
        >>> from kpu.client import init
        >>> init(namespace="my-namespace")
        >>> compute = Compute(name="my-compute", image="...")
    """

    # API configuration
    API_GROUP = "compute.kpu.dev"
    API_VERSION = "v1alpha1"
    PLURAL = "computes"
    FIELD_MANAGER = "kpu-python-client"

    # Condition types
    CONDITION_READY = "Ready"
    CONDITION_SUSPENDED = "Suspended"

    def __init__(
        self,
        name: str,
        *,
        image: Optional[str] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        suspend: bool = False,
        host: Optional[str] = None,
        port: int = 50051,
        on_events: Optional[Callable[[CoreV1Event], None]] = None,
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
            labels: Labels to apply to the Compute resources
            annotations: Annotations to apply to the Compute resources
            suspend: Whether to create the Compute in suspended state
            host: Override the gRPC host (default: use service DNS)
            port: gRPC port (default: 50051)
            on_events: Optional callback to receive Events for this Compute resource
        """
        self.name = name
        self.namespace = _default_namespace
        self._image = image
        self._command = command
        self._args = args
        self._env = env
        self._labels = labels
        self._annotations = annotations
        self._suspend = suspend
        self._host_override = host
        self._port = port
        self._on_events = on_events

        # Initialize Kubernetes client if needed
        if Configuration._default is None:
            init()

        # Kubernetes client APIs
        self._api_client = ApiClient()
        self._dynamic_client = dynamic.DynamicClient(self._api_client)
        self._compute_api = self._dynamic_client.resources.get(
            api_version="compute.kpu.dev/v1alpha1",
            kind="Compute"
        )

        # State tracking
        self._compute_resource: Optional[KpuV1alpha1Compute] = None
        self._grpc_client: Optional[TensorClient] = None
        self._event_watch_task: Optional[asyncio.Task] = None

        # Start event watching if callback is provided
        if self._on_events is not None:
            self._event_watch_task = asyncio.create_task(self._watch_events())

        # Apply the Compute resource using server-side apply
        self._apply_compute()

    async def ready(
            self,
            timeout: int = 60,
    ) -> "Compute":
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

        async with aio.ApiClient() as api_client, aio.DynamicClient(api_client) as dynamic_client:
            compute_api = await dynamic_client.resources.get(
                api_version="compute.kpu.dev/v1alpha1",
                kind="Compute"
            )
            async with Watch(api_client) as watcher:
                async for event in compute_api.watch(
                        namespace=self.namespace,
                        field_selector=f"metadata.name={self.name}",
                        timeout=timeout,
                        watcher=watcher,
                ):
                    event_type = event["type"]
                    obj = event["object"]

                    # Update our cached resource
                    self._compute_resource = KpuV1alpha1Compute.from_dict(obj.to_dict())

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
            self._compute_resource = KpuV1alpha1Compute.from_dict(result.to_dict())

            # Check if ready
            return self._is_ready()

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"Compute {self.namespace}/{self.name} not found")
                return False
            else:
                logger.error(f"Failed to get Compute status: {e}")
                raise

    async def suspend(self) -> "Compute":
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

    async def resume(self) -> "Compute":
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

        logger.info(f"Deleting Compute {self.namespace}/{self.name}")

        try:
            self._compute_api.delete(
                name=self.name,
                namespace=self.namespace,
                body={
                    "gracePeriodSeconds": grace_period_seconds
                }
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

        spec = KpuV1alpha1ComputeSpec(
            image=self._image,
            command=self._command,
            args=self._args,
            env=env_vars,
            labels=self._labels,
            annotations=self._annotations,
            suspend=self._suspend,
        )

        metadata = IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=self.name,
            namespace=self.namespace,
        )

        compute = KpuV1alpha1Compute(
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

            self._compute_resource = KpuV1alpha1Compute.from_dict(result.to_dict())
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

        async with aio.ApiClient() as api_client:
            core_api = CoreV1Api(api_client)
            async with Watch(api_client).\
                    stream(core_api.list_namespaced_event,
                           namespace=self.namespace,
                           field_selector=f"involvedObject.name={self.name},involvedObject.kind=Compute",
                           send_initial_events=False,
                           resource_version_match="NotOlderThan",
                           ) as stream:
                try:
                    async for event in stream:
                        if event["type"] in ["ADDED", "MODIFIED"]:
                            self._on_events(event["object"])
                except asyncio.CancelledError:
                    pass

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

        # Use provided host or construct from service
        if self._host_override is None:
            host = f"{self.name}.{self.namespace}.svc.cluster.local"
        else:
            host = self._host_override

        logger.info(
            f"Initializing gRPC connection to Compute {self.namespace}/{self.name} "
            f"at {host}:{self._port}"
        )

        self._grpc_client = TensorClient(host=host, port=self._port,
                                         metadata=[("compute-id", f"{self.namespace}/{self.name}")])
        # Enter the async context manager
        await self._grpc_client.__aenter__()

    async def _cleanup_grpc_client(self):
        """Cleanup the gRPC client connection."""
        if self._grpc_client is not None:
            await self._grpc_client.__aexit__(None, None, None)
            self._grpc_client = None

    # Delegate gRPC tensor streaming methods to TensorClient

    async def send_tensors(self, *tensors: torch.Tensor):
        """
        Send tensors to the Compute server.

        Args:
            *tensors: Tensors to send

        Returns:
            Response from server

        Raises:
            RuntimeError: If the Compute is not ready
        """
        if self._grpc_client is None:
            raise RuntimeError(
                f"Compute {self.namespace}/{self.name} is not ready. "
                "Call ready() first or use as context manager."
            )
        return await self._grpc_client.send_tensors(*tensors)

    async def receive_tensors(
        self,
        count: int = 1,
        parameters: dict = None
    ) -> List[torch.Tensor]:
        """
        Receive tensors from the Compute server.

        Args:
            count: Number of tensors to request
            parameters: Optional parameters for the request

        Returns:
            List of received tensors

        Raises:
            RuntimeError: If the Compute is not ready
        """
        if self._grpc_client is None:
            raise RuntimeError(
                f"Compute {self.namespace}/{self.name} is not ready. "
                "Call ready() first or use as context manager."
            )
        return await self._grpc_client.receive_tensors(count=count, parameters=parameters)

    async def stream_tensors(
        self,
        *tensors: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Bidirectional streaming: send tensors and receive processed tensors.

        Args:
            *tensors: Tensors to send

        Returns:
            List of processed tensors received from server

        Raises:
            RuntimeError: If the Compute is not ready
        """
        if self._grpc_client is None:
            raise RuntimeError(
                f"Compute {self.namespace}/{self.name} is not ready. "
                "Call ready() first or use as context manager."
            )
        return await self._grpc_client.stream_tensors(*tensors)

    async def __aenter__(self) -> "Compute":
        """
        Async context manager entry: wait for ready.
        """
        await self.ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit: delete the Compute.
        """
        await self.delete()
