import logging
from os.path import exists, expanduser
from typing import Optional

from kubernetes.client import Configuration
from kubernetes.config import KUBE_CONFIG_DEFAULT_LOCATION, ConfigException
from kubernetes.config import (
    list_kube_config_contexts,
    load_kube_config,
    load_incluster_config,
)

logger = logging.getLogger(__name__)


# Default Kubernetes namespace
_default_namespace = "default"


def default_namespace() -> str:
    """
    Get the default namespace.

    Returns the namespace that will be used for Compute resources. This value
    is set either explicitly via init(namespace=...) or auto-detected from
    kubeconfig context or service account.

    If init() has not been called yet, this function will automatically call it
    to trigger namespace auto-detection. The namespace is auto-detected by reading
    from kubeconfig context or the service account namespace file.

    Returns:
        The default namespace (auto-detected or explicitly set via init())
    """
    # Auto-initialize if not already done
    if Configuration._default is None:
        init()

    return _default_namespace


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
            contexts, active_context = list_kube_config_contexts()

            # Extract namespace from active context
            if active_context and "context" in active_context:
                context_namespace = active_context["context"].get("namespace")
                if context_namespace:
                    _default_namespace = context_namespace
                    logger.debug(f"Using namespace '{_default_namespace}' from kubeconfig context")

        load_kube_config()
        logger.debug("Kubernetes client initialized with default kubeconfig")
        return
    else:
        logger.debug("Default kubeconfig not found, trying in-cluster config")
        try:
            load_incluster_config()
        except ConfigException as e:
            logger.debug(f"In-cluster configuration failed: {e}")
        else:
            if namespace is None:
                namespace_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
                if not exists(namespace_file):
                    raise ConfigException(
                        f"Running in-cluster but service account namespace file not found: "
                        f"{namespace_file}"
                    )

                with open(namespace_file, 'r') as f:
                    ns = f.read().strip()
                    if not ns:
                        raise ConfigException(
                            f"Service account namespace file is empty: {namespace_file}"
                        )
                    _default_namespace = ns
                    logger.debug(f"Using namespace '{_default_namespace}' from service account")

            logger.debug("Kubernetes client initialized with in-cluster config")
            return

    # Initialization failed
    raise ConfigException(
        "Could not configure Kubernetes client. "
        f"Neither default kubeconfig ({KUBE_CONFIG_DEFAULT_LOCATION}) "
        "nor in-cluster config could be loaded."
    )
