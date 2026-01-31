"""
End-to-end tests for client configuration.
"""

import pytest
from kubernetes import client as k8s_client

from kpu.client import init, Compute
import kpu.client.init

@pytest.fixture(autouse=True)
def reset_client():
    """Reset client state before each test."""
    k8s_client.Configuration._default = None
    kpu.client.init._default_namespace = "default"

    yield

    # Clean up after test
    k8s_client.Configuration._default = None
    kpu.client.init._default_namespace = "default"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_init_auto(test_image):
    """
    Test automatic client configuration.

    Covers:
    - Auto-detection without calling init()
    - Kubeconfig or in-cluster detection
    - Default namespace detection
    """
    # Don't call init() - let it auto-detect
    async with Compute(
        name="test-init-auto",
        image=test_image,
    ) as compute:
        # Verify compute was created
        assert compute.is_ready()
        assert compute.namespace is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_init_custom_namespace(test_image):
    """
    Test init() with custom namespace.

    Covers:
    - Explicit namespace configuration
    - Namespace override
    """
    # Initialize with specific namespace
    init(namespace="default")

    async with Compute(
        name="test-init-namespace",
        image=test_image,
    ) as compute:
        assert compute.namespace == "default"
        assert compute.is_ready()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_init_custom_config(test_image):
    """
    Test init() with custom Kubernetes client configuration.

    Covers:
    - Custom client configuration
    - Advanced configuration options
    """
    # Create custom config (still using default cluster)
    config = k8s_client.Configuration()
    # In a real scenario, you might set custom host, auth, etc.
    # For E2E, we use the default cluster

    # Load default config into our custom config object
    from kubernetes import config as k8s_config
    k8s_config.load_kube_config(client_configuration=config)

    # Initialize with custom config
    init(client_config=config, namespace="default")

    async with Compute(
        name="test-init-config",
        image=test_image,
    ) as compute:
        assert compute.is_ready()
        assert compute.namespace == "default"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_init_kubeconfig_namespace(test_image):
    """
    Test namespace detection from kubeconfig context.

    Covers:
    - Reading namespace from kubeconfig context
    - Auto-detection without explicit init()
    """
    # Create compute without calling init()
    # Should auto-detect namespace from kubeconfig
    async with Compute(
        name="test-init-kubeconfig-namespace",
        image=test_image,
    ) as compute:
        # Namespace should be detected
        assert compute.namespace is not None
        assert compute.is_ready()


@pytest.mark.e2e
def test_init_double_call():
    """
    Test that calling init() twice raises RuntimeError.

    Covers:
    - init() can only be called once
    - Error handling for double initialization
    """
    # First call should work
    init(namespace="default")

    # Second call should raise
    with pytest.raises(RuntimeError, match="already initialized"):
        init(namespace="default")
