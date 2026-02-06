## SkyTorch End-to-End Tests

This directory contains end-to-end (E2E) tests for the SkyTorch Python client that validate functionality against a real Kubernetes cluster. These tests are run in CI via the `.github/workflows/build-and-run-tests.yaml` GitHub Actions workflow.

### Prerequisites

1. **Kubernetes Cluster**
   - KinD cluster or any Kubernetes cluster
   - SkyTorch operator deployed and running
   - Gateway API installed (for GRPCRoute support)

2. **Container Images**
   - `skytorch-server` image available in the cluster
   - Default test image: `ghcr.io/astefanutti/skytorch-server`

3. **kubectl Configuration**
   - kubectl configured with access to the cluster
   - Default context set appropriately
   - Sufficient permissions to create/delete Compute resources

4. **Python Dependencies**
   ```bash
   pip install -e ".[test]"
   ```

### Setup KinD Cluster (Optional)

If you don't have a cluster, you can create one using KinD:

```bash
# Create KinD cluster with custom configuration
kind create cluster --config tests/e2e/kind.yaml --name skytorch-test

# Load the skytorch-server image into the cluster
docker pull ghcr.io/astefanutti/skytorch-server
kind load docker-image ghcr.io/astefanutti/skytorch-server --name skytorch-test

# Deploy SkyTorch
kubectl apply -f config/e2e

# Verify operator is running
kubectl get pods -n skytorch-system
```

### Running Tests

#### Run All E2E Tests

```bash
pytest tests/e2e -v -m e2e
```

#### Run Specific Test File

```bash
# Compute tests
pytest tests/e2e/test_compute_e2e.py -v

# Cluster tests
pytest tests/e2e/test_cluster_e2e.py -v

# Init tests
pytest tests/e2e/test_init_e2e.py -v
```

#### Run Specific Test

```bash
pytest tests/e2e/test_compute_e2e.py::test_compute_managed -v
```

#### Run with Detailed Output

```bash
pytest tests/e2e -v -s -m e2e
```

The `-s` flag disables output capturing, allowing you to see print statements and logs.

#### Skip Slow Tests

```bash
pytest tests/e2e -v -m "e2e and not slow"
```

### Test Markers

- `e2e`: Tests that require a Kubernetes cluster (all tests in this directory)
- `slow`: Tests that take longer to run (not currently used but available)

### Environment Configuration

Tests use fixtures for configuration, defined in `tests/e2e/conftest.py`:

- **test_image**: Container image for Compute resources (default: `ghcr.io/astefanutti/skytorch-server`)

The gRPC endpoint is automatically discovered from the Compute resource status (Gateway addresses).

#### Option 1: Environment Variables

Set environment variables before running tests:

```bash
export SKYTORCH_TEST_IMAGE="your-custom-image:tag"
pytest tests/e2e -v -m e2e
```

Or inline:

```bash
SKYTORCH_TEST_IMAGE="your-custom-image:tag" pytest tests/e2e -v -m e2e
```

#### Option 2: Edit Fixtures

For persistent local configuration, edit the fixtures in `tests/e2e/conftest.py`:

```python
@pytest.fixture(scope="session")
def test_image():
    """Get the test image for Compute resources."""
    return os.getenv("SKYTORCH_TEST_IMAGE", "your-custom-image:tag")
```

**Namespace Configuration**: Tests will auto-detect the namespace from your kubeconfig context or use "default".

Or configure the cluster's Gateway/Ingress for external access.

### Troubleshooting

#### Tests Fail with Connection Errors

- Verify Kubernetes cluster is accessible: `kubectl get nodes`
- Check SkyTorch operator is running: `kubectl get pods -n skytorch-system`
- Verify port-forwarding is active (if testing externally)

#### Tests Fail with Image Pull Errors

- Ensure `skytorch-server` image is available in the cluster
- For KinD: Load image with `kind load docker-image`
- For other clusters: Push to accessible registry

#### Tests Timeout Waiting for Ready

- Check if pods are starting: `kubectl get pods`
- Check pod events: `kubectl describe pod <pod-name>`
- Verify image can be pulled
- Check resource limits/requests are satisfiable

#### Resource Cleanup Issues

- Tests should clean up automatically
- Manual cleanup: `kubectl delete computes --all`
- Check for stuck finalizers: `kubectl get computes -o yaml`

### Performance Considerations

- Tests create real Kubernetes resources and wait for pods to start
- Each Compute typically takes 10-60 seconds to become ready
- Parallel tests in Cluster can take 1-2 minutes
- Consider using `pytest-xdist` for parallel test execution:

```bash
pip install pytest-xdist
pytest tests/e2e -v -m e2e -n auto
```

Note: Be careful with parallel execution as it can create many resources simultaneously.

### Debugging Failed Tests

To debug a failed test:

1. **Check Kubernetes resources**:
   ```bash
   kubectl get computes
   kubectl get pods
   kubectl describe compute <name>
   ```

2. **View logs**:
   ```bash
   kubectl logs <pod-name>
   ```

3. **Run single test with output**:
   ```bash
   pytest tests/e2e/test_compute_e2e.py::test_compute_managed -v -s
   ```

4. **Add breakpoints** (using pytest's built-in debugger):
   ```python
   import pdb; pdb.set_trace()
   ```

   Then run with:
   ```bash
   pytest tests/e2e/test_compute_e2e.py::test_compute_managed -v -s
   ```
