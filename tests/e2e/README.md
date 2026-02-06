## SkyTorch End-to-End Tests

This directory contains end-to-end (E2E) tests for the SkyTorch Python client that validate functionality against a real Kubernetes cluster.

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
   pip install -e ".[test,torch]"
   ```

### Setup KinD Cluster (Optional)

If you don't have a cluster, you can create one using KinD:

```bash
# Create KinD cluster with custom configuration
kind create cluster --config test/e2e/kind.yaml --name skytorch-test

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
pytest test/e2e -v -m e2e
```

#### Run Specific Test File

```bash
# Compute tests
pytest test/e2e/test_compute_e2e.py -v

# Cluster tests
pytest test/e2e/test_cluster_e2e.py -v

# Init tests
pytest test/e2e/test_init_e2e.py -v
```

#### Run Specific Test

```bash
pytest test/e2e/test_compute_e2e.py::test_compute_managed -v
```

#### Run with Detailed Output

```bash
pytest test/e2e -v -s -m e2e
```

The `-s` flag disables output capturing, allowing you to see print statements and logs.

#### Skip Slow Tests

```bash
pytest test/e2e -v -m "e2e and not slow"
```

### Test Markers

- `e2e`: Tests that require a Kubernetes cluster (all tests in this directory)
- `slow`: Tests that take longer to run (not currently used but available)

### Environment Configuration

Tests use fixtures for configuration, defined in `test/e2e/conftest.py`:

- **test_image**: Container image for Compute resources (default: `ghcr.io/astefanutti/skytorch-server`)

The gRPC endpoint is automatically discovered from the Compute resource status (Gateway addresses).

#### Option 1: Environment Variables

Set environment variables before running tests:

```bash
export SKYTORCH_TEST_IMAGE="your-custom-image:tag"
pytest test/e2e -v -m e2e
```

Or inline:

```bash
SKYTORCH_TEST_IMAGE="your-custom-image:tag" pytest test/e2e -v -m e2e
```

#### Option 2: Edit Fixtures

For persistent local configuration, edit the fixtures in `test/e2e/conftest.py`:

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

### CI/CD Integration

These tests are designed to run in CI/CD pipelines with a Kubernetes cluster available. See `.github/workflows/build-and-run-e2e-tests.yaml` for the complete workflow.

Key steps in CI/CD:

```yaml
# Example GitHub Actions workflow steps
- name: Setup KinD cluster
  uses: helm/kind-action@v1.13.0
  with:
    cluster_name: skytorch
    config: test/e2e/kind.yaml

- name: Install Gateway API CRDs
  run: kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.4.1/experimental-install.yaml

- name: Build and push images
  run: |
    # Build and push operator and PyTorch server images to local registry

- name: Deploy SkyTorch operator
  run: kubectl apply -f config/e2e

- name: Install test dependencies
  run: pip install -e ".[test,torch]"

- name: Run E2E tests
  run: |
    SKYTORCH_TEST_IMAGE="${IMAGE_TAG}" pytest test/e2e -v -m e2e
```

### Test Structure

Each test follows this pattern:

1. **Setup**: Create Compute/Cluster resources
2. **Verify**: Check resources are ready
3. **Execute**: Perform operations (send/receive tensors, etc.)
4. **Assert**: Validate results
5. **Cleanup**: Delete resources (automatic with context manager)

Tests use `async with` context managers for automatic cleanup, ensuring resources are deleted even if tests fail.

### Adding New Tests

To add new E2E tests:

1. Create test functions with `@pytest.mark.e2e` and `@pytest.mark.asyncio`
2. Use fixtures for configuration (test_image) and common setup (test_tensors, etc.)
3. Follow naming convention: `test_<feature>_<scenario>`
4. Include docstrings describing what is covered
5. Ensure proper cleanup (use context managers or manual delete)

Example:

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_new_feature(test_image):
    """
    Test description.

    Covers:
    - Feature 1
    - Feature 2
    """
    async with Compute(name="test-new", image=test_image) as compute:
        # Test code here
        assert compute.is_ready()
```

### Performance Considerations

- Tests create real Kubernetes resources and wait for pods to start
- Each Compute typically takes 10-60 seconds to become ready
- Parallel tests in Cluster can take 1-2 minutes
- Consider using `pytest-xdist` for parallel test execution:

```bash
pip install pytest-xdist
pytest test/e2e -v -m e2e -n auto
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
   pytest test/e2e/test_compute_e2e.py::test_compute_managed -v -s
   ```

4. **Add breakpoints** (using pytest's built-in debugger):
   ```python
   import pdb; pdb.set_trace()
   ```

   Then run with:
   ```bash
   pytest test/e2e/test_compute_e2e.py::test_compute_managed -v -s
   ```
