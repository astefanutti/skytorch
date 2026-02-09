"""Shared test helpers for operation correctness tests."""

import torch


def assert_forward_correct(cpu_fn, cpu_inputs, device, atol=1e-5, rtol=1.3e-6):
    """Run an operation on CPU and sky, compare results.

    Args:
        cpu_fn: Callable that takes inputs and returns result(s).
        cpu_inputs: List of CPU tensors (or non-tensor args) to pass.
        device: SkyTorch device.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """
    # Build sky inputs: move tensors to device, keep non-tensors as-is
    sky_inputs = []
    for inp in cpu_inputs:
        if isinstance(inp, torch.Tensor):
            sky_inputs.append(inp.to(device))
        else:
            sky_inputs.append(inp)

    cpu_result = cpu_fn(*cpu_inputs)
    sky_result = cpu_fn(*sky_inputs)

    _compare_results(cpu_result, sky_result, atol=atol, rtol=rtol)


def assert_grad_correct(fn, cpu_inputs, device, atol=1e-5, rtol=1.3e-6):
    """Run backward on CPU and sky, compare all gradients.

    Args:
        fn: Callable that takes inputs and returns a scalar loss.
        cpu_inputs: List of CPU tensors with requires_grad=True for those
            that should be differentiated.
        device: SkyTorch device.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """
    # Build sky inputs
    sky_inputs = []
    for inp in cpu_inputs:
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            sky_inputs.append(inp.clone().to(device).detach().requires_grad_(True))
        elif isinstance(inp, torch.Tensor):
            sky_inputs.append(inp.to(device))
        else:
            sky_inputs.append(inp)

    # CPU forward + backward
    cpu_loss = fn(*cpu_inputs)
    cpu_loss.backward()

    # Sky forward + backward
    sky_loss = fn(*sky_inputs)
    sky_loss.backward()

    # Compare gradients for all grad-requiring inputs
    cpu_grad_idx = 0
    for cpu_inp, sky_inp in zip(cpu_inputs, sky_inputs):
        if isinstance(cpu_inp, torch.Tensor) and cpu_inp.requires_grad:
            assert sky_inp.grad is not None, (
                f"Gradient is None for input {cpu_grad_idx} on sky device. "
                f"This typically means the operation has an explicit PrivateUse1 "
                f"dispatch key registration that overrides the CompositeImplicitAutograd "
                f"decomposition, breaking gradient tracking. "
                f"Fix: remove the explicit registration so PyTorch uses its "
                f"built-in autograd decomposition."
            )
            torch.testing.assert_close(
                sky_inp.grad.cpu(),
                cpu_inp.grad,
                atol=atol,
                rtol=rtol,
                check_device=False,
            )
            cpu_grad_idx += 1


def _compare_results(cpu_result, sky_result, atol, rtol):
    """Recursively compare CPU and sky results."""
    if cpu_result is None:
        assert sky_result is None, f"Expected None, got {type(sky_result)}"
        return

    if isinstance(cpu_result, torch.Tensor):
        assert isinstance(sky_result, torch.Tensor), (
            f"Expected Tensor, got {type(sky_result)}"
        )
        torch.testing.assert_close(
            sky_result.cpu(), cpu_result, atol=atol, rtol=rtol, check_device=False
        )
        return

    if isinstance(cpu_result, (tuple, list)):
        assert len(cpu_result) == len(sky_result), (
            f"Result length mismatch: CPU={len(cpu_result)}, sky={len(sky_result)}"
        )
        for i, (c, s) in enumerate(zip(cpu_result, sky_result)):
            _compare_results(c, s, atol=atol, rtol=rtol)
        return

    # Scalar or other type — direct comparison
    assert cpu_result == sky_result, f"Mismatch: CPU={cpu_result}, sky={sky_result}"
