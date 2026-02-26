import fnmatch
import logging

import torch

logger = logging.getLogger(__name__)


class StateDict(dict):
    """State dict of sky tensors returned by Compute.execute().

    Attributes:
        model_id: Server-assigned model ID when retain_model=True was used.
        _compute: Reference to the Compute instance (for proxying).
    """

    model_id: int = 0
    _compute = None

    def load_into(
        self,
        model: torch.nn.Module,
        triton_modules: list[str] | None = None,
    ):
        """Load into a model, handling both state_dict and non-persistent buffers.

        Args:
            model: The model to load weights into.
            triton_modules: Optional list of module path patterns to proxy via
                ExecuteModuleForward RPC. Supports wildcards (e.g.,
                "model.layers.*.experts"). Requires that the StateDict was
                created with retain_model=True.
        """
        state_dict_keys = set(model.state_dict().keys())
        persistent = {k: v for k, v in self.items() if k in state_dict_keys}

        if triton_modules:
            if not self.model_id:
                raise RuntimeError(
                    "triton_modules requires retain_model=True in execute(). "
                    "The model was not retained on the server."
                )
            if self._compute is None:
                raise RuntimeError(
                    "triton_modules requires a Compute reference. "
                    "The StateDict is missing its _compute attribute."
                )

            # Resolve which module paths will be proxied
            all_module_paths = {name for name, _ in model.named_modules() if name}
            proxied_paths = set()
            for pattern in triton_modules:
                for path in all_module_paths:
                    if fnmatch.fnmatch(path, pattern):
                        proxied_paths.add(path)

            # Find state_dict keys that belong to proxied modules — these weights
            # live on the server and may not be extractable as torch.Tensor
            # (e.g., MXFP4 custom Triton tensor objects).
            proxied_keys = set()
            for key in state_dict_keys:
                for prefix in proxied_paths:
                    if key.startswith(prefix + ".") or key == prefix:
                        proxied_keys.add(key)
                        break

            missing = state_dict_keys - set(persistent.keys())
            unexpected_missing = missing - proxied_keys
            if unexpected_missing:
                raise RuntimeError(
                    f"Missing keys not covered by triton_modules: {sorted(unexpected_missing)}"
                )

            if proxied_keys:
                logger.debug(
                    f"Skipping {len(proxied_keys)} state_dict keys " f"belonging to proxied modules"
                )

            model.load_state_dict(persistent, assign=True, strict=False)

            # Detect compute dtype from loaded sky tensors so proxied modules
            # (whose parameters were never loaded) use the correct dtype for
            # meta shape prediction instead of the default float32.
            compute_dtype = None
            for v in persistent.values():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    compute_dtype = v.dtype
                    break

            from skytorch.torch.client.proxy import proxy_triton_modules

            proxy_triton_modules(
                model, self._compute, self.model_id, triton_modules, compute_dtype
            )
        else:
            model.load_state_dict(persistent, assign=True)

        # Assign non-persistent buffers (not in state_dict, e.g. inv_freq)
        for name, tensor in self.items():
            if name not in state_dict_keys:
                *parts, attr = name.split(".")
                mod = model
                for p in parts:
                    mod = getattr(mod, p)
                setattr(mod, attr, tensor)
