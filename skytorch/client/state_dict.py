import torch


class SkyStateDict(dict):
    """State dict of sky tensors returned by Compute.execute()."""

    def load_into(self, model: torch.nn.Module):
        """Load into a model, handling both state_dict and non-persistent buffers."""
        state_dict_keys = set(model.state_dict().keys())
        persistent = {k: v for k, v in self.items() if k in state_dict_keys}
        model.load_state_dict(persistent, assign=True)
        # Assign non-persistent buffers (not in state_dict, e.g. inv_freq)
        for name, tensor in self.items():
            if name not in state_dict_keys:
                *parts, attr = name.split(".")
                mod = model
                for p in parts:
                    mod = getattr(mod, p)
                setattr(mod, attr, tensor)
