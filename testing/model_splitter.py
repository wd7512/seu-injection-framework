# model_splitter.py

import torch
import torch.nn as nn
from typing import Tuple


class ModelSplitter:
    def __init__(self, model: nn.Module, sample_input: torch.Tensor):
        self.model = model.eval()
        self.sample_input = sample_input

        with torch.no_grad():
            self.baseline_output = self.model(self.sample_input)

        self.forward_modules = self._get_ordered_modules()

    def _get_ordered_modules(self):
        if isinstance(self.model, nn.Sequential):
            return list(self.model._modules.items())
        elif hasattr(self.model, 'net') and isinstance(self.model.net, nn.Sequential):
            return list(self.model.net._modules.items())
        else:
            raise NotImplementedError("Only supports models or model.net of type nn.Sequential")

    def _map_params_to_layers(self):
        param_to_module_idx = {}
        modules = self.forward_modules
        net_prefix = "net." if hasattr(self.model, 'net') else ""

        for idx, (mod_name, layer) in enumerate(modules):
            for pname, _ in layer.named_parameters():
                full_name = f"{net_prefix}{mod_name}.{pname}"
                param_to_module_idx[full_name] = idx

        return param_to_module_idx

    def split_from_param(self, param_name: str) -> Tuple[torch.Tensor, nn.Module]:
        param_map = self._map_params_to_layers()
        if param_name not in param_map:
            raise ValueError(f"Parameter '{param_name}' not found in model.")

        split_idx = param_map[param_name]
        modules = self.forward_modules
        layers1 = nn.Sequential(*[layer for _, layer in modules[:split_idx]])
        layers2 = nn.Sequential(*[layer for _, layer in modules[split_idx:]])

        with torch.no_grad():
            part1_out = layers1(self.sample_input)
            recombined_output = layers2(part1_out)

        if not torch.allclose(self.baseline_output, recombined_output, atol=1e-6):
            raise ValueError("Split model output does not match full model output.")

        return part1_out, layers2
