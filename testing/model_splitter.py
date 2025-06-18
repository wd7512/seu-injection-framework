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

        self.ordered_ops = self._trace_forward()

    def _trace_forward(self):
        traced = []
        hooks = []

        def hook_fn(module, input, output):
            traced.append((module, input, output))

        for module in self.model.modules():
            if len(list(module.children())) == 0:  # leaf modules only
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model(self.sample_input)

        for h in hooks:
            h.remove()

        return traced

    def split_from_param(self, param_name: str) -> Tuple[torch.Tensor, nn.Module]:
        name_to_module = {name: mod for name, mod in self.model.named_modules()}
        param_to_module = {}

        for name, param in self.model.named_parameters():
            mod_name = ".".join(name.split(".")[:-1])
            param_to_module[name] = name_to_module.get(mod_name)

        if param_name not in param_to_module:
            raise ValueError(f"Parameter '{param_name}' not found.")

        target_module = param_to_module[param_name]

        # Find split index in ordered_ops
        for idx, (mod, _, _) in enumerate(self.ordered_ops):
            if mod is target_module:
                split_idx = idx
                break
        else:
            raise RuntimeError("Split point not found in forward trace.")

        # Compose partial models
        class Part1(nn.Module):
            def __init__(self, ops):
                super().__init__()
                self.ops = ops

            def forward(self, x):
                for mod, _, _ in self.ops:
                    x = mod(x)
                return x

        class Part2(nn.Module):
            def __init__(self, ops):
                super().__init__()
                self.ops = ops

            def forward(self, x):
                for mod, _, _ in self.ops:
                    x = mod(x)
                return x

        ops1 = self.ordered_ops[:split_idx]
        ops2 = self.ordered_ops[split_idx:]

        model1 = Part1(ops1)
        model2 = Part2(ops2)

        with torch.no_grad():
            part1_out = model1(self.sample_input)
            recombined_out = model2(part1_out)

        if not torch.allclose(recombined_out, self.baseline_output, atol=1e-6):
            raise ValueError("Split model output does not match baseline.")

        return part1_out, model2
