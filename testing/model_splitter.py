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
        self.legal_splits = self._get_legal_split_params()

    def _trace_forward(self):
        traced = []
        hooks = []

        def hook_fn(module, input, output):
            traced.append((module, input, output))

        for module in self.model.modules():
            if len(list(module.children())) == 0:  # leaf only
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self.model(self.sample_input)

        for h in hooks:
            h.remove()

        return traced

    def _get_legal_split_params(self):
        legal = set()
        param_to_module = {}
        name_to_module = {name: m for name, m in self.model.named_modules()}

        for name, _ in self.model.named_parameters():
            mod_path = ".".join(name.split(".")[:-1])
            mod = name_to_module.get(mod_path, None)
            param_to_module[name] = mod

        used_modules = [mod for mod, _, _ in self.ordered_ops]
        used_ids = set(id(m) for m in used_modules)

        for name, mod in param_to_module.items():
            if mod is not None and id(mod) in used_ids:
                legal.add(name)

        return legal

    def split_from_param(self, param_name: str) -> Tuple[torch.Tensor, nn.Module]:
        if param_name not in self.legal_splits:
            raise ValueError(f"Parameter '{param_name}' is not a valid split point.")

        name_to_module = {name: m for name, m in self.model.named_modules()}
        mod_path = ".".join(param_name.split(".")[:-1])
        split_module = name_to_module[mod_path]

        for idx, (mod, _, _) in enumerate(self.ordered_ops):
            if mod is split_module:
                split_idx = idx
                break
        else:
            raise RuntimeError("Split point not found in trace.")

        ops1 = self.ordered_ops[:split_idx]
        ops2 = self.ordered_ops[split_idx:]

        class Part(nn.Module):
            def __init__(self, ops):
                super().__init__()
                self.ops = ops

            def forward(self, x):
                for mod, _, _ in self.ops:
                    x = mod(x)
                return x

        model1 = Part(ops1)
        model2 = Part(ops2)

        with torch.no_grad():
            part1_out = model1(self.sample_input)

            try:
                recombined = model2(part1_out)
            except RuntimeError as e:
                # Check if shape error occurred and model2 starts with Linear
                first_op = model2.ops[0][0] if model2.ops else None
                if isinstance(first_op, nn.Linear) and part1_out.dim() > 2:
                    try:
                        part1_out = part1_out.view(part1_out.size(0), -1)
                        recombined = model2(part1_out)
                    except Exception as e2:
                        raise RuntimeError(f"Automatic reshape failed: {e2}") from e
                else:
                    raise e

        if not torch.allclose(recombined, self.baseline_output, atol=1e-6):
            raise ValueError("Split model output does not match full output.")

        return part1_out, model2
    
    def available_split_points(self):
        return sorted(self.legal_splits)

    def get_safe_split_points(self):
        safe_points = []
        for param_name in self.legal_splits:
            try:
                self.split_from_param(param_name)
            except Exception:
                continue
            else:
                safe_points.append(param_name)
        return safe_points
    
    def get_structure_safe_split_points(self):
        safe_points = []
        for name in self.legal_splits:
            # Only split at top-level blocks or classification head
            if name.startswith("blocks.") and name.count('.') <= 2:
                safe_points.append(name)
            elif name.startswith("head.") or name.startswith("norm."):
                safe_points.append(name)
        return safe_points