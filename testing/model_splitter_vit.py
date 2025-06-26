import torch
import torch.nn as nn
from typing import Tuple

class ViTBlockSplitter:
    def __init__(self, model: nn.Module, sample_input: torch.Tensor):
        self.model = model.eval()
        self.sample_input = sample_input

        with torch.no_grad():
            self.baseline_output = self.model(self.sample_input)

        self.num_blocks = len(self.model.blocks)
        self.ordered_parts = self._define_parts()

    def _define_parts(self):
        # Define parts as submodules representing slices of the ViT pipeline
        parts = []

        class Part(nn.Module):
            def __init__(self, forward_fn):
                super().__init__()
                self.forward_fn = forward_fn
            def forward(self, x):
                return self.forward_fn(x)

        # Part 0: embedding + class token prep + first N blocks (N varies with split)
        # Part 1: remaining blocks + norm + head

        def part0_fn_factory(num_blocks_to_include):
            def forward(x):
                B = x.size(0)
                # patch embedding + flatten + class token
                x = self.model.patch_embed(x)  # (B, num_patches, embed_dim)
                cls_token = self.model.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                x = x + self.model.pos_embed
                x = self.model.pos_drop(x)
                for i in range(num_blocks_to_include):
                    x = self.model.blocks[i](x)
                return x
            return forward

        def part1_fn_factory(start_block_idx):
            def forward(x):
                for i in range(start_block_idx, self.num_blocks):
                    x = self.model.blocks[i](x)
                x = self.model.norm(x)
                # classification head uses only the cls token
                cls_token_final = x[:, 0]
                x = self.model.head(cls_token_final)
                return x
            return forward

        # Create all possible splits between blocks (including before block 0 and after last block)
        for split_idx in range(self.num_blocks + 1):
            parts.append((
                Part(part0_fn_factory(split_idx)),
                Part(part1_fn_factory(split_idx))
            ))
        return parts

    def available_split_points(self):
        # Return string keys to indicate split location, e.g. 'after_block_0'
        return [f"after_block_{i}" for i in range(self.num_blocks + 1)]

    def split_from_point(self, split_point: str) -> Tuple[torch.Tensor, nn.Module]:
        # Validate split point
        prefix = "after_block_"
        if not split_point.startswith(prefix):
            raise ValueError(f"Invalid split point name: {split_point}")
        idx = int(split_point[len(prefix):])
        if not (0 <= idx <= self.num_blocks):
            raise ValueError(f"Split index out of range: {idx}")

        part0, part1 = self.ordered_parts[idx]

        with torch.no_grad():
            out0 = part0(self.sample_input)
            out_recombined = part1(out0)

        if not torch.allclose(out_recombined, self.baseline_output, atol=1e-6):
            raise RuntimeError("Split recombination output mismatch")

        return out0, part1

    def get_safe_split_points(self):
        safe = []
        for sp in self.available_split_points():
            try:
                self.split_from_point(sp)
            except Exception:
                continue
            else:
                safe.append(sp)
        return safe
