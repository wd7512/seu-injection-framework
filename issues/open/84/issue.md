# Issue #84: [SWARM REVIEW] Group B: Injection Loop Overhead

## Findings (3 issues)

**Important context:** This group is about *overhead between* forward passes, NOT about eliminating forward passes. The exhaustive injector correctly runs one forward pass per bitflip — that is the experiment. But the per-element bookkeeping has significant waste.

### B1 🟠 Vectorized bitflip path is 1,176× faster but never used by injectors
**Files:** `src/seu_injection/bitops/float32.py:43-60`, both injectors

The optimized `_bitflip_array_optimized()` does `view(np.uint32) ^= mask` — a single vectorized XOR across the entire array. However, both injectors call the scalar `bitflip_float32_optimized()` per-element instead, losing the vectorization benefit.

The bitflip itself is ~3μs per element (negligible vs a forward pass), but the Python loop overhead over 11M params adds up. The fix is to flip ALL selected bits in one operation before the forward pass, then restore after.

**Fix:** Restructure the injection loop to:
1. Clone the entire layer tensor once
2. Vectorize all bitflips using the array path
3. Do a single CPU→GPU transfer of the modified tensor
4. Run one forward pass
5. Restore the original tensor in one operation

### B2 🟠 Per-element CPU→GPU tensor transfers
**Files:** Both injectors

Each injection does:
```python
tensor.data[idx] = torch.tensor(seu_val, device=self.device, dtype=tensor.dtype)
```
This creates a new CUDA tensor per element, incurring kernel launch overhead (~5-10μs). For 11M params, that is 55-110s of pure launch latency.

**Fix:** Same as B1 — flip all values on the numpy side, then transfer once.

### B3 `ensure_tensor()` always clones unnecessarily
**File:** `src/seu_injection/utils/device.py:61`

```python
if isinstance(data, torch.Tensor):
    result = data.clone().detach()
```
Clones even when dtype and device already match the target. Creates 2× unnecessary memory per conversion.

**Fix:** Only clone when dtype or device actually needs changing.


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:08Z |
| **Updated** | 2026-06-02T23:26:08Z |
| **Labels** | enhancement |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/84 |
