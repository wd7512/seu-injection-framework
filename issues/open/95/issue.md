# Issue #95: [RFC] MPS (Apple Silicon GPU) Support — Architecture Proposal

## Summary

The framework currently ignores MPS (Apple Silicon GPU) — it only detects CUDA or CPU. This means every Mac user (including the author, who runs Apple Silicon) gets CPU fallback even though a GPU is available. One example script (`examples/fault_injection_training/fault_injection_training_study.py`) already has a hand-rolled MPS detection function, proving demand exists — but it's duplicated, lives in the wrong place, and has inverted priority (checks MPS before CUDA).

This issue proposes the architecture for full MPS support and is open for discussion before implementation.

---

## Current State

### Core library (2 files)
| File | Line | Pattern |
|------|------|---------|
| `src/seu_injection/utils/device.py` | 36 | `if CUDA → cuda else cpu` — no MPS |
| `src/seu_injection/core/base_injector.py` | 94 | `"cuda" if torch.cuda.is_available() else "cpu"` — no MPS |

### Tests (6 files — all repeat the same inline pattern)
`tests/unit_tests/conftest.py:30`, `tests/smoke/test_basic_functionality.py:87`, `tests/benchmarks/test_performance.py:47`, `tests/benchmarks/standalone_benchmark.py:9`, `tests/unit_tests/test_injector.py:60` (assertion excludes MPS), `tests/unit_tests/test_injector.py:63-68` (CUDA-only branching)

### Benchmarks — will crash on MPS (3 files)
- `standalone_benchmark.py`: 4× `torch.cuda.synchronize()`, 1× `torch.cuda.get_device_name()` — all unguarded
- `test_performance.py`: 3× `torch.cuda.synchronize()`, 1× `torch.cuda.get_device_name()`
- `test_injector.py:439`: guarded correctly (`if device.type == "cuda"`) but doesn't sync MPS

### Examples (4 files)
- `architecture_comparison.py:691`: CUDA/CPU only
- `fault_injection_training_study.py:50`: **Has correct MPS detection** (both `is_available()` and `is_built()`) — but uses wrong priority (MPS before CUDA), and duplicates framework logic that should live in `device.py`
- 2 notebooks: mention CUDA/CPU only

---

## Proposed Architecture

### 1. Unified `detect_device()` — Priority-based auto-detection

Replace the current CUDA-or-CPU-only logic with a priority-based dispatcher:

```
Priority: CUDA > MPS > CPU
```

```python
def detect_device(preferred_device=None, priority=("cuda", "mps", "cpu")):
    if preferred_device is None:
        for device_type in priority:
            if device_type == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if device_type == "mps" and (
                torch.backends.mps.is_available() and torch.backends.mps.is_built()
            ):
                return torch.device("mps")
            if device_type == "cpu":
                return torch.device("cpu")
    # Explicit device: validate availability, warn+fallback if unavailable
    ...
```

**Key decisions:**
- **MPS check uses both `is_available()` and `is_built()`** — follows PyTorch's documented recommended pattern
- **Explicit overrides with graceful fallback** — `detect_device("cuda")` on a Mac warns and falls through priority instead of crashing later
- **`priority` parameter as escape hatch** — consumers that want MPS-first (e.g., MPS-only Macs) can pass `priority=("mps", "cpu")`

### 2. Unified `synchronize()` dispatch

Replace all 8 `torch.cuda.synchronize()` calls with a backend-agnostic utility:

```python
def synchronize(device=None):
    if device.type == "cuda":
        torch.cuda.synchronize(device.index)
    elif device.type == "mps":
        torch.mps.synchronize()  # single-device, no index param
    # CPU is a no-op
```

### 3. Test infrastructure changes

- Fix assertion in `test_injector.py:60`: accept `{"cuda", "cpu", "mps"}`
- Add `requires_cuda`, `requires_mps`, `requires_gpu` pytest markers
- MPS-aware device fixtures in conftest.py
- Benchmark `get_device_name()` helper: CUDA → real name, MPS → `"Apple Silicon GPU"`, CPU → thread count

### 4. MPS-specific quirks to document

| Quirk | Implication |
|-------|-------------|
| No `float64` on MPS (silent CPU fallback) | Add `ensure_device_compatible_dtype()` |
| Some operations non-deterministic (`scatter_add_`, `histc`) | Warn users; advise CUDA/CPU for publication results |
| No `torch.mps.empty_cache()` | Document as no-op |
| `torch.manual_seed()` covers MPS (no separate call needed) | `set_seed()` utility handles all backends |

---

## Migration Plan

| Phase | Scope | Risk |
|-------|-------|------|
| **1. Foundation** | Rewrite `device.py` + unit tests | None (new code, existing API preserved) |
| **2. Framework** | `base_injector.py` + test fixtures call `detect_device()` | Low (replace identical logic) |
| **3. Periphery** | Benchmarks, examples, docstrings | Low (mechanical replacement) |
| **4. Polish** | CI for MPS runners, edge cases | Medium (CI config changes) |

---

## Discussion Points

1. **Priority order**: CUDA > MPS > CPU — correct for performance? Or should we prefer MPS over CUDA when both are available (rare: eGPU Mac)?
2. **Auto-detect vs opt-in**: Should MPS be on by default or behind a config flag for safety?
3. **CI**: Worth adding a `macos-14` runner for MPS tests? (Free tier minutes on GitHub Actions)
4. **Float64 handling**: Should `ensure_device_compatible_dtype()` be called automatically or left to the caller?
5. **Reproducibility note**: MPS non-determinism is hardware-level — should the framework actively warn or just document?

---

## References

- Existing MPS precedent: `examples/fault_injection_training/fault_injection_training_study.py:50-54`
- PyTorch MPS docs: https://pytorch.org/docs/stable/notes/mps.html
- File-level scan results available on request


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-03T18:32:28Z |
| **Updated** | 2026-06-03T18:32:28Z |
| **Labels** | enhancement, question |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/95 |
