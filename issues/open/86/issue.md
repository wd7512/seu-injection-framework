# Issue #86: [SWARM REVIEW] Group D: Correctness & Edge Cases

## Findings (4 issues)

### D1 🟠 `bit_i` range inconsistency: injector accepts 32, bitops reject it
**File:** `src/seu_injection/core/base_injector.py:156`
```python
if bit_i not in range(33):  # Accepts 0..32 (33 values!)
    raise ValueError(f"bit_i must be in [0, 32], got {bit_i}")
```
But the bitops modules use:
```python
# float32.py:26
if not (0 <= bit_position <= 31):  # Accepts 0..31
# float32_legacy.py:31  
if not (0 <= bit_i <= 31):  # Accepts 0..31
```
A user passing `bit_i=32` gets past the injector check (which says it is valid) but then hits a `ValueError` from the bitops with a confusing message.

**Fix:** Change `range(33)` to `range(32)` in `base_injector.py:156`.

### D2 `struct.error` not caught in fallback path
**File:** `src/seu_injection/bitops/float32.py:86`
```python
except (ValueError, TypeError):
    # fall back to legacy
```
But `struct.pack("f", ...)` raises `struct.error` (inherits from `Exception`, not `ValueError`/`TypeError`). Non-float scalars that pass the `np.isscalar()` check produce an unhandled exception instead of falling back.

**Fix:** Also catch `struct.error`, or restructure to avoid needing a fallback path at all.

### D3 Binary accuracy midpoint threshold fragile
**File:** `src/seu_injection/metrics/accuracy.py:558-559`
```python
y_low = np.min(y_true)
y_high = np.max(y_true)
midpoint = (y_high + y_low) / 2
```
If all labels in the batch are identical (e.g., all 0), then `midpoint = 0` and every prediction (including 0.0) maps to the "high" label, giving 100% accuracy when the model is wrong.

**Fix:** Add a guard for the degenerate case where `y_low == y_high`.

### D4 `run_at_least_one_injection=True` by default silently overrides p=0.0
**File:** `src/seu_injection/core/stochastic_seu_injector.py:58`
The default forces at least one injection even when `p=0.0`. This was deliberately introduced (issue #48) to fix failing smoke tests, but it violates the principle of least surprise — `p=0.0` should mean zero probability.

**Fix:** Keep the parameter for smoke tests but change the default to `False`. Update smoke tests to explicitly pass `run_at_least_one_injection=True`.


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:24Z |
| **Updated** | 2026-06-02T23:28:14Z |
| **Labels** | bug |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/86 |
