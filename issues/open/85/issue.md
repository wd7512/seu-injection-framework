# Issue #85: [SWARM REVIEW] Group C: API Design & Constructor Issues

## Findings (6 issues)

### C1 🟠 `p` passed via `**kwargs` — invisible, undocumentable
**File:** `src/seu_injection/core/stochastic_seu_injector.py:57`
```python
p = kwargs.get("p", 0.0)
```
The core parameter `p` (injection probability) is passed via undocumented `**kwargs`. A user looking at `run_injector()` signature has no idea `p` exists. No IDE autocomplete, no docstring visibility.

**Fix:** Add `p: float = 0.0` as an explicit parameter to both `run_injector()` and `_run_injector_impl()`.

### C2 🟠 Constructor runs model evaluation as side-effect
**File:** `src/seu_injection/core/base_injector.py:136-138`
```python
self.baseline_score = criterion(self.model, self.X, self.y, self.device)
```
`__init__` calls the criterion function as a side-effect. A constructor should assemble an object, not perform expensive computation. This makes instantiation slow (~seconds) and non-obvious.

**Fix:** Either lazy-compute `baseline_score` on first access, or separate construction from computation with a class method:
```python
injector = ExhaustiveSEUInjector(model, x=x, y=y)
injector.compute_baseline()
```

### C3 🟠 85% code duplication between injector subclasses
**Files:** `exhaustive_seu_injector.py` + `stochastic_seu_injector.py`
Both injectors share identical: layer iteration, tensor cloning, CPU transfer, result recording, and restoration logic. Only the element-selection strategy differs (all params vs sampled params). This is a Strategy Pattern violation.

**Fix:** Extract the iteration strategy into a pluggable component. The base class should handle the loop; subclasses only provide the selection mask.

### C4 `bitflip_float32` exports the legacy (string-based) version
**File:** `src/seu_injection/bitops/__init__.py`
The public API exports `bitflip_float32` which is the slow string-based legacy implementation. Users who `from seu_injection import bitflip_float32` get the 30× slower version.

**Fix:** Make `bitflip_float32` point to `bitflip_float32_optimized`, move the legacy version to a documented compat module.

### C5 Return type should be TypedDict or dataclass
**File:** Both injectors
```python
def _run_injector_impl(...) -> dict[str, list[Any]]:
```
The results dict has specific well-defined keys (`tensor_location`, `criterion_score`, `layer_name`, `value_before`, `value_after`) but is typed as `dict[str, list[Any]]` — zero IDE support.

**Fix:** Define a `TypedDict` or `@dataclass` for injection results.

### C6 `get_model_info()` is dead code
**File:** `src/seu_injection/utils/device.py:73-111`
Function implemented with 40 lines of code but has zero callers in `src/` or `tests/`. Confirmed via grep across the entire codebase.

**Fix:** Either remove it, or add it to the public API with proper tests if it has value.


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:16Z |
| **Updated** | 2026-06-02T23:28:19Z |
| **Labels** | enhancement |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/85 |
