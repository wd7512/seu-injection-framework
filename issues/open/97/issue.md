# Issue #97: v1.3.0 fixes

## Summary

Consolidated fixes from the v1.3.0 release PR (#96) swarm + Gemini reviews. These findings came from a 5-agent parallel review of PR #96 and Gemini Code Assist's automated review.

---

## 🔴 CRITICAL (must fix)

### C-1. Training study: README vs CSV data contradiction

The README (lines 135-150) claims +13% to +93% improvement from flood training, but the committed CSV (`robustness_results.csv`) shows **−30% to −155%** — the fault-aware model performs *worse* on every measurable bit position. One of these datasets is from a different experimental run. Determining which is correct and aligning the documentation is essential before making scientific claims.

**File:** `examples/fault_injection_training/README.md` vs `examples/fault_injection_training/robustness_results.csv`

### C-2. Training study: Notebook uses confounded model construction

`notebook.ipynb` creates baseline and fault-aware models via **separate constructor calls** (cells 6-7), giving them different random weight initializations. This means any robustness difference could be due to different starting weights, not the flood training intervention. The study script (`fault_injection_training_study.py`) correctly uses `copy.deepcopy()` for a paired design — the notebook should match.

**File:** `examples/fault_injection_training/notebook.ipynb`

### C-3. Training study: Notebook declares unsupported conclusions

The notebook labels all hypotheses "CONFIRMED" with checkmarks:
- "H1 CONFIRMED: Fault-aware training significantly improves robustness" — contradicted by CSV data showing degradation on most bits
- "H2 CONFIRMED: Weight importance distributed more evenly" — zero weight-distribution analysis performed
- "H3 CONFIRMED: Improvements generalize across bit positions" — CSV shows degradation on bits 0, 1, 8
- "Up to 70%+ improvement" / "3-10x resilience" — not supported by CSV output data

**File:** `examples/fault_injection_training/notebook.ipynb`

---

## 🟧 HIGH

### H-1. `_run_injector_impl` removed from abstract contract

Previously `@abstractmethod`, now a concrete template method. Any custom subclass that overrides `_run_injector_impl` will silently have its implementation ignored. The new extension point is `_get_injection_indices()` but this contract change is undocumented in the CHANGELOG.

**File:** `src/seu_injection/core/base_injector.py`

### H-2. CHANGELOG references non-existent `_get_layers()` method

The CHANGELOG line 27 lists `_get_layers()` as an extracted helper — it doesn't exist anywhere in the codebase. The actual helpers are: `_inject_and_evaluate`, `_iterate_layers`, `_prepare_tensor_for_injection`, `_record_injection_result`, `_initialize_results`.

**File:** `CHANGELOG.md`

### H-3. MPS missing from core device detection

`base_injector.py:94` uses `"cuda" if torch.cuda.is_available() else "cpu"` — ignores MPS entirely. On Apple Silicon, the core library falls back to CPU while the example study script correctly detects MPS. Issue #95 tracks the full MPS architecture, but the code inconsistency should be acknowledged.

**File:** `src/seu_injection/core/base_injector.py` (also `tests/unit_tests/conftest.py`, `tests/benchmarks/`)

### H-4. Seed reproducibility test doesn't verify reproducibility

The integration test creates two seeded injectors but only checks they produce the **same count** of injections, not the **same set** of injections. It should compare `results["tensor_location"]` or full result dicts for identity.

**File:** `tests/integration/test_workflows.py`

### H-5. Notebook has zero limitations or caveats

The README has 8 well-written limitations (single seed, synthetic data, small model, etc.). The notebook has none — it declares all hypotheses "CONFIRMED" and jumps to "Key Takeaways" and "Recommendations" as if definitive. This imbalance misrepresents the study's strength to users who interact primarily with the notebook.

**File:** `examples/fault_injection_training/notebook.ipynb`

### H-6. Gemini: Remove unused `original_tensor` clone

`_prepare_tensor_for_injection()` returns both `(numpy_array, original_tensor)` but `_inject_and_evaluate` only uses the numpy array for injection — the `original_tensor` is unused since restoration now uses the CPU scalar `original_val`. The clone and the return should be removed.

**File:** `src/seu_injection/core/base_injector.py`

---

## 🟡 MEDIUM

| # | Finding | File | Line |
|---|---------|------|------|
| M-1 | Invalid `layer_name` only `print()`s warning instead of raising `ValueError` — docstring promises ValueError | `base_injector.py` | 166 |
| M-2 | Missing CUDA/MPS synchronize between injection write and evaluation forward pass — rare race condition on async GPU streams | `base_injector.py` | 289 |
| M-3 | Training study file I/O (savefig, to_csv) has no error handling — 2+ minute experiment lost on write failure | `training_study.py` | 672 |
| M-4 | Tautological assertions: `assert len(...) >= 0` always passes; `test_stochastic_seu_probability_effects` claims to test probability effect but doesn't assert it | `test_injector.py` | 387 |
| M-5 | Multiple tests use `np.random.seed(42)` / `torch.manual_seed(42)` with zero effect on injector's per-instance `default_rng()` | `test_injector.py` | multiple |
| M-6 | 14 E402 lint errors in notebook (imports after `warnings.filterwarnings`) | `notebook.ipynb` | cell 5 |
| M-7 | Generated artifacts (PNG, CSV) committed — CHANGELOG claims "properly gitignored" but they're tracked in git | `examples/fault_injection_training/*.png` | — |
| M-8 | Division-by-near-zero inflates "74.8% improvement" and "92.8% improvement" for sub-0.5% baseline drops | `README.md` | 149 |
| M-9 | `bit_i` validation range change (`range(33)` to `range(32)`) undocumented in CHANGELOG | `CHANGELOG.md` | — |
| M-10 | `ExhaustiveSEUInjector._get_injection_indices` kwargs `warnings.warn` path has zero test coverage | `exhaustive_seu_injector.py` | 53 |
| M-11 | Gemini: RNG should be initialised persistently in `BaseInjector.__init__` rather than recreated per call | `stochastic_seu_injector.py` | — |
| M-12 | Gemini: Undefined variable reference in training study documentation | `fault_injection_training_study.py` | — |

---

## 🟢 LOW

- `np.isscalar` deprecation warning (NumPy 2.x compatibility)
- Docstring inaccuracies: several methods claim to raise exceptions they don't raise
- Zero-dimension (scalar) parameters silently skipped (extreme edge case)
- Cache disabled when `test-matrix.yml` called with `checkout-ref` (minor perf concern)
- Format drift in `fault_injection_training_study.py` (15 formatting diffs vs ruff)
- CHANGELOG inaccurately claims `versions_and_plan.md` guard was added in this PR (pre-existing)
- `# type: ignore` removals in `accuracy.py` — cosmetic only
- `idx` parameter typed as bare `tuple` instead of `tuple[int, ...]` in helpers

---

## Source reviews

- Swarm review: PR #96 comment (5 agents: Security, Logic and Edge Cases, API Design, Test Coverage, Style and Reproducibility, Scientific Methodology)
- Gemini Code Assist: 18 inline comments on PR #96


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-03T19:14:39Z |
| **Updated** | 2026-06-03T19:17:40Z |
| **Labels** | enhancement |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/97 |
