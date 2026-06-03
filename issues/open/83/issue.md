# Issue #83: [SWARM REVIEW] Group A: Import-Time Side Effects & Phantom Exports

## Findings (5 issues)

### A1 🔴 print() at import time
**File:** `src/seu_injection/__init__.py:65`
`print("Initialised version:", __version__)` fires on every `import seu_injection`. Pollutes stdout, breaks CLI parsing, cannot be suppressed by users. A library should never print at import time.

**Fix:** Remove the print statement or route through `logging` at DEBUG level.

### A2 🔴 `SEUInjector` phantom export — breaks users
**File:** `src/seu_injection/__init__.py:57-62`
`__all__` is extended with `"SEUInjector"` but this name is never imported or defined anywhere in the package. `from seu_injection import SEUInjector` raises `ImportError`. The TODO says "deprecation" but there is nothing to deprecate.

**Fix:** Either define `SEUInjector = ExhaustiveSEUInjector` as a backward-compat alias, or remove it from `__all__` entirely.

### A3 `__all__` modified dynamically after list literal
**File:** `src/seu_injection/__init__.py:42-62`
`__all__` is declared as a list literal (lines 42-55), then `__all__.extend(...)` adds `SEUInjector` separately (lines 58-62). This is fragile and non-idiomatic.

**Fix:** Define `__all__` as a single complete list. Remove the `.extend()` call.

### A4 `core/__init__.py` stale docstring
**File:** `src/seu_injection/core/__init__.py:3`
Docstring says *"This module provides the main SEUInjector class"* — but `SEUInjector` does not exist in the current codebase. `ExhaustiveSEUInjector` and `StochasticSEUInjector` are the current classes.

**Fix:** Update docstring to reference the current class names.

### A5 `type: ignore[assignment]` on self.criterion
**File:** `src/seu_injection/core/base_injector.py:95`
```python
self.criterion = criterion  # type: ignore[assignment]
```
This suppresses a type error instead of fixing the root cause — likely because `Callable[..., float]` does not match the actual criterion callable signature.

**Fix:** Either use a `Protocol` to define the criterion type properly, or widen the type annotation to match the runtime usage.


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:01Z |
| **Updated** | 2026-06-02T23:26:01Z |
| **Labels** | enhancement |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/83 |
