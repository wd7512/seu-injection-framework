# Issue #88: [SWARM REVIEW] Group F: Documentation Quality

## Findings (7 issues)

### F1 🟠 Research artifacts (ICLR reviews) shipped in examples/
**File:** `examples/flood_training_study/reviews/`
The flood training study contains ICLR peer review responses (`ICLR_REVIEW_RESPONSE.md`, `ICLR_REVIEW_V2_RESPONSE.md`) and rebuttal letters. These are research artifacts, not examples for users. Shipping peer review discussions in a package confusing.

**Fix:** Move to a `research/` or `papers/` top-level directory, or a separate repo branch.

### F2 🟠 No custom exception hierarchy
**File:** All `src/` modules
Every error uses generic `ValueError`, `TypeError`, or `RuntimeError`. Users cannot catch framework-specific errors without parsing message strings. A `SEUError` base class with subclasses like `SEUValidationError`, `SEUDeviceError`, `SEUComputationError` would dramatically improve debugging.

**Fix:** Define a small exception hierarchy in a `src/seu_injection/exceptions.py` module.

### F3 🟠 `accuracy.py` is 75% documentation boilerplate
**File:** `src/seu_injection/metrics/accuracy.py`
567 lines total, ~130 lines of logic, ~430 lines of docstrings. The phrase "device-aware", "memory-efficient", and "automatic" appears in every function docstring. One function has a 130-line docstring for an 18-line body.

**Fix:** Trim docstrings to ~60% of current length. Remove repetitive boilerplate. Move extended examples to the Sphinx docs or quickstart guide.

### F4 CHANGELOG.md dated "sometime"
**File:** `CHANGELOG.md:8`
```markdown
## [1.1.13] - sometime
```

**Fix:** Either add a real date or remove the placeholder entry.

### F5 No CITATION.cff for GitHub native citation
**File:** Root
GitHub shows a "Cite this repository" widget when `CITATION.cff` exists. Currently users must manually copy the BibTeX from raw README text.

**Fix:** Add a `CITATION.cff` file matching the BibTeX metadata already in README.md.

### F6 Two CONTRIBUTING.md files that drift
**Files:** `CONTRIBUTING.md` (324 lines) + `docs/source/contributing.md` (261 lines)
These overlap significantly but differ in content. CONTRIBUTING.md has more on commits/build, the docs version has more on TODOs/branching. They will inevitably diverge.

**Fix:** Keep one source of truth, have the other reference it with a symlink or redirect.

### F7 README quickstart import path inconsistent with __init__.py
**File:** `README.md:79`
```python
from seu_injection.core import ExhaustiveSEUInjector
```
But `__init__.py` re-exports this at the top level. Users get conflicting guidance on the canonical import path.

**Fix:** Use the simpler top-level import in the README example: `from seu_injection import ExhaustiveSEUInjector`


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:41Z |
| **Updated** | 2026-06-02T23:26:41Z |
| **Labels** | documentation |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/88 |
