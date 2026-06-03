# Issue #87: [SWARM REVIEW] Group E: CI/CD & Tooling Conflicts

## Findings (5 issues)

### E1 🟠 Conflicting ruff configurations silently shadow each other
**File:** Root (both `.ruff.toml` and `pyproject.toml`)

There are TWO ruff configs with incompatible settings:

| Setting | `.ruff.toml` | `pyproject.toml` |
|---|---|---|
| `line-length` | 120 | 88 |
| `target-version` | py311 | py39 |
| Lint rules | E, W, I, UP, D1 | E, W, F, I, N, UP, B, A, C4, T20 |
| `fix` | `true` (auto-fix!) | not set |

**Critical:** When `.ruff.toml` exists at the project root, it SHADOWS `[tool.ruff]` in `pyproject.toml` entirely. Ruff does NOT merge them. So `pyproject.toml`'s ruff config is silently ignored. CI runs with the relaxed rules, not the intended strict config. Also, `fix=true` means `ruff check` can auto-rewrite files in CI.

**Fix:** Delete one of the two configs. Consolidate into `pyproject.toml` (the single-source-of-truth for build tooling).

### E2 🟠 `mdformat .` without `--check` modifies files in CI
**File:** `.github/workflows/ci.yml:42`
```yaml
- name: Markdown formatting
  run: uv run mdformat .
```
This formats ALL markdown files in-place during CI instead of checking them. CI should detect bad formatting and fail, not silently rewrite files.

**Fix:** Change to `uv run mdformat . --check`.

### E3 Tautological test assertions
**File:** `tests/unit_tests/test_injector.py` (stochastic probability test)
```python
assert low_count >= 0  # always true
assert high_count >= 0  # always true
```
These assertions are tautologically true (non-negative integers). The meaningful assertion — that `p=0.9` produces more results than `p=0.1` — is commented out.

**Fix:** Either add the meaningful comparison or restructure the test to verify injection counts scale with p.

### E4 `tests/overhead/` is orphaned
**File:** `tests/overhead/`
The overhead measurement module (147 lines) is not connected to any pytest run or CI job. Contains stale result JSON files from November 2025.

**Fix:** Either integrate it into the test suite or remove it.

### E5 `actions/checkout@v3` in one workflow vs v4 everywhere else
**File:** `.github/workflows/block-dev-docs-changes.yml`
Uses @v3 while all other workflows use @v4. Not urgent but introduces a minor consistency issue and will eventually get deprecated.

**Fix:** Update to `actions/checkout@v4`.


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:32Z |
| **Updated** | 2026-06-02T23:26:32Z |
| **Labels** | bug |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/87 |
