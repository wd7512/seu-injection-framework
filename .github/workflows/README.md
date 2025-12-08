# GitHub Actions Workflows

This directory contains the CI/CD workflows for the SEU Injection Framework.

## Workflow Files

### Core Workflows

- **`ci.yml`** - Continuous Integration workflow that runs on all branches and PRs
- **`dependencies.yml`** - Automated weekly dependency updates
- **`release.yml`** - Release automation workflow
- **`block-dev-docs-changes.yml`** - Prevents direct documentation changes to dev branch

### Reusable Workflows

- **`test-matrix.yml`** - Reusable workflow defining the test matrix (single source of truth)

## Test Matrix Configuration

The test matrix is centrally defined in `test-matrix.yml` to ensure consistency across all workflows. This reusable workflow is used by:

- **`ci.yml`** (full_tests job) - Tests on main branch with frozen dependencies
- **`dependencies.yml`** (update-and-test job) - Tests with updated dependencies

### Test Matrix Specification

```yaml
os: [ubuntu-latest, windows-latest, macos-latest]
python-version: ['3.10', '3.11', '3.12']
```

This results in 9 test configurations (3 OS × 3 Python versions).

### Usage

#### For CI with frozen dependencies:
```yaml
jobs:
  test:
    uses: ./.github/workflows/test-matrix.yml
    with:
      frozen-deps: true
      update-deps: false
```

#### For dependency updates:
```yaml
jobs:
  test:
    uses: ./.github/workflows/test-matrix.yml
    with:
      frozen-deps: false
      update-deps: true
```

## Benefits of Reusable Workflows

1. **Single Source of Truth**: Test matrix defined in one place
2. **Consistency**: All workflows use identical test configurations
3. **Maintainability**: Update test matrix once, applies everywhere
4. **DRY Principle**: Eliminates code duplication across workflows

## Modifying the Test Matrix

To add/remove OS or Python versions, edit only `test-matrix.yml`. Changes automatically apply to all workflows that use it.

Example - Adding Python 3.13:
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  python-version: ['3.10', '3.11', '3.12', '3.13']  # Added 3.13
```

This change will automatically be reflected in both CI and dependency update workflows.
