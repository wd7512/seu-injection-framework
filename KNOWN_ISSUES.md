# Known Issues

## Examples Directory Issue

**Issue**: Examples hang during pandas import (dependency resolution issue)

**Status**: Non-critical for PyPI release

**Details**: 
- Running `python basic_cnn_robustness.py` hangs during pandas import
- Appears to be dependency/environment related
- Examples code is correct but dependency chain is slow/problematic
- Not blocking PyPI release as core framework works correctly

**Workaround**: Use `uv run python` instead of direct python execution

**Priority**: Low - address in future maintenance release

---

This file tracks non-critical issues that don't block the v1.0.0 PyPI release.