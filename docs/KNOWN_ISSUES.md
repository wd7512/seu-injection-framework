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

## Code Quality Improvements (Tracked via TODOs)

**Issue**: Various code quality improvements identified during comprehensive review

**Status**: Tracked via embedded TODO system throughout codebase

**Details**: 
- Import optimization opportunities (function-specific imports)
- Error handling standardization (consistent exception types)
- Test quality enhancements (stronger assertions)
- Dead code cleanup (unused utility functions)
- Performance optimization opportunities (documented in critical paths)

**Workaround**: These are tracked as TODOs in relevant source files for contextual awareness

**Priority**: Variable by category - see individual TODO priorities in source code

---

**Note**: The framework uses an embedded TODO system to track improvements. This is normal development practice and indicates active maintenance, not technical debt.

This file tracks non-critical issues that don't block the v1.1.1 PyPI release.