# Known Issues

*Last Updated: November 13, 2025*

This document tracks known limitations, open issues, and planned improvements for the SEU Injection Framework.

______________________________________________________________________

## 🐛 Current Known Issues

### 1. Examples Directory - Pandas Import Performance

**Issue**: Example scripts experience slow startup times during pandas import

**Details**:

- Running `python basic_cnn_robustness.py` or `architecture_comparison.py` can hang temporarily during pandas import
- Related to dependency resolution chain in certain Python environments
- Does not affect framework functionality when imported as a library

**Workaround**: Use `uv run python examples/basic_cnn_robustness.py` for fast execution

**Priority**: Low

______________________________________________________________________

## 📋 Code Quality Improvements (Tracked via TODOs)

### 2. Performance Optimization Opportunities

**Locations**: `src/seu_injection/bitops/float32.py`, `src/seu_injection/core/injector.py`

**Details**:

- `bitflip_float32()` uses O(32) string manipulation (optimized version `bitflip_float32_optimized()` provides 10-100x speedup)
- Some code paths could benefit from additional vectorization
- GPU operations have room for optimization in large model scenarios

**Priority**: MEDIUM

### 3. Error Handling Standardization

**Locations**: `src/seu_injection/metrics/accuracy.py`, `src/seu_injection/utils/device.py`

**Details**:

- Mixed use of `ValueError`, `RuntimeError`, and `TypeError` without clear patterns
- Custom exception types would improve error clarity and debugging

**Priority**: MEDIUM

### 4. Test Quality Enhancements

**Locations**: `tests/unit_tests/test_injector.py`, `tests/unit_tests/test_utils.py`

**Details**: Edge cases not yet tested (current coverage: 94%):

- NaN/Inf handling in bitflip operations
- Multi-GPU device handling
- Custom criterion function validation edge cases
- Layer-specific injection with complex model hierarchies

**Priority**: LOW

### 5. Import Optimization

**Locations**: `src/seu_injection/bitops/float32.py`

**Details**: `struct` module imported globally but only used in 2 specific functions

**Priority**: LOW

### 6. Unused Utility Functions

**Locations**: `src/seu_injection/utils/device.py` - `get_model_info()` function

**Details**: Function appears unused throughout codebase, candidate for removal

**Priority**: LOW

### 7. API Complexity

**Locations**: `src/seu_injection/__init__.py`, `src/seu_injection/core/injector.py`

**Details**:

- Current API requires understanding of IEEE 754 bit positions, layer names, device management
- Simplified convenience functions planned for common use cases

**Priority**: MEDIUM

______________________________________________________________________

## 🔧 Planned Improvements

### Future Version Roadmap

**v1.2.0 (Planned)** - Usability Enhancements

- Simplified convenience functions (`quick_robustness_check()`, etc.)
- Enhanced error messages with helpful suggestions
- Comprehensive ReadTheDocs documentation site

**v1.3.0 (Planned)** - Analysis Tools

- Built-in visualization utilities
- Statistical analysis functions
- Automated report generation

**v2.0.0 (Planned)** - Extended Features

- Additional fault models (multi-bit upsets, stuck-at faults)
- Performance optimizations for very large models
- Support for additional ML frameworks

See `dev_docs/NEXT_STEPS.md` for detailed roadmap.

______________________________________________________________________

## ✅ Current Status

- **Test Coverage**: 94% (116 tests passing)
- **Code Quality**: All ruff checks passing, zero violations
- **Type Safety**: Comprehensive type hints in progress
- **Performance**: Optimized functions provide 10-100x speedup over naive implementations

______________________________________________________________________

## 📊 TODO System Reference

The framework uses standardized TODO comments throughout the codebase:

### TODO Format

```python
# TODO CATEGORY: Brief description
# ISSUE: Detailed problem explanation
# SOLUTION: Specific implementation approach
# IMPACT: User/performance implications
# PRIORITY: HIGH/MEDIUM/LOW
```

### TODO Categories

- **PERFORMANCE CRITICAL**: Issues affecting framework speed
- **PRODUCTION READINESS**: API and usability improvements
- **ERROR HANDLING**: Exception handling and validation
- **TESTING IMPROVEMENTS**: Coverage and test quality
- **CODE QUALITY**: Import optimization, cleanup
- **MAINTAINABILITY**: Long-term code maintenance
- **VECTORIZATION SUCCESS**: Examples of optimal implementations

### Finding TODOs

```bash
# Search all TODO items
grep -r "TODO" src/seu_injection/

# Search by category
grep -r "TODO PERFORMANCE" src/

# View all priorities
grep -r "PRIORITY:" src/
```

______________________________________________________________________

## 🤝 Contributing to Issue Resolution

Want to help resolve any of these issues?

1. **Review the specific TODO** in the source file for detailed context
1. **Check existing PRs/issues** on GitHub to avoid duplication
1. **Open an issue** to discuss your proposed solution before implementing
1. **Follow contribution guidelines** in `CONTRIBUTING.md`
1. **Include tests** for any changes

High-impact areas for contributors:

- 📊 Visualization utilities for robustness analysis
- 🔧 Convenience functions for common use cases
- 📚 Documentation improvements and examples
- 🧪 Additional test cases for edge scenarios

______________________________________________________________________

## 📞 Reporting New Issues

Found a new issue not listed here?

1. **Check GitHub Issues**: [seu-injection-framework/issues](https://github.com/wd7512/seu-injection-framework/issues)
1. **Search existing issues**: Your issue may already be reported
1. **Create new issue**: Include:
   - Python version, OS, and framework version
   - Minimal reproducible example
   - Full error messages and stack traces
   - Expected vs actual behavior
1. **Email maintainer**: wwdennis.home@gmail.com for questions

______________________________________________________________________

## 📖 Additional Resources

- **Installation Guide**: `docs/installation.md` - Comprehensive setup instructions and troubleshooting
- **Quick Start**: `docs/quickstart.md` - 10-minute tutorial to get started
- **Contributing Guide**: `CONTRIBUTING.md` - How to contribute to the project
- **Development Guide**: `dev_docs/AI_AGENT_GUIDE.md` - Technical architecture and development workflows
- **Changelog**: `CHANGELOG.md` - Full version history and changes

______________________________________________________________________

**Note**: The embedded TODO system provides developers with contextual information exactly where it's needed, reducing the maintenance burden of separate tracking documents.
