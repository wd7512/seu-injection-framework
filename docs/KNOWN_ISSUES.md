# Known Issues

## Known Issues (Summary)

This file lists only unique, actionable issues not covered in other documentation. For installation, usage, contributing, and roadmap, see the respective guides.

______________________________________________________________________

### 1. Examples Directory - Pandas Import Performance

Some example scripts (e.g. `basic_cnn_robustness.py`) may hang during pandas import in certain Python environments. Use `uv run python` for faster startup. Does not affect framework usage as a library.

### 2. Performance Optimization

Some functions (e.g. `bitflip_float32`) are not fully vectorized. Optimized versions exist but further improvements are possible, especially for GPU workloads.

### 3. Error Handling

Exception types are not fully standardized. Custom exception classes are planned for future releases.

### 4. Test Coverage

Some edge cases (NaN/Inf, multi-GPU, custom criterion) are not fully tested. Coverage is high (>90%) but not exhaustive.

### 5. Unused Code

Some utility functions (e.g. `get_model_info` in `device.py`) appear unused and may be removed in future cleanups.

______________________________________________________________________

For details on planned improvements, contributing, or reporting issues, see:

- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `docs/installation.md`
- `docs/quickstart.md`

For the full TODO system and categories, refer to code comments and the contributing guide.
