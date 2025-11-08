# Legacy Files Directory

This directory contains historical files from the development process that are no longer actively used but preserved for reference.

## File Descriptions

### Removed Files (No Longer Needed)
- `benchmark.py` - Superseded by comprehensive benchmarks in `tests/benchmarks/`
- `validate_tests.py` - No longer needed after successful migration to modern testing framework
- `requirements.txt` - Replaced by UV dependency management in `pyproject.toml`
- `pytorch_requirements.txt` - Integrated into UV dependency groups

These files represented earlier development phases but have been superseded by the modern framework structure completed in Phase 3.

## Migration Context

The SEU injection framework has undergone significant evolution:
- **Legacy Era (v0.0.1-0.0.6)**: Basic prototype with manual dependency management
- **Phase 1-2**: Migration to modern package structure with `src/seu_injection/`
- **Phase 3**: Performance optimization and production readiness

All functionality from these legacy files has been integrated into the current framework or replaced by superior implementations.