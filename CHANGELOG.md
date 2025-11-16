# Changelog

All notable changes to the SEU Injection Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.10] - 2025-11-15

- **injector.py file refactored**: replaced by `injector_base.py` abstractclass with two new python files inheriting from this
- **overhead calculation script added**: look in `tests/overhead`

## [1.1.9] - 2025-11-13

*Pipeline overhaul*

- **Consolidated github actions**: two `python-tests.yml` and `pylint.yml` deleted and `ci.yml` has three test levels
- **mdformat added**: mdformat added to ensure consistent `.md` style, similar to ruff for `.py` files
- **Unit tests folder created**: unit tests moved to their own folder and coverage changed to 50% globally
- **Shipsnet folder created**: with readme added

## [1.1.8] - 2025-11-11

- **Branching Rework**: `dev` branch created and most older branches removed
- **Markdown Cleanup**: development markdown files kept on `dev` branch but removed from `main`
- **versions_and_plan.md created**: manual tracking of ideas going forward

## [1.1.7] - 2025-11-11

### Fixed

- **Release Workflow**: Optimized wheel verification to use CPU-only PyTorch
- **Disk Space**: Added cleanup steps to prevent out-of-space errors during builds
- **Version Consistency**: Ensured all version numbers are synchronized across files

### Changed

- **CI/CD**: Improved release workflow with best practices from popular PyTorch packages
- **Verification**: Streamlined wheel verification process for faster builds

## [1.1.6] - 2025-11-11

### Fixed

- **Release Workflow**: Added disk space cleanup before wheel verification
- Prevented "No space left on device" errors during GitHub Actions builds

## [1.1.5] - 2025-11-11

### Fixed

- **Dependencies**: Updated dependency specifications for better compatibility
- **Build Process**: Minor improvements to build configuration

## [1.0.0] - 2025-11-09

### 🎉 **Initial Public Release**

First stable release of the SEU Injection Framework for Single Event Upset (SEU) injection in neural networks for harsh environment applications.

### Added

#### **Core Framework**

- Complete `SEUInjector` class for systematic fault injection in PyTorch models
- IEEE 754 float32 bit manipulation with optimized performance (10-100x speedup)
- Classification accuracy metrics for robustness evaluation
- Comprehensive device management (CPU/GPU) with automatic detection

#### **API Features**

- **Deterministic SEU injection** with precise bit position control
- **Stochastic SEU injection** with configurable probability distributions
- **Layer-specific targeting** for focused vulnerability analysis
- **Batch processing support** for efficient large-scale studies
- **Multiple data input formats** (tensors, numpy arrays, DataLoaders)

#### **Testing & Quality Assurance**

- **109 comprehensive tests** covering all functionality (94% code coverage)
- **Smoke, unit, integration, and benchmark test suites**
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Performance regression testing** with automated benchmarks

#### **Documentation & Examples**

- **Professional API documentation** with comprehensive docstrings
- **Installation guide** supporting UV, pip, and development setup
- **Quick start tutorial** for immediate productivity
- **Research methodology documentation** covering SEU physics and injection strategies

#### **Development Infrastructure**

- **Modern package structure** with `src/seu_injection/` layout
- **UV package manager integration** for fast, reproducible builds
- **Automated code quality** with ruff, mypy, and bandit
- **Comprehensive CI/CD** with automated testing and quality gates

### Technical Specifications

#### **Performance Achievements**

- **Bitflip Operations**: 10-100x performance improvement via optimized bit manipulation
- **Memory Efficiency**: \<2x baseline memory usage during injection campaigns
- **Test Suite**: Complete validation in \<15 seconds on modern hardware
- **Coverage**: 94% test coverage with enhanced error reporting

#### **Compatibility**

- **Python**: 3.9, 3.10, 3.11, 3.12
- **PyTorch**: >=2.0.0 with torchvision >=0.15.0
- **Dependencies**: Modern scientific Python stack (NumPy, SciPy, scikit-learn)
- **Platforms**: Windows, macOS, Linux with full GPU support

#### **Research Applications**

- **Space missions**: Radiation-hardened neural networks for spacecraft
- **Nuclear facilities**: Fault-tolerant ML models for harsh environments
- **Aviation systems**: Robustness analysis for safety-critical applications
- **Defense systems**: Resilience evaluation for mission-critical deployments

## Development History

### Phase 1: Foundation Setup (Internal)

- Migrated from pip to UV with modern `pyproject.toml`
- Implemented comprehensive test suite architecture
- Fixed critical framework bugs during testing integration
- Established 80% coverage threshold with enhanced error messaging

### Phase 2: Package Structure Modernization (Internal)

- Complete migration from `framework/` to `src/seu_injection/` structure
- Implemented proper package hierarchy with logical module separation
- Added comprehensive type hints and professional documentation
- Maintained 100% backward compatibility during transition

### Phase 3: Performance Optimization (Internal)

- Achieved 10-100x speedup in bitflip operations via direct bit manipulation
- Optimized memory usage and GPU utilization patterns
- Enhanced test infrastructure with benchmark validation
- Completed production-ready performance targets

### Phase 4.1: Documentation Enhancement (Internal)

- Added professional API documentation with comprehensive docstrings
- Enhanced test infrastructure with robust dependency management
- Improved test runner logic with accurate coverage reporting
- Achieved 94% test coverage with quality integration

### Phase 4.2: Distribution Preparation (Current)

- Prepared PyPI release infrastructure with proper metadata
- Created community guidelines and contribution workflows
- Developed comprehensive examples and research documentation
- Established GitHub issue templates and release automation

## Research Background

This framework implements the methodology described in:
**"A Framework for Developing Robust Machine Learning Models in Harsh Environments"**

### Key Innovations

- **Systematic SEU injection** with IEEE 754 compliance
- **Statistical robustness analysis** with comprehensive metrics
- **Performance-optimized operations** for large-scale studies
- **Research reproducibility** with deterministic random seeds

### Citation

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.0.0}
}
```

## Installation & Usage

For installation instructions and usage examples, see:

- [Installation Guide](docs/installation.md) - Comprehensive setup instructions
- [Quick Start Guide](docs/quickstart.md) - 10-minute tutorial
- [Main README](README.md) - Project overview and basic usage

## Support

- **Documentation**: https://github.com/wd7512/seu-injection-framework
- **Issues**: https://github.com/wd7512/seu-injection-framework/issues
- **Research Questions**: Use issue template for research discussions
- **Contributions**: See CONTRIBUTING.md for development workflow

## License

MIT License - see LICENSE file for details.

______________________________________________________________________

*This release represents the culmination of comprehensive development phases focused on performance, quality, and research community adoption.*

## [1.1.6] - 2025-11-10

### Fixed

- Added disk space cleanup step in release workflow to prevent "No space left on device" errors during wheel verification
- Removed unnecessary CUDA packages from test installations by using CPU-only PyTorch for wheel verification
- Improved release workflow reliability for large dependency installations

### Changed

- Release workflow now frees up ~10-20GB before wheel installation by removing unused system packages
- Optimized wheel verification step to use `--no-deps` flag with pre-installed dependencies

### Advisory

This is a CI/CD improvement release; no functional changes to the package itself.

______________________________________________________________________

## [1.1.5] - 2025-11-10

### Reverted

- Reverted changes from 1.1.5 due to release workflow issues
- This version was tagged but not successfully published to PyPI

### Advisory

Skip this version; use 1.1.6 or 1.1.4 instead.

______________________________________________________________________

## [1.1.4] - 2025-11-10

### Internal / Maintenance

- Completed type hint revisions across `bitops/float32.py`, `metrics/accuracy.py`, and `core/injector.py` (mypy: 0 issues).
- Replaced deprecated `typing.Dict`/`List` generics with native `dict`/`list`.
- Resolved ruff import ordering and formatting warnings (all checks pass).
- Minor refactor in `SEUInjector` to initialize optional attributes explicitly and fix indentation errors.
- No public API changes; behavior remains identical to 1.1.3.

### Verification

- Lint: `ruff check` reports no issues.
- Types: `mypy src/seu_injection` returns success.
- Build: Wheel & sdist generated successfully.

### Advisory

Safe upgrade; purely code quality improvements.

______________________________________________________________________

## [1.1.3] - 2025-11-10

### Fixed

- Release workflow corrected to install dev & analysis extras before running smoke tests (previous failure: missing `pytest`).
- Added lint (ruff) and type (mypy) steps to release pipeline for stronger gatekeeping.

### Verification

- Full local build and import version check will precede tagging.
- Workflow now uses `uv sync --extra dev --extra analysis --frozen` ensuring test dependencies present.

### Advisory

This patch is workflow/packaging automation oriented; no library API or behavior changes versus 1.1.1.

______________________________________________________________________

## [1.1.1] - 2025-11-10

### Fixed / Maintenance

- Re-run release with corrected GitHub workflow sequencing after tagging before Trusted Publishing configuration.
- Clarified release workflow filename usage for PyPI Trusted Publishing (requires `release.yml`).
- No functional code changes; package contents identical to 1.1.0 besides version metadata.

### Integrity Verification

- Rebuilt sdist and wheel (`uv build`) for 1.1.1 and validated import version.
- Lint & type checks re-confirmed clean prior to publish.

### Advisory

If you successfully installed 1.1.0 from a local build, upgrading to 1.1.1 is optional; this is a metadata/publishing correction release.

______________________________________________________________________

## [1.1.0] - 2025-11-10

### Changed

- Slimmed core runtime dependencies to only essential libraries (torch, numpy, scipy, tqdm) for faster, lighter installs.
- Moved heavier data/analysis libs (pandas, scikit-learn, matplotlib, seaborn, statsmodels, scikit-image, joblib, torchvision) into categorized optional extras (`analysis`, `vision`, `notebooks`, `dev`, `docs`, `all`).
- Dynamic version retrieval in `__init__.py` via `importlib.metadata` to prevent drift and simplify release management.
- README installation section refactored for PyPI-first workflow; separated development setup.

### Added

- `py.typed` marker enabling type information distribution and added `Typing :: Typed` Trove classifier.
- Fallback internal `accuracy_score` implementation allowing minimal install without scikit-learn while preserving behavior when analysis extras installed.
- GitHub Actions CI workflow (`ci.yml`) for multi-Python tests, lint (ruff), type checks (mypy), and security scan (bandit).
- GitHub Actions release workflow (`release.yml`) for tag-triggered build & publish using PyPI Trusted Publishing.

### Internal / Maintenance

- Confirmed sdist/wheel build via `uv build` and validated dynamic import of version.
- Updated packaging metadata and extras grouping for clearer user ergonomics.

### Notes

- If upgrading from 1.0.0 and you rely on scikit-learn/pandas/matplotlib functionality, install with an appropriate extra, e.g.:
  ```bash
  pip install "seu-injection-framework[analysis]"
  ```
- Core API surface remains backward compatible; no breaking changes introduced.

### Citation Update

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.0}
}
```

______________________________________________________________________
