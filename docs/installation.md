# Installation Guide

This guide provides comprehensive instructions for installing the SEU Injection Framework on various platforms and environments.

## Quick Install (Recommended)

**Method 1: Install from PyPI (Easiest)**

The fastest way to get started for end users:

```bash
# Install the latest stable release (v1.1.8)
pip install seu-injection-framework

# Or with analysis tools
pip install "seu-injection-framework[analysis]"

# Verify installation
python -c "from seu_injection import SEUInjector; print('✅ Installation successful!')"
```

**Method 2: Development Setup with UV**

For contributors and advanced users:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone the repository
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework

# Install all dependencies (IMPORTANT: Use --all-extras for full functionality)
uv sync --all-extras

# Verify installation
uv run python run_tests.py smoke
```

> **⚠️ Important**: Always use `--all-extras` when running `uv sync` to avoid missing dependencies that cause test failures. See troubleshooting section below for common setup issues.

## Installation Methods

### Method 1: UV Package Manager (Recommended)

UV provides fast, reliable dependency resolution and virtual environment management.

**Advantages:**
- Fastest dependency resolution (10-100x faster than pip)
- Reproducible builds with `uv.lock`
- Automatic virtual environment management
- Cross-platform compatibility

**Installation:**

```bash
# Install UV
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version

# Clone and install framework
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework

# Install with all extras (recommended for development)
uv sync --all-extras

# Or install only core dependencies
uv sync
```

### Method 2: pip from PyPI (Production Use)

Install the stable release from PyPI:

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Unix/macOS
source .venv/bin/activate

# Install from PyPI
pip install seu-injection-framework

# Or with extras
pip install "seu-injection-framework[all]"
```

### Method 3: pip from Source (Development)

Install from the repository source:

```bash
# Clone repository
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Unix/macOS
source .venv/bin/activate

# Install in editable mode
pip install -e ".[all]"
```

### Method 4: Docker (Coming Soon)

Docker containers for reproducible research environments will be available in future releases.

## Installation Options

### Core Installation

Minimal installation with only essential dependencies:

```bash
# UV
uv sync

# pip
pip install -e .
```

**Includes:**
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- scikit-learn >= 1.1.0
- pandas >= 1.4.0
- matplotlib >= 3.5.0

### Development Installation

Includes testing, linting, and development tools:

```bash
# UV
uv sync --group dev

# pip
pip install -e ".[dev]"
```

**Additional tools:**
- pytest, pytest-cov (testing)
- ruff (linting)
- mypy (type checking)
- bandit (security)

### Notebook Installation

For Jupyter notebook support:

```bash
# UV
uv sync --group notebooks

# pip
pip install -e ".[notebooks]"
```

**Includes:**
- JupyterLab
- ipywidgets
- seaborn (advanced visualizations)

### Complete Installation

All features including extras:

```bash
# UV (recommended)
uv sync --all-extras

# pip
pip install -e ".[dev,notebooks,extras]"
```

## Platform-Specific Instructions

### Windows

**Prerequisites:**
- Python 3.9 or later
- Windows 10/11 or Windows Server 2019+
- PowerShell or Command Prompt

**CUDA Support (Optional):**
```powershell
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### macOS

**Prerequisites:**
- Python 3.9 or later
- macOS 11 (Big Sur) or later
- Xcode Command Line Tools: `xcode-select --install`

**Apple Silicon (M1/M2/M3):**
```bash
# MPS acceleration supported
# No additional configuration needed
```

### Linux

**Prerequisites:**
- Python 3.9 or later
- GCC compiler
- CUDA toolkit (optional, for GPU support)

**CUDA Support:**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verification

Verify your installation:

```bash
# Run smoke tests
uv run python run_tests.py smoke

# Or with standard Python
python -m pytest tests/smoke/ -v

# Quick import test
python -c "from seu_injection import SEUInjector; print('Installation successful!')"
```

## GPU Support

### CUDA Setup

For NVIDIA GPU acceleration:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check CUDA version
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (MPS)

Metal Performance Shaders (MPS) acceleration is automatically available on Apple Silicon Macs:

```python
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
```

## Troubleshooting

### Common Issues

#### Issue: "No module named pytest" or Tests Fail Immediately

**Problem:** You ran `uv sync` without `--all-extras`, missing development dependencies

**Solution:**
```bash
# Install ALL dependencies including testing tools
uv sync --all-extras

# Verify testing tools are available
uv run pytest --version
uv run python -c "import pytest; print('pytest available')"
```

#### Issue: "No module named 'testing'" Import Errors

**Problem:** Using an older version of the repository that lacks proper package structure

**Solution:**
```bash
# Switch to latest development branch
git checkout ai_refactor
git pull origin ai_refactor

# Reinstall with all dependencies
uv sync --all-extras

# Verify the testing module is importable
uv run python -c "from testing import get_example_network; print('testing module available')"
```

#### Issue: Individual Test Files Fail with Coverage Errors

**Problem:** Running single test files may fail coverage requirements even though code is correct

**Solutions:**
```bash
# Option 1: Run tests without coverage
uv run pytest tests/test_injector.py --no-cov -v

# Option 2: Run full test suite (recommended)
uv run pytest tests/ -v

# Option 3: Use the custom test runner
uv run python run_tests.py unit
```

#### Issue: Assert vs ValueError Security Warnings

**Problem:** Static analysis tools warn about assert statements in production code

**Solution:** This has been fixed in the ai_refactor branch. The framework now uses proper `if/raise ValueError` patterns instead of `assert` statements for input validation, ensuring validation works even with Python optimization flags.

#### Issue: ModuleNotFoundError for seu_injection

**Problem:** Python cannot find the seu_injection module

**Solution:**
```bash
# Ensure you're in the correct directory
cd seu-injection-framework

# Reinstall in editable mode
pip install -e .

# Or verify UV environment
uv sync --all-extras
```

#### Issue: CUDA out of memory

**Problem:** GPU runs out of memory during SEU injection

**Solution:**
- Reduce batch size in your experiments
- Use CPU mode: `device='cpu'`
- Use smaller models for testing
- Close other GPU-intensive applications

#### Issue: Permission denied during installation

**Problem:** Cannot install packages due to permissions

**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows

# Or use user installation
pip install --user -e .
```

#### Issue: Slow dependency resolution with pip

**Problem:** pip takes a long time to resolve dependencies

**Solution:**
- Switch to UV package manager (10-100x faster)
- Use pre-built wheels
- Check your internet connection
- Try a different PyPI mirror

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues:** [GitHub Issues](https://github.com/wd7512/seu-injection-framework/issues)
2. **Run diagnostics:**
   ```bash
   python -c "import sys; print(sys.version)"
   python -c "import torch; print(torch.__version__)"
   uv --version
   ```
3. **Create a new issue:** Include your Python version, OS, and error messages

## Next Steps

After successful installation:

1. **Quick Start:** Follow the [Quickstart Tutorial](quickstart.md)
2. **API Documentation:** Review the [API Reference](api/index.md)
3. **Examples:** Explore [Example Scripts](examples/)
4. **Research:** Check out [Research Notebooks](examples/notebooks/)

## System Requirements

### Minimum Requirements
- Python 3.9+
- 4 GB RAM
- 1 GB free disk space
- CPU with SSE4.2 support

### Recommended Requirements
- Python 3.11+
- 8 GB RAM
- 5 GB free disk space
- NVIDIA GPU with 4GB+ VRAM (for GPU acceleration)
- CUDA 11.8 or later

### Tested Platforms
- ✅ Windows 10/11 (x64)
- ✅ macOS 11+ (Intel & Apple Silicon)
- ✅ Ubuntu 20.04/22.04 LTS
- ✅ Debian 11+
- ✅ Fedora 35+

## Version Compatibility

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.9 | ✅ Supported | Minimum required version |
| 3.10 | ✅ Supported | Fully tested |
| 3.11 | ✅ Supported | Recommended |
| 3.12 | ✅ Supported | Latest features |
| 3.13+ | ⚠️ Experimental | May work but not officially tested |

## Development Setup

For contributors and developers:

```bash
# Clone with development tools
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework

# Install with all development dependencies
uv sync --all-extras

# Install pre-commit hooks (coming soon)
# pre-commit install

# Run full test suite
uv run python run_tests.py all

# Check code quality
uv run ruff check src/ tests/
```

## Uninstallation

To remove the framework:

```bash
# If installed with pip
pip uninstall seu-injection-framework

# Remove virtual environment
rm -rf .venv  # Unix/macOS
rmdir /s .venv  # Windows

# Remove cloned repository
cd ..
rm -rf seu-injection-framework
```

---

**Last Updated:** November 2025  
**Version:** 1.0.0 (Phase 3 Complete)
