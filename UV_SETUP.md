# UV Setup Guide for SEU Injection Framework

This project now uses [UV](https://github.com/astral-sh/uv) for fast, reliable Python package management.

## Installation

### 1. Install UV (if not already installed)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Unix/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Project Dependencies

**Basic installation (core dependencies only):**
```bash
uv sync
```

**With development tools:**
```bash
uv sync --extra dev
```

**With Jupyter notebook support:**
```bash
uv sync --extra notebooks
```

**Full installation (all extras):**
```bash
uv sync --all-extras
```

## Usage

**Run Python with project dependencies:**
```bash
uv run python your_script.py
```

**Run tests:**
```bash
uv run pytest
```

**Format code:**
```bash
uv run black .
uv run isort .
```

**Add new dependencies:**
```bash
uv add package-name
uv add --dev package-name  # for development dependencies
```

## Migration from pip

The old `requirements.txt` and `pytorch_requirements.txt` files are now consolidated into `pyproject.toml`. 
All dependencies are automatically managed by UV with proper version resolution.

**Benefits of UV:**
- 10-100x faster dependency resolution
- Reliable cross-platform lock files
- Better conflict resolution
- Integrated tool management
- Compatible with standard Python packaging