# Contributing to SEU Injection Framework

Thank you for contributing to the SEU Injection Framework!

## Project Goals

- Production-grade tools for systematic fault injection studies
- Reproducible research with deterministic SEU injection
- Performance-optimized operations for large-scale robustness analysis

## 📋 Quick Start for Contributors

### Prerequisites

- Python 3.9+ (3.9, 3.10, 3.11, 3.12 supported)
- [UV package manager](https://github.com/astral-sh/uv) (recommended) or pip
- Git for version control
- Basic understanding of PyTorch and neural network fault tolerance

### Development Environment Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/wd7512/seu-injection-framework.git
   cd seu-injection-framework
   ```

1. **Install dependencies with UV** (recommended):

   ```bash
   uv sync --all-extras
   ```

   Or with pip:

   ```bash
   pip install -e ".[dev,notebooks,extras]"
   ```

1. **Verify installation**:

   ```bash
   uv run python run_tests.py smoke
   ```

### **Essential Development Commands**

```bash
# Development workflow:
uv sync --all-extras                   # Install dependencies
uv run python run_tests.py smoke       # Quick validation (30s)
uv run python run_tests.py all         # Full suite

# Code quality:
uv run ruff check                      # Linting
uv run ruff format                     # Code formatting
uv run ty check src/seu_injection      # Type checking
```

## 🧪 Quality Requirements

All contributions must meet:

- ✅ **All tests pass**
- ✅ **Zero ruff violations**
- ✅ **Type hints** for public APIs
- ✅ **Docstrings** for public methods

### Running Quality Checks

Before submitting any pull request:

```bash
# Run all tests
uv run python run_tests.py all

# Check code quality
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run ty check src/seu_injection
```

## 🔄 Development Workflow

### Branching Strategy

This repository uses a two-branch model:

- **`dev`** — Integration branch (pseudo-main). All feature work, fixes, and hotfixes land here first. Version bumps happen here. Never deleted.
- **`main`** — Release branch. Only receives commits via `dev` → `main` merge. Protected — no direct pushes.
- **Feature branches** — Branch from `dev`, merge into `dev` via PR. Delete after merge. Never target `main` directly.

Workflow:

```
feature branch → PR into dev → accumulate features → PR from dev → main → tag → release
```

See [`AGENTS.md`](AGENTS.md) for the full workflow definition.

### Pull Request Process

1. Create feature branch from `dev`: `git checkout -b feature/your-feature-name dev`
1. Implement changes following quality standards
1. Test: `uv run python run_tests.py all`
1. Update documentation if needed
1. Submit pull request against `dev` with clear description
1. After PR is approved, merge into `dev`

### Commit Message Format

Use conventional commits for clarity:

```
type(scope): description

Examples:
feat(bitops): add optimized GPU bitflip operations
fix(injector): resolve tensor device mismatch issue
docs(api): enhance SEUInjector docstring examples
test(integration): add CNN robustness workflow tests
perf(float32): optimize bit manipulation performance
```

## 🎓 Types of Contributions

1. **Bug Reports & Fixes** — Include minimal reproducible examples
1. **New Features** — Discuss in issue first, maintain backward compatibility
1. **Performance Optimizations** — Benchmark and document improvements
1. **Research Contributions** — New methodologies, metrics, validation studies
1. **Documentation** — API enhancements, tutorials, examples

## 🔬 Research Contributions

Research contributions must be:

- **Reproducible** — Fully documented experimental design
- **Validated** — Tested across multiple model architectures
- **Cited** — Proper attribution to related work

Publication guidelines: Acknowledge the framework and consider contributing benchmark results.

## 📊 Performance Expectations

Performance contributions must include quantitative benchmarks and memory analysis.

Targets:

- Bitflip operations: <1ms per operation
- Memory overhead: <2x baseline
- Test suite: <15 seconds

## 🛡️ Security

Report security vulnerabilities to <wwdennis.home@gmail.com> (not public issues). Response within 48 hours.

## 📚 Resources

- **README.md** — Project overview
- **AGENTS.md** — Workflow rules for AI agents
- **docs/** — Sphinx documentation and tutorials

## 📖 Building Documentation

The project uses [Sphinx](https://www.sphinx-doc.org/) for documentation. The documentation is automatically built and published to [GitHub Pages](https://wd7512.github.io/seu-injection-framework/) on every push to the main branch.

### Building Locally

To build the documentation locally:

```bash
# Install documentation dependencies
pip install "seu-injection-framework[docs]"

# Or install Sphinx dependencies directly
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser

# Build the documentation
cd docs
make html

# View the documentation
# Open docs/build/html/index.html in your browser
```

The built HTML documentation will be in `docs/build/html/`.

### Documentation Structure

- `docs/source/` — Documentation source files (RST and Markdown)
- `docs/source/conf.py` — Sphinx configuration
- `docs/source/api/` — API reference documentation (auto-generated from docstrings)
- `docs/source/user_guide/` — User guides and tutorials

### Contributing to Documentation

When contributing documentation:

1. **API Documentation**: Add docstrings to your code following Google or NumPy style
1. **User Guides**: Add or update Markdown files in `docs/source/user_guide/`
1. **Build Locally**: Always build and review documentation before submitting
1. **Check for Warnings**: Address any Sphinx warnings in your build output

Example docstring format:

```python
def my_function(param1, param2):
    """Brief description of function.

    Longer description with more details about what the function does,
    its behavior, and any important notes.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Example:
        >>> result = my_function(1, 2)
        >>> print(result)
        3
    """
    return param1 + param2
```

## 📞 Support

- Documentation: README.md and docs/
- Issues: GitHub issues
- Email: <wwdennis.home@gmail.com>

Response times: 48 hours for bugs, 1 week for features/PRs.

______________________________________________________________________

Thank you for contributing!
