# Contributing to SEU Injection Framework

Thank you for contributing to the SEU Injection Framework!

## 🎯 Project Goals

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
uv run python run_tests.py all         # Full suite (116 tests)

# Code quality:
uv run ruff check                      # Linting
uv run ruff format                     # Code formatting
uv run mypy src/seu_injection         # Type checking
```

## 🧪 Quality Requirements

All contributions must meet:

- ✅ **94% test coverage** maintained
- ✅ **All tests pass** (116 tests)
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
uv run mypy src/seu_injection --ignore-missing-imports

# Security analysis
uv run bandit -r src/seu_injection -f txt
```

## � Code Quality & TODO System

The framework uses an embedded TODO system throughout the codebase to track improvements:

### TODO Format Standards

````python
# TODO CATEGORY: Brief description of issue
# ISSUE: Detailed explanation of problem
# IMPACT: Effect on users or performance
## 📊 TODO System

The framework uses embedded TODOs to track improvements. See `docs/KNOWN_ISSUES.md` for details.

Format:

```python
# TODO CATEGORY: Brief description
# PRIORITY: HIGH/MEDIUM/LOW
````

## 🔄 Development Workflow

### Branching Strategy

- **`main`**: Stable, production-ready code
- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `bugfix/issue-description`

### Pull Request Process

1. Create feature branch: `git checkout -b feature/your-feature-name`
1. Implement changes following quality standards
1. Test: `uv run python run_tests.py all`
1. Update documentation if needed
1. Submit pull request with clear description

````

### TODO Categories

- **PERFORMANCE**: Critical path optimizations and bottlenecks (highest priority)
- **CODE QUALITY**: Import optimization, dead code, refactoring needs
- **ERROR HANDLING**: Exception consistency and input validation improvements
- **TEST QUALITY**: Test improvements and coverage enhancements
- **MAINTAINABILITY**: Code organization and documentation improvements

### Working with TODOs

- **Before Changes**: Review relevant TODOs in files you're modifying
- **During Development**: Add TODOs for issues you discover but can't fix immediately
- **After Changes**: Update or remove TODOs that your changes address
- **Code Review**: TODOs are normal and expected - they indicate active development priorities

**Note**: TODOs are part of our living documentation system and are not considered technical debt unless marked with HIGH priority.

## �🔄 Development Workflow

### Branching Strategy

- **`main`**: Stable, production-ready code
- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `bugfix/issue-description`
- **Research contributions**: `research/methodology-name`

### Pull Request Process

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
````

1. **Implement your changes** following quality standards

1. **Test thoroughly**:

   ```bash
   uv run python run_tests.py all
   # Ensure 94%+ coverage maintained
   ```

1. **Update documentation** if needed:

   - Add docstrings for new public APIs
   - Update README.md for new features
   - Add examples for complex functionality

1. **Submit pull request** with:

   - Clear description of changes
   - Test coverage report
   - Performance impact assessment
   - Research applications (if applicable)

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

1. **Bug Reports & Fixes** - Include minimal reproducible examples
1. **New Features** - Discuss in issue first, maintain backward compatibility
1. **Performance Optimizations** - Benchmark and document improvements
1. **Research Contributions** - New methodologies, metrics, validation studies
1. **Documentation** - API enhancements, tutorials, examples

## 🔬 Research Contributions

Research contributions must be:

- **Reproducible** - Fully documented experimental design
- **Validated** - Tested across multiple model architectures
- **Cited** - Proper attribution to related work

Publication guidelines: Acknowledge the framework and consider contributing benchmark results.

## 📊 Performance Expectations

Performance contributions must include quantitative benchmarks and memory analysis.

Targets:

- Bitflip operations: \<1ms per operation
- Memory overhead: \<2x baseline
- Test suite: \<15 seconds

## 🛡️ Security

Report security vulnerabilities to wwdennis.home@gmail.com (not public issues). Response within 48 hours.

## 🤝 Community

Treat all community members with respect. Use GitHub Issues for bug reports, feature requests, and discussions.

## 🚀 Getting Started

1. Set up development environment with UV
1. Run test suite: `uv run python run_tests.py all`
1. Choose a "good first issue"
1. Submit pull request

## 📚 Resources

- **README.md** - Project overview
- **docs/** - Documentation and tutorials
- **dev_docs/AI_AGENT_GUIDE.md** - For AI agents

## 📞 Support

- Documentation: README.md and docs/
- Issues: GitHub issues
- Email: wwdennis.home@gmail.com

Response times: 48 hours for bugs, 1 week for features/PRs.

______________________________________________________________________

Thank you for contributing!
