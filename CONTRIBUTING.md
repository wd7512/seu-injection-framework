# Contributing to SEU Injection Framework

Thank you for your interest in contributing to the SEU Injection Framework! This project aims to provide robust, high-quality tools for Single Event Upset (SEU) injection in neural networks for harsh environment applications.

## 🎯 Project Vision & Goals

The SEU Injection Framework serves the research community by providing:
- **Production-grade tools** for systematic fault injection studies
- **Reproducible research** with deterministic SEU injection
- **Performance-optimized operations** for large-scale robustness analysis
- **Community-driven development** with rigorous quality standards

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

2. **Install dependencies with UV** (recommended):
   ```bash
   uv sync --all-extras
   ```
   
   Or with pip:
   ```bash
   pip install -e ".[dev,notebooks,extras]"
   ```

3. **Verify installation**:
   ```bash
   uv run python run_tests.py smoke
   ```

### **Essential Development Commands**

```bash
# Standard development workflow:
uv sync --all-extras                       # Install dependencies
uv run python run_tests.py smoke          # Quick validation (10 tests, ~30s)
uv run python run_tests.py unit           # Unit tests (43 tests, ~2min)  
uv run python run_tests.py integration    # Integration tests (7 tests, ~5min)
uv run python run_tests.py all            # Full suite (109 tests, ~10s)
uv run pytest --cov=src/seu_injection --cov-fail-under=70   # Coverage validation (94% achieved)

# Code quality checks:
uv run ruff check                          # Linting
uv run ruff format                         # Code formatting
uv run mypy src/seu_injection             # Type checking (future)

# Performance testing:
uv run python run_tests.py benchmarks     # Performance validation

# Documentation generation:
# API documentation is automatically generated from docstrings in src/seu_injection/
# Ensure all public functions and classes have comprehensive docstrings
```

## 🧪 Quality Standards & Requirements

### Code Quality Gates

All contributions must meet these **mandatory** requirements:

#### **Testing Requirements**
- ✅ **94% minimum test coverage** (current: 94%)
- ✅ **All tests must pass** (109 tests, 2 skipped for CUDA)
- ✅ **New features require corresponding tests** (unit + integration)
- ✅ **Performance tests** for optimization-related changes

#### **Code Quality Standards**
- ✅ **Zero ruff violations** (`uv run ruff check src/ tests/`)
- ✅ **Proper formatting** (`uv run ruff format src/ tests/`)
- ✅ **Type hints** for all public APIs
- ✅ **Comprehensive docstrings** for public methods

#### **Security & Safety**
- ✅ **Bandit security scan** passes (`uv run bandit -r src/seu_injection`)
- ✅ **No critical security vulnerabilities**
- ✅ **Input validation** for all public APIs

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
```python
# TODO CATEGORY: Brief description of issue
# ISSUE: Detailed explanation of problem
# IMPACT: Effect on users or performance
# SOLUTION: Proposed fix or improvement
# PRIORITY: HIGH/MEDIUM/LOW - indicates urgency
```

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
   ```

2. **Implement your changes** following quality standards

3. **Test thoroughly**:
   ```bash
   uv run python run_tests.py all
   # Ensure 94%+ coverage maintained
   ```

4. **Update documentation** if needed:
   - Add docstrings for new public APIs
   - Update README.md for new features
   - Add examples for complex functionality

5. **Submit pull request** with:
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

### 1. **Bug Reports & Fixes**
- Use the bug report issue template
- Include minimal reproducible examples
- Test on multiple platforms when possible
- Provide performance impact assessment

### 2. **New Features**
- Discuss in feature request issue first
- Maintain backward compatibility
- Include comprehensive tests and documentation
- Consider research community needs

### 3. **Performance Optimizations**
- Benchmark against current implementation
- Document performance improvements quantitatively
- Ensure accuracy is maintained
- Add performance regression tests

### 4. **Research Contributions**
- New SEU injection methodologies
- Novel robustness metrics
- Integration with emerging ML frameworks
- Validation studies and benchmarks

### 5. **Documentation Improvements**
- API documentation enhancements
- Tutorial and example improvements
- Research methodology documentation
- Installation and setup guides

## 🔬 Research Community Guidelines

### Academic Standards
- **Reproducibility**: All research contributions must be fully reproducible
- **Methodology**: Clear documentation of experimental design and assumptions
- **Validation**: Comprehensive testing across multiple model architectures
- **Citation**: Proper attribution to related work and methodologies

### Research Contribution Process
1. **Proposal**: Create research issue with methodology overview
2. **Implementation**: Follow development workflow with emphasis on reproducibility
3. **Validation**: Comprehensive experiments with statistical significance testing
4. **Documentation**: Detailed methodology documentation and usage examples
5. **Review**: Peer review focusing on scientific rigor and practical applicability

### Publication Guidelines
If your contribution leads to academic publication:
- Acknowledge the SEU Injection Framework and contributors
- Share citation information for community benefit
- Consider contributing benchmark results back to the project

## 📊 Performance Expectations

### Benchmarking Requirements
All performance-related contributions must include:
- **Quantitative benchmarks** with statistical significance testing
- **Memory usage analysis** to prevent regressions
- **Cross-platform validation** (Windows, macOS, Linux)
- **GPU/CPU performance comparison** where applicable

### Performance Targets
- **Bitflip operations**: <1ms per operation for typical neural networks
- **Memory overhead**: <2x baseline during injection campaigns
- **Test suite**: Complete validation in <15 seconds on modern hardware
- **Import time**: Framework import <2 seconds

## 🛡️ Security Considerations

### Security Review Process
- All dependencies must be vetted for security vulnerabilities
- Input validation required for all public APIs
- No arbitrary code execution capabilities
- Secure random number generation for stochastic operations

### Reporting Security Issues
For security vulnerabilities:
1. **Do not** create public GitHub issues
2. **Email directly**: wwdennis.home@gmail.com
3. **Include**: Detailed description and reproduction steps
4. **Response**: We aim to respond within 48 hours

## 🤝 Community Standards

### Code of Conduct
We are committed to providing a welcoming and inclusive environment:
- **Respect**: Treat all community members with respect and professionalism
- **Collaboration**: Foster constructive discussion and knowledge sharing
- **Inclusivity**: Welcome contributors from all backgrounds and experience levels
- **Learning**: Support each other's growth and learning journey

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests, research discussions
- **Pull Request Reviews**: Code quality and technical discussions
- **Research Applications**: Share your use cases and findings with the community

## 🚀 Getting Started Checklist

New contributors should:

- [ ] Read through existing issues and pull requests
- [ ] Set up development environment with UV
- [ ] Run complete test suite successfully
- [ ] Choose a "good first issue" or small bug fix
- [ ] Follow the pull request process
- [ ] Engage with code review feedback constructively

## 📚 Additional Resources

### Documentation
- **README.md**: Project overview and quick start guide
- **CHANGELOG.md**: Version history and notable changes
- **docs/**: Comprehensive documentation and tutorials

### Research Resources
- **Research Paper**: Framework methodology and validation
- **Example Notebooks**: Practical usage demonstrations
- **Performance Benchmarks**: Baseline measurements and targets

### Development Tools
- **UV Package Manager**: Modern Python dependency management
- **Ruff**: Fast Python linter and formatter
- **pytest**: Testing framework with coverage reporting
- **MyPy**: Static type checking for Python

### AI Agent Development
- **AI Agent Guide**: [`AI_AGENT_GUIDE.md`](AI_AGENT_GUIDE.md) - Comprehensive guidance for AI agents working on this codebase (includes critical restrictions and mandatory workflows)

## 📞 Support & Questions

### Getting Help
- **Documentation**: Check README.md and docs/ first
- **Issues**: Create GitHub issue with detailed description
- **Research Questions**: Use research question issue template
- **Email**: wwdennis.home@gmail.com for direct communication

### Response Times
- **Bug reports**: Within 48 hours for initial response
- **Feature requests**: Within 1 week for initial feedback
- **Pull requests**: Within 1 week for initial review
- **Security issues**: Within 24 hours

---

## 🙏 Acknowledgments

We appreciate all contributions to the SEU Injection Framework. Contributors are acknowledged in:
- **CHANGELOG.md**: Notable contributions for each release
- **GitHub Contributors**: Automatic recognition for merged pull requests
- **Research Publications**: Co-authorship opportunities for significant research contributions

Thank you for helping make neural networks more robust in harsh environments!

---

*This project maintains high standards to serve the research community effectively. We appreciate your commitment to quality and collaboration.*