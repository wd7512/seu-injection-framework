# API Reference Documentation

This directory contains comprehensive API documentation for all modules in the SEU Injection Framework. The documentation is organized by module and provides detailed information about classes, functions, and their usage.

## Documentation Structure

### Core Modules
- [`injector.md`](./injector.md) - SEUInjector class and fault injection methods
- [`criterion.md`](./criterion.md) - Criterion classes and evaluation functions

### Bit Manipulation
- [`bitops.md`](./bitops.md) - IEEE 754 float32 bit manipulation operations
- [`float32.md`](./bitops/float32.md) - Detailed float32 bitflip functions

### Metrics and Evaluation
- [`accuracy.md`](./metrics/accuracy.md) - Classification accuracy functions
- [`metrics.md`](./metrics.md) - Overview of all evaluation metrics

### Utilities
- [`utils.md`](./utils.md) - Utility functions and helper classes

## Navigation Guide

### By Use Case

**Getting Started**
- Start with [`injector.md`](./injector.md) for the main SEUInjector class
- Review [`accuracy.md`](./metrics/accuracy.md) for evaluation functions
- See [`basic_usage.md`](../tutorials/basic_usage.md) for complete examples

**Bit Manipulation**
- [`bitops.md`](./bitops.md) - Overview of bit manipulation capabilities
- [`float32.md`](./bitops/float32.md) - Detailed IEEE 754 operations

**Custom Evaluation**
- [`criterion.md`](./criterion.md) - Criterion classes for custom metrics
- [`accuracy.md`](./metrics/accuracy.md) - Built-in accuracy functions

### By Module

```
seu_injection/
├── core/
│   ├── injector.py          → docs/api/injector.md
│   └── criterion.py         → docs/api/criterion.md
├── bitops/
│   └── float32.py          → docs/api/bitops/float32.md
├── metrics/
│   └── accuracy.py         → docs/api/metrics/accuracy.md
└── utils/                  → docs/api/utils.md
```

## API Documentation Standards

All API documentation in this directory follows these standards:

### Format
- **Markdown format** with consistent section structure
- **Code examples** for all public functions and classes  
- **Parameter details** with types, ranges, and descriptions
- **Return value specifications** with types and meanings
- **Exception documentation** for all possible errors
- **Performance notes** where relevant

### Section Structure
Each module documentation includes:

1. **Overview** - Purpose and key features
2. **Quick Start** - Basic usage examples  
3. **Classes** - Detailed class documentation
4. **Functions** - Function-by-function reference
5. **Examples** - Comprehensive usage examples
6. **Performance** - Optimization guidance
7. **See Also** - Related documentation links

### Cross-References
- Links to related modules and functions
- References to tutorials and examples
- Links to external documentation (PyTorch, NumPy, etc.)

## Generation and Maintenance

### Automated Generation
This documentation can be automatically generated from docstrings using:

```bash
# Generate API docs from source code
python scripts/generate_api_docs.py

# Update specific module documentation  
python scripts/generate_api_docs.py --module seu_injection.core.injector

# Generate with examples
python scripts/generate_api_docs.py --include-examples
```

### Manual Maintenance
- Update documentation when adding new functions or classes
- Ensure examples remain current with API changes
- Review cross-references for accuracy
- Validate all code examples for correctness

## Documentation Tools

### Sphinx Integration
For automated HTML generation:

```bash
# Install Sphinx and extensions
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Generate HTML documentation
cd docs/
sphinx-build -b html . _build/html
```

### Development Setup
```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build and serve documentation locally
cd docs/
python -m http.server 8000
# Navigate to http://localhost:8000/_build/html/
```

## Contributing to Documentation

### Adding New Modules
1. Create corresponding `.md` file in appropriate subdirectory
2. Follow the established section structure
3. Include comprehensive examples
4. Add cross-references to related modules
5. Update this index file

### Updating Existing Documentation
1. Ensure docstrings in source code are comprehensive
2. Update `.md` files to reflect API changes
3. Validate all code examples
4. Update cross-references as needed
5. Regenerate automated documentation

### Documentation Review Checklist
- [ ] All public functions documented
- [ ] Code examples test successfully  
- [ ] Parameter types and ranges specified
- [ ] Return values clearly described
- [ ] Exceptions documented with conditions
- [ ] Performance characteristics noted
- [ ] Cross-references accurate and helpful
- [ ] Examples demonstrate real-world usage

---

This API documentation provides comprehensive reference material for developers using the SEU Injection Framework. For tutorials and getting-started guides, see the [`tutorials/`](../tutorials/) directory.