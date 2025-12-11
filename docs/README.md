# SEU Injection Framework Documentation

This directory contains the Sphinx documentation for the SEU Injection Framework.

## 📖 Viewing the Documentation

The latest documentation is available at: **https://wd7512.github.io/seu-injection-framework/**

## 🏗️ Building Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

Or install with the docs extra:

```bash
pip install "seu-injection-framework[docs]"
```

### Build Commands

```bash
# Build HTML documentation
make html

# Clean build artifacts
make clean

# View the documentation
# Open build/html/index.html in your browser
```

## 📁 Directory Structure

```
docs/
├── source/              # Documentation source files
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Main documentation index
│   ├── api/            # API reference documentation (auto-generated from docstrings)
│   │   ├── core.rst    # Core injector classes
│   │   ├── bitops.rst  # Bit operation functions
│   │   └── metrics.rst # Evaluation metrics
│   ├── installation.md
│   ├── quickstart.md
│   ├── contributing.md
│   └── known_issues.md
├── build/              # Built documentation (generated, not in git)
│   └── html/           # HTML output
└── Makefile            # Build automation
```

## 📝 Contributing to Documentation

### API Documentation

API documentation is automatically generated from docstrings in the source code. To improve API docs:

1. Add or update docstrings in `src/seu_injection/`
2. Follow Google or NumPy docstring style
3. Include examples in docstrings
4. Rebuild docs to see changes

Example:

```python
def my_function(param1, param2):
    """Brief description of the function.

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
    pass
```

### Adding New Pages

1. Create a new `.rst` or `.md` file in `source/`
2. Add it to a `toctree` directive in `index.rst` or another parent page
3. Build to verify

## 🚀 Deployment

Documentation is automatically built and deployed to GitHub Pages via GitHub Actions:

- **Trigger**: Push to `main` branch or manual workflow dispatch
- **Workflow**: `.github/workflows/docs.yml`
- **URL**: https://wd7512.github.io/seu-injection-framework/

The workflow:
1. Checks out the repository
2. Installs Sphinx and dependencies
3. Builds HTML documentation
4. Deploys to GitHub Pages

## 🛠️ Sphinx Configuration

Key configuration in `source/conf.py`:

- **Theme**: `sphinx_rtd_theme` (Read the Docs theme)
- **Extensions**:
  - `sphinx.ext.autodoc` - Auto-generate API docs from docstrings
  - `sphinx.ext.napoleon` - Support for Google/NumPy style docstrings
  - `sphinx.ext.viewcode` - Add links to source code
  - `sphinx.ext.intersphinx` - Link to other project docs (PyTorch, NumPy)
  - `sphinx_autodoc_typehints` - Include type hints in docs
  - `myst_parser` - Support Markdown files

## 🔍 Troubleshooting

### Build Warnings

Common warnings and fixes:

- **"failed to import module"**: Install package dependencies (`torch`, `numpy`)
- **"cross-reference target not found"**: Fix internal links or add files
- **"Lexing literal_block failed"**: Check code block formatting in Markdown

### Clean Build

If you encounter persistent issues:

```bash
make clean
rm -rf build/
make html
```

## 📚 Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [MyST Parser (Markdown support)](https://myst-parser.readthedocs.io/)
- [Sphinx AutoDoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
