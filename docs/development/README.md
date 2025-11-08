# SEU Injection Framework - Development Documentation

## 📋 **Quick Reference**

### **Current Status (November 2025)**
- **✅ Phase 1-2 COMPLETE**: Modern src/seu_injection/ package structure, comprehensive testing
- **🎯 Phase 3 NEXT**: Performance optimization (32x speedup target), advanced testing
- **Tests**: 63/64 passing, 80.60% coverage (exceeds 80% requirement)
- **Package**: Clean src/ layout with backward compatibility, legacy framework/ removed

### **Essential Commands**
```bash
# Standard development workflow:
uv sync --all-extras                       # Install dependencies
uv run python run_tests.py smoke          # Quick validation (10 tests, ~30s)
uv run python run_tests.py unit           # Unit tests (35 tests, ~2min)  
uv run python run_tests.py integration    # Integration tests (8 tests, ~5min)
uv run python run_tests.py all            # Full suite (64 tests, ~8min)
uv run pytest --cov=src/seu_injection --cov-fail-under=80   # Coverage validation
```

---

## 🎯 **Production Readiness Plan**

### **Project Vision**
Transform the SEU (Single Event Upset) injection framework from a research prototype into a production-ready Python package for studying neural network fault tolerance in harsh environments (space, nuclear, aviation, defense).

### **Phase Completion Status**

#### ✅ **Phase 1-2: Foundation & Structure (COMPLETED)**
- **✅ Modern Package Structure**: Complete src/seu_injection/ layout with proper modules
- **✅ UV Package Management**: Modern pyproject.toml with dependency groups and reproducible builds  
- **✅ Comprehensive Testing**: 64 tests across smoke/unit/integration/benchmarks with 80.60% coverage
- **✅ Bug Fixes**: 6 critical framework bugs identified and resolved
- **✅ Backward Compatibility**: Maintained old framework/* imports during transition
- **✅ Legacy Cleanup**: Successfully removed framework/ directory after validation
- **✅ API Evolution**: Updated layer_name__ → layer_name, enhanced device handling

#### 🎯 **Phase 3: Performance & Quality (NEXT PRIORITY)**
1. **Performance Optimization** (Highest Priority):
   - Replace string bitflip operations with NumPy bit manipulation (32x speedup target)
   - Implement `bitflip_float32_optimized()` with direct memory manipulation  
   - Add vectorized GPU operations for batch processing
   
2. **Advanced Testing & Quality**:
   - Complete mypy type checking integration
   - Add property-based testing with Hypothesis
   - Performance regression testing with benchmarks
   - Convert remaining framework/* imports to seu_injection/*

3. **Production Features**:
   - Enhanced documentation with API examples
   - CI/CD pipeline with GitHub Actions
   - PyPI packaging and distribution

---

## 🏗️ **Architecture Overview**

### **Package Structure**
```
src/seu_injection/                    # Main package
├── __init__.py                      # Public API with backward compatibility
├── core/                            # Core injection functionality  
│   ├── __init__.py
│   └── injector.py                 # SEUInjector class (enhanced from attack.py)
├── bitops/                         # Bit manipulation operations
│   ├── __init__.py
│   └── float32.py                  # Bitflip operations (needs Phase 3 optimization)
├── metrics/                        # Evaluation metrics
│   ├── __init__.py  
│   └── accuracy.py                 # Classification metrics (enhanced from criterion.py)
└── utils/                          # Utilities and helpers
    ├── __init__.py
    └── device.py                   # Device management and tensor utilities

tests/                              # Comprehensive test suite
├── smoke/                          # Quick validation tests (10 tests)
├── integration/                    # End-to-end workflow tests (8 tests)  
├── benchmarks/                     # Performance tests (5 tests)
├── test_*.py                       # Unit tests (35 tests)
└── conftest.py                     # Shared fixtures

testing/                            # Research infrastructure
├── example_networks.py             # Test models (100% coverage, essential for tests)
├── *.ipynb                         # Research notebooks (examples)
└── benchmark_results.jsonl         # Performance data
```

### **Import Patterns**
```python
# ✅ NEW (Recommended):
from seu_injection import SEUInjector, classification_accuracy, bitflip_float32

# ⚠️ OLD (Deprecated, but works):
from framework.attack import Injector
from framework.criterion import classification_accuracy  
```

---

## 🧪 **Testing Strategy**

### **Test Categories & Coverage**
- **Smoke Tests** (10): Basic functionality, imports, device compatibility
- **Unit Tests** (35): Individual component testing (bitops, metrics, injector)  
- **Integration Tests** (8): End-to-end workflows with real models
- **Benchmark Tests** (5): Performance validation and regression detection
- **Example Networks** (6): Complete coverage of testing infrastructure

### **Quality Gates**
- **80% Test Coverage**: Minimum threshold with enhanced error messaging (current: 80.60%)
- **100% Test Pass Rate**: All tests must pass before merging (63/64 passing, 1 skipped)
- **Performance Requirements**: Bitflip operations must complete in reasonable time
- **Import Validation**: Both old and new import patterns must work during transition

### **Running Tests**
```bash
# Quick validation
uv run python run_tests.py smoke

# Full test suite with coverage
uv run python run_tests.py all

# Specific categories  
uv run pytest tests/integration/ -v
uv run pytest tests/benchmarks/ -m "not slow"

# Coverage analysis
uv run pytest --cov=src/seu_injection --cov-report=html:htmlcov
```

---

## 🐛 **Critical Issues Fixed**

### **Phase 1-2 Bug Resolution**
1. **Tensor Conversion Bug**: Numpy arrays weren't converted to tensors properly
2. **DataLoader Type Error**: DataLoader treated as subscriptable tensor  
3. **Boolean Ambiguity**: `if X or y:` caused tensor boolean evaluation errors
4. **IEEE 754 Precision**: Adjusted tolerances from 1e-7 to 1e-6 for floating-point comparisons
5. **API Parameter Evolution**: layer_name__ → layer_name for cleaner interface
6. **Device Handling**: Enhanced torch.device consistency and error handling

### **Testing Improvements**  
- **NaN Resolution**: Fixed integration test failing with mean of empty arrays
- **Coverage Enhancement**: Added 6 tests for testing/example_networks.py (0% → 100%)
- **Benchmark Integration**: Created proper performance test suite
- **Error Messaging**: Enhanced coverage failure detection with actionable guidance

---

## 🚀 **Development Workflow**

### **Mandatory Workflow for Changes**
```bash
# 1. Validate current state
uv run python run_tests.py all   # Must show 63/64 tests passing

# 2. Create feature branch
git checkout -b feature/your-changes

# 3. Incremental development with testing
# ... implement changes ...
uv run python run_tests.py smoke # Quick validation after each change
uv run python run_tests.py all   # Full validation before commit

# 4. Ensure coverage requirement maintained  
uv run pytest --cov=src/seu_injection --cov-fail-under=80

# 5. Document changes and commit
git add . && git commit -m "feat: detailed description"
```

### **Package Management**
```bash
# Adding dependencies
uv add new-package                    # Core dependencies
uv add --group dev new-dev-package   # Development dependencies
uv add --group notebooks jupyter-ext # Notebook dependencies

# Environment maintenance
uv sync --all-extras                 # Install all dependency groups
uv lock                              # Update uv.lock after changes
```

---

## 📊 **Performance Targets**

### **Current State**
- **Bitflip Operations**: String-based O(n) approach (needs optimization)
- **Test Coverage**: 80.60% (exceeds 80% requirement) 
- **Test Suite**: ~8 minutes for complete validation
- **Package Size**: Compact src/ layout with clear module separation

### **Phase 3 Goals**  
- **Bitflip Speed**: 32x improvement via direct bit manipulation
- **Memory Efficiency**: <2x baseline memory usage during injection
- **GPU Utilization**: Vectorized operations for batch processing
- **Type Safety**: Complete mypy integration with strict checking

---

## 🤖 **AI Agent Handoff Guidelines**

### **Essential Context for Continuity**
- **Phase 2 Complete**: Modern package structure achieved with 80.60% coverage
- **Current Focus**: Performance optimization is highest priority for Phase 3
- **Quality Standards**: Maintain 80% coverage, all tests must pass
- **Backward Compatibility**: Support old imports until v2.0.0

### **Critical Files to Understand**
- `pyproject.toml`: UV dependencies, pytest config, coverage settings
- `run_tests.py`: Intelligent test runner with enhanced error messaging
- `src/seu_injection/`: Complete package structure with type hints
- `tests/`: Comprehensive test suite across multiple categories

### **Success Metrics**
- ✅ **Phase 1-2 Complete**: Modern structure, testing, bug fixes
- 🎯 **Next Target**: 32x bitflip performance improvement  
- 📊 **Quality Gates**: 80%+ coverage, 100% test pass rate
- 🔄 **Compatibility**: Seamless transition from old to new APIs

---

## 📚 **Key Lessons Learned**

### **Package Migration Success Factors**
- **Systematic Approach**: Build new structure before removing old code
- **Comprehensive Testing**: Validate both old and new import patterns work
- **Backward Compatibility**: Maintain transition period with deprecation warnings
- **Coverage Evolution**: 100% → 80% with enhanced error messaging more sustainable

### **Testing Framework Insights**  
- **Test Categories**: Smoke/Unit/Integration separation improves organization
- **Coverage Debugging**: Verbose pytest output essential for identifying gaps
- **Mock Testing**: CUDA availability mocking critical for CI/CD compatibility
- **Error Handling**: Enhanced failure messages speed up debugging

### **Quality Assurance Learnings**
- **80% Coverage Threshold**: Practical and maintainable with clear failure guidance
- **Change Documentation**: Comprehensive tracking critical for team workflows
- **UV Integration**: Modern tooling significantly improves developer experience
- **Import Validation**: Both old and new patterns must be continuously tested

---

## 🔗 **Related Resources**

- **Research Paper**: "A Framework for Developing Robust Machine Learning Models in Harsh Environments"
- **Repository**: https://github.com/wd7512/seu-injection-framework
- **Coverage Reports**: `htmlcov/index.html` (generated after test runs)
- **Package Documentation**: Auto-generated from docstrings in src/seu_injection/