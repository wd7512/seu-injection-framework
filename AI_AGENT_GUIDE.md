# 🤖 AI Agent Development Guide

**⚠️ FOR AI AGENTS ONLY - NOT FOR HUMAN USERS**

This document provides essential context, workflows, and critical restrictions for AI agents working on the SEU Injection Framework codebase.

---

## 🎯 **Current Project State (November 2025)**

### **Repository Status**
- **Branch**: `ai_refactor` (production-ready development branch)
- **Phase**: Phase 4 ready - Documentation and public release preparation
- **Test Status**: 109 tests (107 passed, 2 skipped), 94% coverage
- **Performance**: 10-100x speedup achieved in bitflip operations
- **Quality**: Enterprise-grade codebase with zero breaking changes

### **Critical Context**
- **Framework Purpose**: Single Event Upset (SEU) injection for neural network robustness testing
- **Target Users**: Researchers studying fault tolerance in harsh environments (space, nuclear, aviation)
- **Research Basis**: Academic paper on robust ML models in harsh environments
- **Production Ready**: Zero breaking changes, comprehensive testing, modern tooling

---

## ⚠️ **CRITICAL RESTRICTIONS FOR AI AGENTS**

### **🚨 GIT COMMAND RESTRICTIONS**
- **❌ NEVER execute any git commands** (`git add`, `git commit`, `git push`, `git checkout`, etc.)
- **❌ NEVER create or switch branches**
- **❌ NEVER modify repository state directly**
- **✅ ONLY provide suggested commit messages for human execution**
- **✅ ONLY modify files using file editing tools**

**Reason**: Human must maintain control over repository history and commits

### **🔒 FILE MODIFICATION RESTRICTIONS**
- **❌ NEVER modify `uv.lock`** (managed by UV package manager)
- **❌ NEVER modify `.git/`** directory or contents
- **⚠️  CAREFUL with `pyproject.toml`** (dependency changes require human review)
- **✅ SAFE to modify**: Source code, tests, documentation, configuration

### **🧪 TESTING REQUIREMENTS**
- **✅ ALWAYS run tests after ANY code changes**
- **✅ MANDATORY workflow**: `uv run python run_tests.py smoke` → make changes → `uv run python run_tests.py all`
- **❌ NEVER skip test validation** before suggesting changes
- **✅ REQUIRED**: 94% coverage must be maintained (currently 109 tests)

---

## 🗂️ **Essential File Structure & Context**

### **Critical Files & Their Purposes**
```
seu-injection-framework/
├── pyproject.toml              # UV dependencies, pytest config, coverage=50% enforcement
├── uv.lock                     # ⚠️ DO NOT MODIFY - UV managed dependencies
├── run_tests.py               # ALWAYS use this for testing (never raw pytest)
├── src/seu_injection/         # Modern package structure (Phase 2 complete)
│   ├── __init__.py           # Public API exports
│   ├── core/injector.py      # SEUInjector class - performance bottlenecks documented
│   ├── bitops/float32.py     # Bitflip operations - major TODOs for 10-100x speedup
│   ├── metrics/accuracy.py   # Evaluation metrics
│   └── utils/device.py       # Device management utilities
├── tests/                     # 109 comprehensive tests (94% coverage)
│   ├── conftest.py           # Shared fixtures
│   ├── test_*.py            # Unit tests
│   ├── integration/         # End-to-end tests
│   ├── smoke/               # Quick validation (10 tests)
│   └── benchmarks/          # Performance tests
└── docs/                     # Essential documentation only
    ├── DEVELOPMENT_ARCHIVE.md # Historical phases (Phases 1-3 complete)
    ├── installation.md       # User installation guide
    ├── quickstart.md        # User getting started
    └── KNOWN_ISSUES.md      # Current limitations
```

### **Package Import Evolution**
```python
# NEW (Phase 2 complete): Use this import structure
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy
from seu_injection.bitops.float32 import bitflip_float32

# OLD (removed): DO NOT use these imports
# from framework.attack import Injector  # ❌ REMOVED
# from framework.criterion import accuracy  # ❌ REMOVED
```

---

## ⚡ **Performance & Architecture Context**

### **Major Performance Bottlenecks (Documented in Code TODOs)**

1. **String-Based Bitflip Operations** (`src/seu_injection/bitops/float32.py`):
   - **Problem**: O(32) string manipulation per bit flip (100-500μs per scalar)
   - **Target**: O(1) direct bit manipulation (~3μs per scalar) 
   - **Impact**: 30-150x slower than possible
   - **Status**: Optimized version exists but not used in critical paths

2. **Critical Path Performance** (`src/seu_injection/core/injector.py`):
   - **Problem**: ResNet-18 injection takes 30-60 minutes per bit position
   - **Target**: 1-2 minutes per bit position
   - **Root Cause**: Uses slow `bitflip_float32()` instead of `bitflip_float32_optimized()`
   - **Solution**: Change function calls in injection loops

3. **Memory Inefficiencies**:
   - **Problem**: GPU→CPU tensor conversion during injection
   - **Impact**: Unnecessary memory transfers and CPU processing
   - **Solution**: Keep tensors on GPU, use torch operations

### **Architecture Issues (Documented in Code TODOs)**

4. **API Complexity** (`src/seu_injection/core/injector.py`):
   - **Problem**: High learning curve (IEEE 754, device management, criterion functions)
   - **Need**: Simplified convenience functions for common scenarios
   - **Priority**: HIGH for v1.0 user experience

5. **Type Safety Gaps**:
   - **Missing**: Comprehensive type hints throughout codebase
   - **Need**: Better IDE support and error prevention
   - **Priority**: MEDIUM for production readiness

---

## 🔄 **Mandatory Workflows for AI Agents**

### **Pre-Change Validation**
```bash
# 1. ALWAYS validate current state first
uv run python run_tests.py all
# Must show: 109 tests (107 passed, 2 skipped), 94% coverage

# 2. Check for linting issues
uv run ruff check src tests
# Must show: All clear or only acceptable warnings
```

### **Development Workflow**
```bash
# 3. Make incremental changes with testing
uv run python run_tests.py smoke  # Quick validation (30s)
# ... implement changes in small increments ...
uv run python run_tests.py unit   # Unit tests (1-2min)  
# ... continue changes ...
uv run python run_tests.py all    # Full validation before completion (3-5min)

# 4. Ensure coverage standards maintained
uv run pytest --cov=src/seu_injection --cov-fail-under=80
# Must maintain 94% coverage (well above 50% minimum)
```

### **Error Handling Protocol**
```bash
# If tests fail:
1. Read error messages carefully (run_tests.py provides enhanced error reporting)
2. Fix issues incrementally
3. Re-run smoke tests after each fix
4. NEVER suggest completion if any tests fail
5. Provide specific error context to human if unable to resolve
```

---

## 📝 **Code TODO System**

### **TODO Format Standard**
```python
# TODO CATEGORY: Brief description of issue
# PROBLEM: Specific technical problem description
# SOLUTION: Concrete implementation approach  
# IMPACT: Performance/usability implications
# PRIORITY: HIGH/MEDIUM/LOW based on user impact
# REFERENCE: Link to related TODOs in other files if applicable
```

### **TODO Categories in Use**
- **PERFORMANCE CRITICAL**: Issues affecting core framework speed (bitflip operations, injection loops)
- **ARCHITECTURE**: API design, type safety, error handling improvements
- **PIPELINE FIX**: Configuration and CI/CD related issues 
- **TESTING IMPROVEMENTS**: Coverage gaps, edge cases, performance testing
- **DIRECTORY STRUCTURE**: Organization and cleanup (mostly resolved)
- **VECTORIZATION SUCCESS**: Positive examples of optimal implementations

### **TODO Locations & Priorities**
```
🔥 HIGHEST PRIORITY:
- src/seu_injection/bitops/float32.py (string→bit manipulation conversion)
- src/seu_injection/core/injector.py (critical path performance fixes)

🟡 HIGH PRIORITY:
- src/seu_injection/__init__.py (API complexity improvements)
- tests/test_injector.py (coverage gap testing)

🟢 MEDIUM PRIORITY:
- pyproject.toml (pipeline configuration explanations)
- tests/conftest.py (testing infrastructure improvements)
```

---

## 🔍 **Quality Gates & Standards**

### **Test Requirements**
- **Minimum Coverage**: 50% (pipeline enforced)
- **Current Achievement**: 94% (well above minimum)
- **Test Count**: 109 total (107 passed, 2 skipped expected)
- **Test Categories**: Unit (majority), Integration (8), Smoke (10), Benchmarks (4)

### **Code Quality Standards**
- **Linting**: Zero ruff violations required
- **Type Safety**: Gradual improvement (not strictly enforced yet)
- **Documentation**: Comprehensive docstrings in public API methods
- **Performance**: No regressions in benchmark tests

### **Breaking Changes Policy**
- **❌ NO BREAKING CHANGES** allowed without explicit human approval
- **✅ Backward Compatible** improvements encouraged
- **✅ New Features** with comprehensive tests welcome
- **⚠️ API Changes** require careful consideration and migration path

---

## 🐛 **Critical Bug Patterns to Avoid**

### **Historical Issues (Fixed in Phases 1-3)**
Based on 6 critical bugs found and fixed during development:

1. **Tensor Validation**: 
   ```python
   # ❌ WRONG: Causes boolean ambiguity with tensors
   if X or y:
   
   # ✅ CORRECT: Explicit None checking
   if X is not None or y is not None:
   ```

2. **DataLoader Detection**:
   ```python
   # ❌ WRONG: DataLoader treated as tensor (subscriptable error)
   result = data_loader[0]
   
   # ✅ CORRECT: Check for DataLoader type first
   if hasattr(data_loader, '__iter__') and not hasattr(data_loader, 'shape'):
   ```

3. **IEEE 754 Precision**:
   ```python
   # ❌ WRONG: Too strict tolerance causes test failures
   assert abs(result - expected) < 1e-7
   
   # ✅ CORRECT: Appropriate tolerance for floating-point
   assert abs(result - expected) < 1e-6
   ```

4. **Device Handling**:
   ```python
   # ❌ WRONG: Inconsistent device types
   device = "cuda"  # String
   tensor.to(device)  # May cause issues
   
   # ✅ CORRECT: Consistent torch.device usage
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   tensor = tensor.to(device)
   ```

### **📚 Key Development Lessons Learned**

**What Worked Well:**
- **UV Package Management**: 20x faster dependency resolution vs pip
- **Comprehensive Testing**: 94% coverage caught 6 critical bugs early in development  
- **Incremental Migration**: Zero breaking changes maintained throughout 3 major phases
- **Living Documentation**: TODOs embedded in code reduce maintenance vs separate files
- **Directory Standards**: Following Python conventions (src/ layout, tests/ not testing/)

**Performance Insights:**
- **String-based bitflip operations are 100-500x slower** than direct bit manipulation
- **NumPy vectorization achieved 10-100x speedups** in critical paths
- **Memory efficiency**: Zero-copy operations keep memory usage <1.1x baseline
- **Test parallelization**: Category-based test execution improves development speed

**Architecture Decisions:**
- **src/seu_injection/ structure**: Clean separation of concerns, easier imports
- **TODO system in code**: Developers see issues while working on specific modules
- **Documentation consolidation**: Reduces maintenance burden significantly
- **Quality gates**: 94% coverage threshold catches regressions early

**Critical Recognition:**
- **Directory confusion**: tests/ vs testing/ caused weeks of development confusion
- **Import evolution**: Smooth migration path essential for user adoption
- **Documentation debt**: Separate planning files become stale quickly
- **Performance bottlenecks**: String operations in hot paths are major issue

---

## 🎯 **Development Priorities & Next Steps**

### **Phase 4: Documentation & Public Release** 🎯 READY
**Prerequisites Met**: All Phases 1-3 complete, 94% coverage, production-ready performance

**Immediate Priorities**:
1. **Professional API Documentation**: Enhanced docstrings throughout `src/seu_injection/`
2. **User Experience**: Simplified convenience functions for common use cases
3. **Distribution Ready**: PyPI preparation with UV/pip support
4. **Community Infrastructure**: Contributing guidelines, issue templates

### **Performance Optimization Opportunities** ⚡ HIGH IMPACT
1. **Replace slow bitflip calls**: Change `bitflip_float32()` → `bitflip_float32_optimized()` in critical loops
2. **Vectorize injection operations**: Process entire tensors instead of element-wise
3. **GPU optimization**: Keep operations on GPU instead of CPU conversion

### **API Enhancement Opportunities** 🎨 USER IMPACT  
1. **Quick analysis functions**: `quick_robustness_check(model, data)` → simple score
2. **Convenience constructors**: `SEUInjector.from_model(model)` with defaults
3. **Better error messages**: Custom exception types with helpful guidance

---

## 📊 **Success Metrics & Validation**

### **Development Success Criteria**
- ✅ **Test Pass Rate**: 100% of runnable tests (107/109, 2 skipped expected)
- ✅ **Coverage Maintenance**: ≥94% (current standard, well above 50% minimum)
- ✅ **Performance Standards**: No regression in benchmark tests
- ✅ **Code Quality**: Zero ruff violations, comprehensive docstrings
- ✅ **Backward Compatibility**: All existing functionality preserved

### **Completion Validation Commands**
```bash
# Required before any agent handoff:
uv run python run_tests.py all                    # Must: 107 passed, 2 skipped
uv run pytest --cov=src/seu_injection --cov-fail-under=90  # Must: >94% coverage
uv run ruff check src tests                       # Must: Zero violations
uv run python -c "from seu_injection import SEUInjector; print('✅ API functional')"
```

---

## 💡 **Agent Communication Guidelines**

### **Progress Reporting Format**
```
## Changes Made
- [Specific file]: [Specific change description]
- [Test results]: [Exact test count and coverage percentage]

## Current Status  
- ✅ [Completed items with validation]
- 🔄 [In-progress items with next steps]
- ❌ [Issues encountered with error details]

## Validation
- Tests: [Pass count]/109, Coverage: [percentage]%
- Linting: [Status]
- Functionality: [Basic smoke test result]

## Suggested Commit Message
[Detailed commit message for human to execute]
```

### **Error Escalation**
When unable to resolve issues:
1. **Document exact error messages** and commands that triggered them
2. **Provide context** about what was being attempted
3. **Suggest potential solutions** based on historical patterns
4. **Preserve work** by documenting changes made before encountering issues
5. **Never leave repository in broken state** (failing tests, linting errors)

---

## 🔄 **Agent Handoff Protocol**

### **Required Handoff Information**
1. **Current Branch State**: Confirm ai_refactor branch, test status, coverage
2. **Completed Work**: Specific files modified with validation results
3. **TODO Progress**: Which code TODOs were addressed or updated
4. **Test Status**: Exact command outputs for test runs and coverage
5. **Next Priorities**: Specific next steps based on current TODO priorities
6. **Blocked Issues**: Any problems requiring human intervention

### **Repository State Verification**
Before ending any session, agents must verify:
```bash
# Repository health check:
uv run python run_tests.py all    # ✅ 107 passed, 2 skipped
uv run ruff check src tests       # ✅ All clear
git status                        # 📋 Document any uncommitted changes
```

---

## 🏗️ **Research Context for Agents**

### **Framework Purpose & Applications**
- **SEU Injection**: Single Event Upset simulation for harsh environment testing
- **Target Domains**: Space missions (satellites, rovers), nuclear facilities, aviation systems
- **Research Value**: Systematic study of neural network fault tolerance under radiation
- **Academic Basis**: Published research on robust ML models in harsh environments

### **Technical Foundation**
- **IEEE 754 Bit Manipulation**: Core framework capability for precise fault simulation
- **PyTorch Integration**: Deep integration with modern ML framework
- **Performance Optimization**: NumPy vectorization for 10-100x speedup achievements
- **Multi-Platform Support**: Windows, Linux, macOS with Python 3.9-3.12

This context helps agents understand the scientific importance and technical requirements when making development decisions.

---

**🤖 Agent Status**: Ready for Phase 4 implementation with comprehensive guidance and critical restrictions in place.