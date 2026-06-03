# Issue #13: [FEATURE] Split injector class into a base class

### Feature Summary

Move the very large injector class into its down class. Therefore the very large docstrings can exist here and we can build more versatile injectors in the future that can inherit from this.

### Feature Category

API Enhancement

### Problem Statement

To build new injectors we must build a new function into the existing class, this is not scaleable and the file gets very very long with docstrings. 

### Proposed Solution

create an injector.py file with a base injector class and core methods. 

create a exhaustive_seu_injector.py file which performs the exhaustive seu injection. This class will inherit from the base injector

create the stochastic injector in stochastic_seu_injector.py 

adjust the unit tests with the new classes. ensure these all this pass without changing the test logic.

### Proposed API (if applicable)

```python

```

### Priority Level

None

### Research Impact

This will benefit the creation of new injectors for researchers

### Alternatives Considered

_No response_

### Implementation Considerations

- [ ] This feature requires new dependencies
- [ ] This feature affects performance-critical paths
- [ ] This feature requires GPU/CUDA support
- [x] This feature needs extensive testing
- [x] This feature affects the public API
- [ ] I would be interested in implementing this feature

### Additional Context

_No response_

### Pre-submission Checklist

- [x] I have searched existing issues for similar requests
- [x] I have provided sufficient detail about the use case
- [x] I have considered the impact on existing users
- [x] I have thought about backward compatibility

---

| Field | Value |
|-------|-------|
| **State** | closed |
| **Created** | 2025-11-15T22:07:15Z |
| **Updated** | 2025-11-15T23:50:12Z |
| **Labels** | enhancement |
| **Author** | @Will-D-AER |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/13 |
