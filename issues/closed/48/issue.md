# Issue #48: [BUG] Low p-values in stochastic injector fails smoke tests

### Bug Description

FAILED tests/integration/test_workflows.py::TestSEUInjectionWorkflows::test_robustness_analysis_pipeline - AssertionError: Should have at least some analysis results
assert 0 > 0

- where 0 = len({})

### Steps to Reproduce

```python
Run Tests
```

### Expected Behavior

Stochastic injector should run at least one injection per layer, even if p value is very small. Add an additional default input to the run function i.e. `run_at_least_one_injection = True`

Smoke tests not fail due to low p-values

### Error Message

```shell

```

### Environment

Windows

### Python Version

3.11.8

### SEU Injection Framework Version

1+

### PyTorch Version

_No response_

### Installation Method

Source install (git clone + uv/pip install -e)

### CUDA/GPU Information

- [ ] Issue occurs with GPU/CUDA
- [ ] Issue occurs with CPU only
- [ ] Issue is related to device handling

### Additional Context

_No response_

### Pre-submission Checklist

- [x] I have searched existing issues for duplicates
- [x] I have provided a minimal reproducible example
- [x] I have included all relevant version information
- [x] I have tested with the latest version

______________________________________________________________________

| Field       | Value                                                       |
| ----------- | ----------------------------------------------------------- |
| **State**   | closed                                                      |
| **Created** | 2025-12-09T13:33:46Z                                        |
| **Updated** | 2025-12-09T14:29:10Z                                        |
| **Labels**  | bug                                                         |
| **Author**  | @wd7512                                                     |
| **URL**     | https://github.com/wd7512/seu-injection-framework/issues/48 |
