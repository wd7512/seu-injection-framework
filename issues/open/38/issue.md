# Issue #38: [BUG] Change Update Dependencies Workflow to Run Full Matrix Tests

### Bug Description

The existing update deps workflow only does partial testing. This does not ensure that updating the dependencies wont break part of the codebase. 

### Steps to Reproduce

```python
Run the workflow.
```

### Expected Behavior

Run the full matrix of tests before making a pr into main and document that they all pass. 

### Error Message

```shell

```

### Environment

Windows

### Python Version

3.11.8

### SEU Injection Framework Version

1.0.0+

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

---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2025-12-08T23:14:08Z |
| **Updated** | 2025-12-08T23:14:52Z |
| **Labels** | bug |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/38 |
