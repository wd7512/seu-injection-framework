# SEU Injection Framework - Overhead Performance Tracker

This file tracks the performance overhead of SEU injection operations across different releases to monitor regressions and improvements.

## Latest Benchmark Results

**Date**: 2025-11-13\
**Version**: 1.1.9\
**Hardware**: CPU (x86_64)\
**Python**: 3.12

### Small Networks (< 1K parameters)

| Network Type | Parameters | Baseline Inference | Overhead (absolute) | Overhead (%) | Status |
| ------------ | ---------- | ------------------ | ------------------- | ------------ | ------ |
| Small MLP    | 641        | 0.03 ms            | 1.5 ms              | ~5000%       | ✅     |
| Medium MLP   | 2,817      | 0.04 ms            | 1.2 ms              | ~3000%       | ✅     |

### Medium Networks (1K-100K parameters)

| Network Type | Parameters | Baseline Inference | Overhead (absolute) | Overhead (%) | Status |
| ------------ | ---------- | ------------------ | ------------------- | ------------ | ------ |
| Large MLP    | 11,265     | 0.05 ms            | 1.4 ms              | ~2700%       | ✅     |
| Small CNN    | ~50K       | 1.5 ms             | 42 ms               | ~2700%       | ✅     |

### Key Metrics

- **Average overhead per injection**: 1-2 ms for MLPs, 40-45 ms for CNNs
- **Throughput**: 700-1450 injections/sec (MLPs), ~25 injections/sec (CNNs)
- **Memory overhead**: Minimal (parameter backup only)

### Notes

- High relative percentages (2000-5000%) are expected due to very fast baseline inference (\<0.1ms for small models)
- Absolute overhead time is the relevant metric for planning large-scale studies
- Stochastic sampling (1% probability) used for measurements
- All measurements on CPU; GPU acceleration can provide 10-100× speedup for large models

## Historical Trends

### Version 1.1.9 (2025-11-13)

- Initial implementation of overhead measurement utilities
- Baseline metrics established for small-to-medium networks

## How to Update This Tracker

Run the overhead calculation example and update the table:

```bash
python examples/overhead_calculation_example.py
```

Or run the performance benchmarks:

```bash
python -m pytest tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_seu_injection_overhead -v -s
```

## Performance Goals

- ✅ Overhead < 2ms per injection for small MLPs
- ✅ Overhead < 50ms per injection for small CNNs
- 🎯 Optimize to < 1ms per injection for MLPs (future)
- 🎯 Add GPU benchmark results (future)
