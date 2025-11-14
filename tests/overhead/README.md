# SEU Injection Overhead Measurement

This script measures the performance overhead of SEU injection operations compared to baseline inference.

## Usage

```bash
python measure_overhead.py
```

## Output

The script generates two files:

- `overhead_results.json` - Structured data with timing metrics
- `overhead_results.csv` - Tabular format for analysis

## What it measures

- Baseline inference time without SEU injection
- Inference time with SEU injection enabled
- Overhead (absolute and relative)
