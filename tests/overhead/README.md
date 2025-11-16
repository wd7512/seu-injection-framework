# SEU Injection Overhead Measurement

This script measures the performance overhead of SEU injection operations compared to baseline inference.

## Usage

```bash
python measure_overhead.py
```

## Output

The script generates a single JSON file in `tests/overhead/results`:

- `overhead_results_<timestamp>.json` — Structured data with timing metrics and system info

## What it measures

- Baseline inference time without SEU injection (with warmup)
- Inference time with SEU injection enabled (with warmup)
- Overhead (absolute and relative)
- System specifications (OS, CPU, RAM, Python version)
- Laptop detection and battery/plugged-in status (if available)

## Methodology

- Baseline and injection timings use a warmup phase for accuracy
- Number of timing iterations is doubled (100 for baseline)
- Results are saved with a timestamp in the filename
