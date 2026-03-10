# MNIST Activation Function Robustness Experiment

Compares SEU (Single Event Upset) robustness of different activation functions on MNIST.

## Activations Tested

| Activation | Definition | Notes |
|------------|------------|-------|
| **ReLU** | max(0, x) | Baseline |
| **LeakyReLU** | x if x≥0, else 0.01x | Prevents dying ReLU |
| **GELU** | x · Φ(x) | Gaussian Error Linear Unit |
| **RReLU** | clamp(x, 0, 3.0) | Reliable ReLU - capped at M=3.0 |
| **RWG** | max(0, x · e^(-x²)) | Randomized Weighted Gaussian |

## Model Architecture

- **MiniCNN**: 2 conv layers + 1 FC layer
- **Parameters**: ~2,010 (well under 5,000 limit)
- **Input**: 28×28 MNIST images
- **Output**: 10 classes

## Usage

### Full Experiment
```bash
python examples/mnist/activation_comparison.py
```

### Debug Mode (quick test)
```bash
python examples/mnist/activation_comparison.py --debug
```

### Custom Configuration
```bash
python examples/mnist/activation_comparison.py \
    --debug \
    --epochs 10 \
    --train-samples 500 \
    --test-samples 100
```

### Single Activation
```bash
python examples/mnist/activation_comparison.py --activation relu
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--debug` | False | Enable debug mode (smaller dataset, fewer epochs) |
| `--epochs` | 20 | Training epochs |
| `--train-samples` | 1000 | Number of training samples |
| `--test-samples` | 200 | Number of test samples |
| `--batch-size` | 32 | Batch size |
| `--activation` | None | Test single activation (relu, leakyrelu, gelu, rrelu, rwg) |
| `--output-dir` | examples/mnist/results | Output directory |

## Output

- `all_results.csv`: All injection results with accuracy scores
- `experiment_results.json`: Summary statistics per activation
- `activation_comparison.png`: Visualization plots

## Bit Positions Tested

- **0**: Sign bit
- **1**: Exponent MSB
- **5**: Exponent LSB  
- **15**: Mantissa MSB (significant magnitude change)

## Expected Runtime

| Mode | Time |
|------|------|
| Debug | ~2-3 minutes |
| Full | ~15-20 minutes |

## Research Questions

1. Which activation function is most robust to bit flips?
2. Does RReLU's cap provide fault tolerance benefits?
3. How does GELU's smooth nonlinearity affect error propagation?
4. Are certain bit positions more sensitive for specific activations?
