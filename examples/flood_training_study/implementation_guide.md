# Implementation Guide: Flood Level Training for SEU Robustness

[Back to README](README.md)

---

## Quick Start

### Minimal Example (5 minutes)

```python
import torch
import torch.nn as nn

# 1. Define FloodingLoss wrapper
class FloodingLoss(nn.Module):
    def __init__(self, base_loss, flood_level=0.08):
        super().__init__()
        self.base_loss = base_loss
        self.flood_level = flood_level
    
    def forward(self, predictions, targets):
        loss = self.base_loss(predictions, targets)
        return torch.abs(loss - self.flood_level) + self.flood_level

# 2. Use in training (replace nn.CrossEntropyLoss)
# Before:
# criterion = nn.CrossEntropyLoss()

# After:
criterion = FloodingLoss(nn.CrossEntropyLoss(), flood_level=0.08)

# 3. Train normally - everything else unchanged!
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)  # Flooding applied automatically
        loss.backward()
        optimizer.step()
```

**That's it!** You now have flood training.

---

## Complete Implementation

### Full FloodingLoss Class with Features

```python
import torch
import torch.nn as nn
from typing import Optional

class FloodingLoss(nn.Module):
    """
    Implements flooding regularization for any base loss function.
    
    Flooding prevents models from achieving arbitrarily low training loss
    by maintaining a minimum loss threshold (flood level b).
    
    Loss formula: L_flood = |L(θ) - b| + b
    
    Args:
        base_loss: Base loss function (e.g., nn.CrossEntropyLoss())
        flood_level: Target flood level (b), typically 0.05-0.15
        adaptive: If True, automatically adjust flood level (experimental)
        warmup_epochs: Number of epochs before flooding starts (default: 0)
        
    Example:
        >>> base_criterion = nn.CrossEntropyLoss()
        >>> criterion = FloodingLoss(base_criterion, flood_level=0.08)
        >>> loss = criterion(predictions, targets)
        
    References:
        Ishida et al. (2020): "Do We Need Zero Training Loss After 
        Achieving Zero Training Error?" NeurIPS 2020.
    """
    
    def __init__(
        self, 
        base_loss: nn.Module, 
        flood_level: float = 0.08,
        adaptive: bool = False,
        warmup_epochs: int = 0
    ):
        super().__init__()
        self.base_loss = base_loss
        self.flood_level = flood_level
        self.adaptive = adaptive
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute flooded loss."""
        loss = self.base_loss(predictions, targets)
        
        # Optional warmup: no flooding for first N epochs
        if self.current_epoch < self.warmup_epochs:
            return loss
        
        # Apply flooding
        flooded_loss = torch.abs(loss - self.flood_level) + self.flood_level
        return flooded_loss
    
    def step_epoch(self, val_loss: Optional[float] = None):
        """Call this at the end of each epoch."""
        self.current_epoch += 1
        
        # Adaptive flood level adjustment (experimental)
        if self.adaptive and val_loss is not None:
            # Adjust flood level based on validation loss
            target_flood = val_loss * 1.5
            # Smooth update (10% learning rate)
            self.flood_level = 0.9 * self.flood_level + 0.1 * target_flood
            # Clamp to reasonable range
            self.flood_level = max(0.05, min(0.20, self.flood_level))
```

### Usage in Training Loop

```python
# Setup
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = FloodingLoss(nn.CrossEntropyLoss(), flood_level=0.08)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader, criterion.base_loss)  # Use base loss for validation
    
    # Update flood level (if adaptive)
    criterion.step_epoch(val_loss)
    
    print(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, Flood Level={criterion.flood_level:.4f}")
```

---

## Hyperparameter Selection

### Choosing the Flood Level (b)

**Method 1: Based on Validation Loss (Recommended)**

1. Train a baseline model without flooding
2. Measure final validation loss: `val_loss_final`
3. Set flood level: `b = 1.5 * val_loss_final` to `2.0 * val_loss_final`

```python
# Example
baseline_val_loss = 0.05  # From baseline training
flood_level = 1.5 * baseline_val_loss  # = 0.075
```

**Method 2: Grid Search**

Test multiple values and choose based on validation accuracy + robustness:

```python
flood_levels = [0.05, 0.08, 0.10, 0.12, 0.15]
results = {}

for b in flood_levels:
    criterion = FloodingLoss(nn.CrossEntropyLoss(), flood_level=b)
    model = train(criterion)
    val_acc = evaluate_accuracy(model, val_loader)
    seu_robustness = evaluate_seu_robustness(model)  # Using SEU Injection Framework
    results[b] = {'val_acc': val_acc, 'seu_robustness': seu_robustness}

# Select best based on combined metric
best_b = max(results.keys(), key=lambda b: results[b]['seu_robustness'] - 0.5 * (1 - results[b]['val_acc']))
```

**Method 3: Task-Specific Guidelines**

| Task Type | Suggested Range | Example |
|-----------|----------------|---------|
| Image Classification (Simple) | 0.05 - 0.10 | MNIST: 0.05 |
| Image Classification (Complex) | 0.10 - 0.20 | CIFAR-10: 0.15, ImageNet: 0.20 |
| NLP (Classification) | 0.08 - 0.15 | Sentiment: 0.10 |
| Regression | 0.01 - 0.05 | Depends on scale |
| Binary Classification | 0.05 - 0.12 | This study: 0.08 |

**Rule of Thumb**: Start with `b=0.08` and adjust based on validation performance.

---

## Integration with SEU Injection Framework

### Complete Workflow

```python
from seu_injection.core import StochasticSEUInjector
from seu_injection.metrics import classification_accuracy

# 1. Train with flooding
model = train_with_flooding(flood_level=0.08)

# 2. Evaluate baseline accuracy
model.eval()
baseline_acc = classification_accuracy(model, x_test, y_test)
print(f"Baseline Accuracy: {baseline_acc:.2%}")

# 3. Evaluate SEU robustness
injector = StochasticSEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Test critical bit positions
for bit_pos, bit_name in [(0, 'Sign'), (1, 'Exp MSB'), (15, 'Mantissa')]:
    results = injector.run_injector(bit_i=bit_pos, p=0.05)  # 5% sampling
    mean_acc = np.mean(results['criterion_score'])
    accuracy_drop = baseline_acc - mean_acc
    print(f"Bit {bit_pos} ({bit_name}): Mean Acc={mean_acc:.2%}, Drop={accuracy_drop:.2%}")

# 4. Compare with standard training
standard_model = train_standard()
# ... repeat injection protocol ...
```

### Automated Comparison Script

```python
def compare_training_methods(model_fn, train_loader, test_loader, x_test, y_test):
    """Compare standard vs flood training for SEU robustness."""
    
    results = {}
    
    for method in ['standard', 'flood']:
        print(f"\n{'='*60}\n{method.upper()} TRAINING\n{'='*60}")
        
        # Train
        if method == 'standard':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = FloodingLoss(nn.CrossEntropyLoss(), flood_level=0.08)
        
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train(model, train_loader, criterion, optimizer, epochs=100)
        
        # Baseline
        baseline = evaluate_accuracy(model, test_loader)
        
        # SEU robustness
        injector = StochasticSEUInjector(model, classification_accuracy, x_test, y_test)
        seu_results = injector.run_injector(bit_i=0, p=0.05)
        mean_acc = np.mean(seu_results['criterion_score'])
        
        results[method] = {
            'baseline': baseline,
            'seu_accuracy': mean_acc,
            'accuracy_drop': baseline - mean_acc
        }
    
    # Report
    print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
    print(f"Baseline Accuracy:")
    print(f"  Standard: {results['standard']['baseline']:.2%}")
    print(f"  Flood:    {results['flood']['baseline']:.2%}")
    print(f"\nSEU Robustness (Accuracy Drop):")
    print(f"  Standard: {results['standard']['accuracy_drop']:.2%}")
    print(f"  Flood:    {results['flood']['accuracy_drop']:.2%}")
    improvement = (results['standard']['accuracy_drop'] - results['flood']['accuracy_drop']) / results['standard']['accuracy_drop'] * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    return results
```

---

## Production Deployment

### Pre-Deployment Checklist

Before deploying a flood-trained model to production:

- [ ] **Validate baseline accuracy** is within acceptable range (< 1% degradation)
- [ ] **Confirm SEU robustness improvement** via systematic injection testing
- [ ] **Test with multiple random seeds** (at least 3) to ensure reproducibility
- [ ] **Evaluate on held-out validation set** from target environment
- [ ] **Document training configuration**:
  - [ ] Flood level used
  - [ ] Base loss function
  - [ ] Optimizer and learning rate
  - [ ] Number of epochs
  - [ ] Random seeds
- [ ] **Combine with hardware protections** (ECC memory, TMR) for critical systems
- [ ] **Establish monitoring** for deployed performance
- [ ] **Define failure thresholds** and recovery procedures

### Deployment Code Template

```python
class ProductionModel:
    """Production-ready model with flood training and monitoring."""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.baseline_acc = self.load_baseline_metrics()
        self.failure_count = 0
        self.prediction_count = 0
        
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make prediction with monitoring."""
        with torch.no_grad():
            outputs = self.model(inputs)
        
        self.prediction_count += 1
        
        # Optional: Monitor for anomalies
        if self.detect_anomaly(outputs):
            self.failure_count += 1
            self.log_warning("Potential SEU detected")
        
        return outputs
    
    def health_check(self) -> dict:
        """Check model health."""
        return {
            'predictions': self.prediction_count,
            'failures': self.failure_count,
            'failure_rate': self.failure_count / max(1, self.prediction_count),
            'status': 'healthy' if self.failure_count < 100 else 'degraded'
        }
```

---

## Troubleshooting

### Issue: Flood training not converging

**Symptoms**: Training loss stays high, validation accuracy poor

**Solutions**:
1. **Reduce flood level**: Try b = 0.05 instead of 0.08
2. **Increase training epochs**: Model may need more time to find good minimum
3. **Check learning rate**: May be too low, try increasing
4. **Verify base loss**: Ensure base_loss is correctly configured

### Issue: No robustness improvement

**Symptoms**: Flood-trained model performs same as standard under SEU

**Solutions**:
1. **Check if already well-regularized**: If using heavy dropout/weight decay, flooding may have diminishing returns
2. **Increase flood level**: Try b = 0.10 or 0.12
3. **Verify injection protocol**: Ensure SEU injection is working correctly
4. **Test different bit positions**: Check sign bit (0) specifically

### Issue: Too much baseline accuracy loss

**Symptoms**: Flood training reduces baseline accuracy >1%

**Solutions**:
1. **Reduce flood level**: Try b = 0.05 or lower
2. **Use warmup**: Start flooding after 10-20 epochs
3. **Adaptive flooding**: Let flood level adjust automatically
4. **Verify task difficulty**: May be too easy/hard for flooding

---

## FAQs

**Q: Can I use flood training with any loss function?**  
A: Yes! FloodingLoss wraps any base loss (CrossEntropy, MSE, BCE, etc.)

**Q: Does flood training work with all optimizers?**  
A: Yes, it's loss-level regularization, independent of optimizer choice.

**Q: Can I combine flood training with dropout/weight decay?**  
A: Yes, they're complementary. Flood training adds regularization on top.

**Q: How much does flood training slow down training?**  
A: About 4-6% overhead, mostly from the abs() operation.

**Q: Should I use flooding for inference?**  
A: No, flooding only affects training. Use base_loss for validation/inference.

**Q: What if I don't know my optimal flood level?**  
A: Start with b=0.08. If accuracy drops too much, reduce to 0.05. If no robustness gain, increase to 0.10-0.12.

**Q: Can flood training replace hardware protections (ECC, TMR)?**  
A: No, it's complementary. Use both for defense-in-depth in critical systems.

**Q: Does flood training help with adversarial robustness?**  
A: Potentially, but this study focused on SEU robustness. Worth investigating!

---

## Additional Resources

- **[experiment.py](experiment.py)**: Complete working example
- **[README.md](README.md)**: Study overview and navigation
- **[04_results.md](04_results.md)**: Detailed experimental results
- **[references.md](references.md)**: Academic references

---

**Last Updated**: December 11, 2025  
**Maintained by**: SEU Injection Framework Team  
**License**: MIT (code), CC BY 4.0 (documentation)
