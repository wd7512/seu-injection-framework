# Quickstart Guide

Get started with the SEU Injection Framework in 10-15 minutes! This guide walks you through a complete workflow from installation to analyzing neural network robustness under Single Event Upsets.

## Prerequisites

- Python 3.9 or later
- Basic familiarity with PyTorch
- 10-15 minutes

See the [Installation Guide](./installation.md) if you haven't installed the framework yet.

## What You'll Build

In this tutorial, you'll:

1. Create a simple neural network
1. Train it on a toy dataset
1. Inject bit flips systematically
1. Analyze robustness results

**Time to complete:** ~10 minutes

______________________________________________________________________

## Step 1: Setup and Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from seu_injection.core import ExhaustiveSEUInjector
from seu_injection.metrics import classification_accuracy
```

## Step 2: Create Training Data

```python
x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train, x_test = map(torch.tensor, (x_train, x_test), [torch.float32] * 2)
y_train, y_test = map(torch.tensor, (y_train, y_test), [torch.float32] * 2)
```

## Step 3: Build and Train a Model

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x_train), y_train)
    loss.backward()
    optimizer.step()
```

## Step 4: Basic SEU Injection

```python
injector = ExhaustiveSEUInjector(
    trained_model=model, x=x_test, y=y_test, criterion=classification_accuracy
)
results = injector.run_injector(bit_i=0)
```

## Step 5: Visualize Results

```python
import matplotlib.pyplot as plt
bit_positions = [0, 1, 2, 11, 21, 31]
mean_accuracies = [
    np.mean(injector.run_injector(bit_i=bit)['criterion_score']) for bit in bit_positions
]
plt.plot(bit_positions, mean_accuracies, marker='o')
plt.xlabel('Bit Position')
plt.ylabel('Mean Accuracy')
plt.title('Robustness Across Bit Positions')
plt.show()
```

______________________________________________________________________

## Next Steps

Congratulations! 🎉 You've completed the quickstart tutorial. You now know how to:

- Set up SEU injection experiments
- Inject bit flips systematically
- Analyze robustness across bit positions

### Continue Learning

- [Example Notebook on GitHub](https://github.com/wd7512/seu-injection-framework/blob/main/examples/Example_Attack_Notebook.ipynb)
- [ShipsNet Experiments on GitHub](https://github.com/wd7512/seu-injection-framework/tree/main/examples/shipsnet/)
