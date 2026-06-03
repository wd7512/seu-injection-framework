# Fault Injection Training for Improved Robustness

This example demonstrates how training with fault injection improves neural network
robustness to Single Event Upsets (SEUs) in harsh environments.

## 📋 Research Question

**How does training with fault injection improve the robustness of neural networks
to Single Event Upsets (SEUs)?**

## 🎯 Key Findings

> **These numbers are taken directly from the committed `robustness_results.csv`,
> which is the actual output of a single-seed run.** In this run, fault-aware
> training did **not** improve SEU robustness — on every measurable bit position
> the fault-aware model degraded slightly or was statistically indistinguishable
> from the baseline. See the [Limitations](#️-limitations) and the discussion
> below before drawing conclusions.

| Metric | Result |
|--------|--------|
| **Clean accuracy (both models)** | ~92% (maintained) |
| **Bit 1 (exp MSB), the only large baseline drop** | ~13% drop for *both* models; −0.5% change (no improvement) |
| **Bits 0/8 (sub-0.2% baseline drops)** | Fault-aware slightly worse; dominated by sampling noise |
| **Bits 15/23 (mantissa)** | No measurable impact for either model (floor effect) |
| **Inference overhead** | 0% (same architecture) |

### Hypothesis Assessment

- ❌ **H1: Robustness Improvement** — In this committed run, fault-aware training
  did not reduce the accuracy drop under SEUs. Bit 1 (exp MSB, the most critical
  position) showed essentially no change (−0.5%), and bits 0/8 showed small
  *degradations*. The hypothesised improvement is **not supported** by these data.
- 🔶 **H2: Weight Distribution** — Gradient noise is hypothesised to encourage
  flatter minima, but this study does not include weight-distribution or
  loss-landscape analysis. This mechanism is asserted, not tested.
- ❌ **H3: Generalization** — No bit position shows a reliable improvement, so
  there is no signal to generalise across positions. Mantissa bits (15, 23)
  show zero impact from either model (floor effect).
- ✅ **H4: Training Convergence** — Clean data accuracy maintained (~92%)

> **Note on reproducibility:** These results are stochastic and come from a
> single seed. Sub-1% baseline drops (bits 0, 8, 15, 23) are dominated by
> sampling noise and the corresponding "robustness factors" should not be
> over-interpreted. The only bit with a substantial baseline drop is bit 1
> (~13%), and there the two models are within ~0.5% of each other. A robust
> conclusion would require multiple seeds and confidence intervals (see
> Limitations).

---

## 📁 Files

### `fault_injection_training_study.py`
Complete standalone Python script that runs the full experiment.

**Usage:**
```bash
python fault_injection_training_study.py
```

**Output:**
- `training_comparison.png` — Training loss comparison
- `robustness_comparison.png` — Robustness metrics visualisation
- `robustness_results.csv` — Detailed experimental results

### `notebook.ipynb`
Interactive Jupyter notebook with step-by-step narrative.

**Usage:**
```bash
jupyter notebook notebook.ipynb
```

**Features:**
- Literature review
- Interactive code cells
- Inline visualisations
- Detailed conclusions

---

## 🔬 Methodology

### Baseline Training
Standard training without fault injection:
```python
def train_baseline(model, X, y, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
```

### Fault-Aware Training
Training with simulated fault effects via gradient noise:
```python
def train_fault_aware(model, X, y, fault_prob=0.01, fault_freq=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()

        # Inject noise to simulate fault effects
        if epoch % fault_freq == 0:
            for param in model.parameters():
                noise = torch.randn_like(param.grad) * fault_prob * param.grad.abs().mean()
                param.grad.add_(noise)

        optimizer.step()
```

### Robustness Evaluation
Using the SEU Injection Framework:
```python
injector = StochasticSEUInjector(model, classification_accuracy, x=X_test, y=y_test)
results = injector.run_injector(bit_i=bit_position, p=0.1)
```

---

## 🧪 Experimental Setup

### Model Architecture
- **Type:** Feedforward neural network (MLP)
- **Layers:** 2 → 64 → 32 → 16 → 1
- **Activation:** ReLU (hidden), Sigmoid (output)
- **Parameters:** ~2,800 total

### Dataset
- **Type:** Two Moons (sklearn.datasets.make_moons)
- **Samples:** 2,000 (1,400 train, 600 test)
- **Features:** 2D continuous
- **Task:** Binary classification

### Training Parameters
- **Epochs:** 100
- **Learning rate:** 0.01
- **Optimizer:** Adam
- **Fault probability:** 1%
- **Fault frequency:** Every 10 epochs

### Evaluation
- **Method:** Stochastic SEU injection
- **Sampling rate:** 10% of parameters
- **Bit positions:** 0 (sign), 1 (exp MSB), 8 (exp LSB), 15 (mantissa), 23 (mantissa LSB)

---

## 📊 Results Summary

The table below is transcribed directly from the committed `robustness_results.csv`
(single-seed run). Positive "Improvement (%)" means the fault-aware model had a
*smaller* accuracy drop; negative means it was *worse*. A robustness factor <1.0×
means the fault-aware model was less robust on that bit.

| Bit Position | Type          | Baseline Drop | Fault-Aware Drop | Improvement | Robustness Factor |
|--------------|---------------|---------------|------------------|-------------|-------------------|
| **0**        | Sign bit      | 0.11%         | 0.14%            | −30.2%*     | 0.77×             |
| **1**        | Exp MSB       | 13.39%        | 13.46%           | −0.5%       | 0.99×             |
| **8**        | Exp LSB       | 0.02%         | 0.05%            | −154.5%*    | 0.39×             |
| 15           | Mantissa      | 0.00%         | 0.00%            | —           | N/A               |
| 23           | Mantissa LSB  | 0.00%         | 0.00%            | —           | N/A               |

\* Asterisked "improvements" are computed on <0.2% baseline drops, where the
relative percentage is meaningless — a sub-0.05% absolute change is pure sampling
noise. The only bit with a substantial baseline drop is bit 1 (~13%), where the
two models differ by just −0.5% (no meaningful improvement).

**Bottom line:** in this single-seed run, fault-aware training did not improve SEU
robustness on this task. This is consistent with the high run-to-run variance
documented in the Limitations — the result should be read as "no measurable
benefit here," not as evidence that fault-aware training is ineffective in
general.

**Note:** Bit positions 15 and 23 showed no impact because flipping these less
significant mantissa bits has minimal effect on this simple dataset.

---

## 💡 Recommendations

> These are tentative, hypothesis-generating suggestions. The single-seed run in
> this example did **not** demonstrate a robustness benefit, so none of the
> following should be treated as a validated best practice without further
> multi-seed evaluation on your own task.

For research into deploying neural networks in harsh environments:

1. **Evaluate fault-aware training as a candidate** — do not adopt it blindly;
   measure robustness on your own task and architecture first
2. **Run multiple seeds** with confidence intervals before claiming any
   improvement — single-seed differences here were within the noise band
3. **Test robustness** across multiple bit positions before deployment
4. **Focus measurement on exponent bits** (especially bit 1) — these cause the
   largest accuracy drops and are where any real effect would be visible
5. **Monitor accuracy** in production environments

---

## 📚 Literature References

1. **Mitigating Multiple Single-Event Upsets** (arXiv 2502.09374, Feb 2025) —
   Up to 3× improvement with fault-aware training
2. **FAT-RABBIT** (ResearchGate 385101469) — Uniform weight importance reduces
   catastrophic failures
3. **DieHardNet** (HAL hal-04818068) — 100× reduction in critical errors with
   zero overhead
4. **Zero-Overhead Fault-Aware Solutions** (arXiv 2205.14420) — Vanilla models
   lose 37% performance without mitigation

*arXiv IDs use YYMM format: 2502 = February 2025, not the year 2502.*

---

## ⚠️ Limitations

This study has several important limitations that affect the generalisability of its findings:

1. **Synthetic dataset** — Two Moons is a low-dimensional (2D) binary classification
   benchmark. Results may not transfer to real-world tasks (image, text, speech).
2. **Small model** — ~2,800 parameters. Production-scale models (millions+) may
   behave differently under fault-aware training.
3. **Single architecture** — Only a feedforward MLP was tested. CNNs, RNNs, and
   Transformers are not covered.
4. **Single seed** — Results come from one random seed (42). Confidence intervals
   and run-to-run variance are not reported. The reported numbers should be
   treated as indicative, not definitive.
5. **Gradient-noise proxy** — Fault-aware training uses gradient noise rather than
   actual bit-flip injection during training. The hypothesised mechanism
   (flatter minima) is not directly verified with loss-landscape analysis.
6. **No ablation study** — The effects of `fault_prob`, `fault_freq`, model depth,
   and dataset complexity were not explored.
7. **No comparison baselines** — Standard robustness techniques (dropout, weight
   decay, label smoothing, adversarial training) were not compared.
8. **Device dependence** — Results were generated on Apple Silicon (MPS).
   Floating-point reduction order differs between MPS, CUDA, and CPU, so
   quantitative results may vary across hardware.

**Contributions extending this work** are welcome: larger models, more datasets,
multi-seed trials, and real hardware fault-injection experiments.

---

## 📦 Dependencies

```bash
pip install seu-injection-framework[analysis]
```

Includes: torch, numpy, matplotlib, seaborn, scikit-learn, pandas, tqdm, jupyter

---

## 📝 License

MIT License — see repository LICENSE file for details.

---

*Built for the research community studying neural network robustness in harsh environments.*
