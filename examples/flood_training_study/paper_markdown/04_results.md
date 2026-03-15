# 4. Results

[← Previous: Methodology](03_methodology.md) | [Back to README](README.md) | [Next: Discussion →](05_discussion.md)

______________________________________________________________________

## 4.1 Overview

We conducted 36 systematic experiments across 3 datasets, 6 flood levels, and 2 dropout configurations. All results are available in `comprehensive_results.csv` and `comprehensive_results.json`.

**Key Metrics Analyzed:**

- Baseline accuracy (no SEU injection)
- Final training and validation losses
- Mean accuracy drop under SEU injection (15% sampling rate, ~345 injections per bit position)
- Critical fault rate (faults causing >10% accuracy degradation)

**Important note on bit-position vulnerability:** Across all 36 configurations, bit 1 (exponent MSB) is the dominant source of SEU vulnerability. Bits 0, 8, 9, and 31 cause near-zero accuracy drops. The mean accuracy drop and critical fault rate reported below are averages across all 5 tested bit positions; the bit-1-specific impact is roughly 5x larger.

______________________________________________________________________

## 4.2 Results by Dataset

### 4.2.1 Moons Dataset

**With Dropout (0.2):**

| Flood Level | Baseline Acc | Train Loss | Val Loss | Acc Drop | CFR   |
| ----------- | ------------ | ---------- | -------- | -------- | ----- |
| 0.00 (std)  | 91.25%       | 0.2119     | 0.1985   | 1.81%    | 6.21% |
| 0.05        | 90.75%       | 0.2125     | 0.2002   | 1.86%    | 6.06% |
| 0.10        | 91.75%       | 0.2041     | 0.1973   | 1.99%    | 6.23% |
| 0.15        | 91.00%       | 0.2220     | 0.2017   | 1.71%    | 5.38% |
| 0.20        | 92.00%       | 0.2169     | 0.1991   | 2.06%    | 6.28% |
| 0.30        | 88.50%       | 0.3016     | 0.2679   | 1.52%    | 5.68% |

**Without Dropout:**

| Flood Level | Baseline Acc | Train Loss | Val Loss | Acc Drop | CFR   |
| ----------- | ------------ | ---------- | -------- | -------- | ----- |
| 0.00 (std)  | 92.25%       | 0.1977     | 0.1984   | 2.30%    | 6.58% |
| 0.05        | 91.50%       | 0.1976     | 0.1995   | 2.39%    | 7.46% |
| 0.10        | 91.50%       | 0.1969     | 0.2032   | 2.31%    | 6.41% |
| 0.15        | 91.50%       | 0.1985     | 0.1995   | 2.70%    | 7.56% |
| 0.20        | 92.00%       | 0.2002     | 0.2001   | 1.95%    | 5.88% |
| 0.30        | 88.75%       | 0.3019     | 0.2687   | 1.83%    | 6.00% |

**Observations:**

- At b=0.30, flooding reduces SEU vulnerability noticeably (1.52% with dropout vs 1.81% baseline)
- At moderate flood levels (b=0.10-0.20), results are mixed: some configurations show slightly *worse* robustness than baseline
- The b=0.15 with-dropout configuration achieves the best robustness for moons (1.71%)
- Without dropout, results are noisier and no flood level consistently outperforms baseline

### 4.2.2 Circles Dataset

**With Dropout (0.2):**

| Flood Level | Baseline Acc | Train Loss | Val Loss | Acc Drop | CFR   |
| ----------- | ------------ | ---------- | -------- | -------- | ----- |
| 0.00 (std)  | 79.50%       | 0.4473     | 0.4874   | 1.39%    | 5.39% |
| 0.05        | 78.25%       | 0.4484     | 0.4864   | 1.41%    | 5.94% |
| 0.10        | 80.00%       | 0.4348     | 0.4885   | 1.43%    | 5.66% |
| 0.15        | 79.75%       | 0.4340     | 0.4912   | 1.21%    | 4.91% |
| 0.20        | 79.50%       | 0.4460     | 0.4906   | 1.39%    | 5.89% |
| 0.30        | 79.75%       | 0.4408     | 0.4853   | 1.72%    | 6.34% |

**Without Dropout:**

| Flood Level | Baseline Acc | Train Loss | Val Loss | Acc Drop | CFR   |
| ----------- | ------------ | ---------- | -------- | -------- | ----- |
| 0.00 (std)  | 79.75%       | 0.4240     | 0.4873   | 1.42%    | 7.25% |
| 0.05        | 80.25%       | 0.4211     | 0.4860   | 1.54%    | 7.23% |
| 0.10        | 79.50%       | 0.4210     | 0.4864   | 1.86%    | 8.89% |
| 0.15        | 79.75%       | 0.4216     | 0.4860   | 1.87%    | 9.13% |
| 0.20        | 80.00%       | 0.4223     | 0.4814   | 1.98%    | 8.93% |
| 0.30        | 80.00%       | 0.4230     | 0.4877   | 1.86%    | 8.18% |

**Observations:**

- Circles is the most challenging dataset (baseline accuracy ~79-80%)
- With dropout, b=0.15 shows the best robustness (1.21% drop, 4.91% CFR)
- Without dropout, flooding consistently *worsens* SEU vulnerability compared to baseline
- This suggests flooding's benefits are dataset- and configuration-dependent, not universal

### 4.2.3 Blobs Dataset

**With Dropout (0.2):**

| Flood Level | Baseline Acc | Train Loss | Val Loss | Acc Drop | CFR   |
| ----------- | ------------ | ---------- | -------- | -------- | ----- |
| 0.00 (std)  | 100.00%      | 0.0000     | 0.0090   | 2.59%    | 5.86% |
| 0.05        | 99.50%       | 0.0548     | 0.0392   | 2.12%    | 4.59% |
| 0.10        | 99.75%       | 0.1093     | 0.0844   | 1.40%    | 3.33% |
| 0.15        | 99.75%       | 0.1529     | 0.1494   | 1.33%    | 3.30% |
| 0.20        | 100.00%      | 0.2020     | 0.1957   | 1.58%    | 4.10% |
| 0.30        | 99.50%       | 0.3141     | 0.2873   | 2.03%    | 5.19% |

**Without Dropout:**

| Flood Level | Baseline Acc | Train Loss | Val Loss | Acc Drop | CFR   |
| ----------- | ------------ | ---------- | -------- | -------- | ----- |
| 0.00 (std)  | 100.00%      | 0.0000     | 0.0111   | 2.14%    | 4.65% |
| 0.05        | 97.75%       | 0.0509     | 0.0572   | 2.24%    | 5.86% |
| 0.10        | 99.75%       | 0.1150     | 0.1147   | 2.27%    | 5.82% |
| 0.15        | 98.00%       | 0.1535     | 0.1460   | 1.68%    | 4.08% |
| 0.20        | 100.00%      | 0.2113     | 0.1957   | 1.63%    | 4.03% |
| 0.30        | 100.00%      | 0.3014     | 0.3045   | 2.00%    | 4.76% |

**Observations:**

- Blobs is the easiest dataset (baseline accuracy 97.75-100%)
- Blobs shows the **strongest and clearest** flooding benefit, especially with dropout
- With dropout, b=0.10 and b=0.15 reduce accuracy drop by ~46-49% compared to baseline (2.59% → 1.40-1.33%)
- Without dropout, the benefit is smaller and concentrated at b=0.15-0.20
- Blobs' high separability may allow the model to benefit more from regularization

______________________________________________________________________

## 4.3 Cross-Dataset Analysis

### 4.3.1 Effect of Flood Level

Averaging across all datasets and dropout configurations:

| Flood Level | Mean Baseline Acc | Mean Acc Drop | Relative Improvement |
| ----------- | ----------------- | ------------- | -------------------- |
| 0.00 (std)  | 90.46%            | 1.94%         | 0% (baseline)        |
| 0.05        | 89.67%            | 1.93%         | 0.9% better          |
| 0.10        | 90.38%            | 1.87%         | 3.6% better          |
| 0.15        | 89.96%            | 1.75%         | 10.0% better         |
| 0.20        | 90.58%            | 1.77%         | 9.2% better          |
| 0.30        | 89.42%            | 1.83%         | 6.0% better          |

**Key Findings:**

- Flooding provides measurable robustness improvement, peaking at **b=0.15 (10.0% reduction in mean accuracy drop)**
- The relationship is **non-monotonic**: improvement peaks at b=0.15 then decreases at higher flood levels
- Even the best cross-dataset improvement (10.0%) is modest in absolute terms (1.94% → 1.75% mean accuracy drop)
- The improvement is **not consistent** across all datasets—blobs drives most of the average improvement

### 4.3.2 Cost-Benefit Analysis

| Flood Level | Acc Cost | Robustness Gain |
| ----------- | -------- | --------------- |
| 0.05        | 0.79%    | 0.9%            |
| 0.10        | 0.08%    | 3.6%            |
| **0.15**    | **0.50%**| **10.0%**       |
| 0.20        | -0.12%*  | 9.2%            |
| 0.30        | 1.04%    | 6.0%            |

*b=0.20 shows negative accuracy cost (i.e., slightly higher baseline accuracy on average). This is likely due to random variation rather than a genuine benefit.

**Optimal Configuration**: b=0.15 provides the best balance of robustness gain (10.0%) with modest accuracy cost (0.50%).

### 4.3.3 Dropout Interaction

Comparing with vs without dropout (averaged across datasets and flood levels):

| Configuration | Mean Baseline Acc | Mean Acc Drop | Mean CFR |
| ------------- | ----------------- | ------------- | -------- |
| With Dropout  | 90.03%            | 1.70%         | 5.35%    |
| No Dropout    | 90.13%            | 2.00%         | 6.59%    |

**Observations:**

- Dropout reduces SEU vulnerability by **15.1%** (2.00% → 1.70% mean accuracy drop) with negligible accuracy cost (0.10%)
- Dropout also substantially reduces critical fault rate (6.59% → 5.35%)
- Flooding provides benefits primarily **in conjunction with dropout**; without dropout, flooding often fails to improve robustness
- **Dropout + Flooding** is the most robust combination, but the benefit comes largely from dropout itself

______________________________________________________________________

## 4.4 Statistical Significance

### 4.4.1 Standard Training vs Flood Training at b=0.15

Comparing standard training (b=0.0) to b=0.15 (best cross-dataset flood level):

**Moons with Dropout:**

- Standard: 1.81% accuracy drop
- Flood (0.15): 1.71% accuracy drop
- **Improvement**: 5.5%

**Circles with Dropout:**

- Standard: 1.39% accuracy drop
- Flood (0.15): 1.21% accuracy drop
- **Improvement**: 13.0%

**Blobs with Dropout:**

- Standard: 2.59% accuracy drop
- Flood (0.15): 1.33% accuracy drop
- **Improvement**: 48.6%

**Blobs without Dropout:**

- Standard: 2.14% accuracy drop
- Flood (0.15): 1.68% accuracy drop
- **Improvement**: 21.5%

*Note: These comparisons use mean accuracy drop across all bit positions. Formal hypothesis testing would require paired t-tests on individual injection results. The effect is strong for blobs, moderate for circles with dropout, and modest for moons.*

______________________________________________________________________

## 4.5 Per-Bit-Position Analysis

### 4.5.1 Vulnerability by Bit Position

Across all configurations, bit 1 (exponent MSB) accounts for essentially all observed SEU vulnerability:

| Bit Position | Role          | Typical Acc Drop | Typical CFR | Impact    |
| ------------ | ------------- | ---------------- | ----------- | --------- |
| 0            | Sign          | <0.1%            | 0%          | Negligible|
| 1            | Exponent MSB  | 7-13%            | 20-45%      | **Critical** |
| 8            | Exponent LSB  | <0.1%            | 0%          | Negligible|
| 9            | Mantissa MSB  | <0.1%            | 0%          | Negligible|
| 31           | Mantissa LSB  | ~0%              | 0%          | None      |

**Key Insight**: The exponent MSB (bit 1) is the single dominant vulnerability. Flipping this bit causes roughly 2× magnitude changes in parameter values, which propagates catastrophically through the network. All other tested bits cause minimal disruption. This has important implications for targeted hardware protection.

### 4.5.2 Flood Level Effect on Bit-1 Vulnerability

The benefit of flooding is primarily visible in bit-1 vulnerability:

- **Blobs (dropout, b=0.0)**: Bit-1 accuracy drop = 12.97%, CFR = 29.3%
- **Blobs (dropout, b=0.10)**: Bit-1 accuracy drop = 7.00%, CFR = 16.7%
- **Blobs (dropout, b=0.15)**: Bit-1 accuracy drop = 6.67%, CFR = 16.5%

This ~46% reduction in bit-1 vulnerability for blobs represents the strongest observed flooding effect.

______________________________________________________________________

## 4.6 Training Dynamics

### 4.6.1 Does Flooding Actually Constrain Training?

The relationship between flood level and training loss is dataset-dependent:

**Blobs** (where flooding is most effective):

| Flood Level | Mean Final Train Loss | Flood Active? |
| ----------- | --------------------- | ------------- |
| 0.05        | 0.053                 | Yes           |
| 0.10        | 0.112                 | Yes (matched) |
| 0.15        | 0.153                 | Yes (matched) |
| 0.20        | 0.207                 | Yes (matched) |
| 0.30        | 0.308                 | Yes (matched) |

**Circles** (where flooding is least effective):

| Flood Level | Mean Final Train Loss | Flood Active?                  |
| ----------- | --------------------- | ------------------------------ |
| 0.05        | 0.435                 | No (loss >> flood level)       |
| 0.10        | 0.428                 | No (loss >> flood level)       |
| 0.15        | 0.428                 | No (loss >> flood level)       |
| 0.20        | 0.434                 | No (loss >> flood level)       |
| 0.30        | 0.432                 | No (loss >> flood level)       |

**Critical Observation**: For the circles dataset, the base training loss (~0.43) is well above all tested flood levels. The flooding constraint is **never active** for circles, which explains why flooding has minimal effect on this dataset. This underscores that flood levels must be calibrated above the natural training loss convergence point.

### 4.6.2 Validation Loss Trends

| Flood Level | Mean Val Loss | Change from Standard |
| ----------- | ------------- | -------------------- |
| 0.00        | 0.232         | baseline             |
| 0.10        | 0.262         | +13.1%               |
| 0.20        | 0.294         | +26.7%               |
| 0.30        | 0.350         | +51.0%               |

Validation loss increases with flood level, reflecting the regularization cost. The increase is driven primarily by blobs (where flooding is active and constraining) and moons at higher flood levels.

______________________________________________________________________

## 4.7 Key Findings Summary

1. **Flooding can improve SEU robustness**: Up to 10.0% average improvement at b=0.15, and up to ~49% for individual dataset-configurations (blobs with dropout)
1. **Effect is dataset-dependent**: Blobs benefits strongly, circles shows minimal benefit (flooding inactive), moons shows modest benefit
1. **Flooding requires calibration**: The flood level must exceed the natural training loss convergence point to be active; otherwise it has no effect
1. **Optimal cross-dataset configuration**: b=0.15 with dropout (10.0% improvement, 0.50% accuracy cost)
1. **Dropout is independently beneficial**: 15.1% robustness improvement from dropout alone, with negligible accuracy cost
1. **Bit-1 dominance**: The exponent MSB (bit 1) accounts for essentially all observed vulnerability; targeted protection of this bit position would be highly effective
1. **Non-monotonic relationship**: Higher flood levels do not always yield better robustness; b=0.15 is the cross-dataset optimum

______________________________________________________________________

## 4.8 Data Availability

All experimental results are publicly available:

- **CSV format**: `comprehensive_results.csv` (36 configurations)
- **JSON format**: `comprehensive_results.json` (with per-bit-position details)
- **Reproducible code**: `comprehensive_experiment.py`

[← Previous: Methodology](03_methodology.md) | [Back to README](README.md) | [Next: Discussion →](05_discussion.md)
