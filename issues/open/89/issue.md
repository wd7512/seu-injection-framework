# Issue #89: [SWARM REVIEW] Group G: Research Methodology & Reproducibility

## Findings (2 issues)

### G1 🟠 No documented threat model
**File:** All injectors

The framework does not specify which radiation environment, bit error rate, or upset model it targets. Key questions users must answer for themselves:
- Is this modeling space radiation (galactic cosmic rays, solar protons)? Nuclear environment? Avionics at altitude?
- What is the assumed bit error rate? Experiments use `p=0.15` (15% of parameters flipped) — orders of magnitude above realistic space SEU rates (~10⁻⁷ upsets/bit/day).
- Does flipping stored float32 weights accurately model real SEUs? Real radiation upsets affect memory cells (SRAM cells, DRAM, registers, combinational logic) — the mapping from a struck memory cell to a weight-bit-flip is non-trivial.
- What about activation upsets, instruction corruption, multi-bit upsets, timing effects, and total-dose degradation?

**Fix:** Add a THREAT_MODEL.md document that specifies:
- Target radiation environment(s) and flux/energy spectra
- Assumed physical fault model and its limitations
- What is in scope (weight bitflips) and what is out of scope (activation faults, MCU, TID)
- Guidance on selecting `p` values that correspond to realistic mission durations

### G2 🟠 No seed parameter on stochastic injector — reproducibility gap
**File:** `src/seu_injection/core/stochastic_seu_injector.py:89`
```python
injection_mask = np.random.random(tensor_cpu.shape) < p
```
Uses the global `np.random` state with no seed management. Experiment scripts set `np.random.seed()` once at the top of a 36-configuration sweep, meaning all configurations share RNG state. Isolating a single configuration produces a different result than running it within the sweep.

**Fix:** Add a `seed: Optional[int] = None` parameter to `StochasticSEUInjector.__init__`. Internally use `np.random.RandomState(seed)` (or the new NumPy Generator API) so each injector instance has independent RNG state.


---

| Field | Value |
|-------|-------|
| **State** | open |
| **Created** | 2026-06-02T23:26:49Z |
| **Updated** | 2026-06-02T23:26:49Z |
| **Labels** | question |
| **Author** | @wd7512 |
| **URL** | https://github.com/wd7512/seu-injection-framework/issues/89 |
