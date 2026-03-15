# Literature Collection

Living collection of background reading for SEU injection research in neural networks.

## Purpose

This collection documents relevant academic papers, methodologies, and findings that inform the SEU Injection Framework development and research direction.

## Structure

```
literature/
├── README.md                    # This file - overview & index
├── bibliography.yaml            # BibTeX for citation management
├── VERIFICATION.md              # Verification protocol
└── papers/
    ├── _template.md             # Template for new papers
    ├── fault_injection/         # SEU/fault injection methodology
    ├── bit_flip_attacks/        # BFA and adversarial bit flips
    ├── mitigation/              # Hardening & fault-tolerant training
    └── surveys/                 # Overview papers
```

## Verification

All papers include verification links. See [VERIFICATION.md](./VERIFICATION.md).

| Badge | Meaning |
|-------|---------|
| `arXiv` | Preprint - verify at arXiv.org |
| `DOI` | Peer-reviewed - verify at doi.org |

## Adding Papers

1. Add paper to appropriate topic folder as `.md`
2. Use `_template.md` as base
3. Include verification links
4. Update `bibliography.yaml` with entry
5. Add entry to topic index below

## Topic Index

### Fault Injection Methodology
- [Selective Hardening for Neural Networks in FPGAs](./papers/fault_injection/selective_hardening_fpgas.md) - Libano et al. (2019) `(DOI)`
- [FsimNNs Platform](./papers/fault_injection/fsimnns_platform.md) - Lu et al. (2025) `(arXiv)`

### Bit-Flip Attacks & Defenses
- [Survey of Bit-Flip Attacks on DNN](./papers/bit_flip_attacks/bit_flip_attack_survey.md) - Qian et al. (2023) `(DOI)`

### Mitigation Techniques
- [Fault-Aware Training for SEU Mitigation](./papers/mitigation/fault_aware_training.md) - Vinck et al. (2025) `(arXiv)`

### Surveys & Overview
- [Fault-Tolerant Methodologies for DNNs](./papers/surveys/fault_tolerant_dnn_survey.md) `(DOI)`

## Key Papers Summary

| Year | Paper | Key Contribution | Verification |
|------|-------|------------------|--------------|
| 2019 | Selective Hardening for NNs in FPGAs | Radiation effects on FPGA-based NNs | DOI |
| 2023 | Survey of Bit-Flip Attacks | Comprehensive defense survey | DOI |
| 2025 | Fault-Aware Training | FAT methodology (3x robustness) | arXiv |
| 2025 | FsimNNs | GNN-based SEU simulation platform | arXiv |

## Related

- [Research Paper (ICAART 2025)](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)
- [SEU Injection Framework Documentation](../docs/)
