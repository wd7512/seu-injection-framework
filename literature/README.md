# Literature Collection

Living collection of background reading for SEU injection research in neural networks.

## Purpose

This collection documents relevant academic papers, methodologies, and findings that inform the SEU Injection Framework development and research direction.

## Structure

```
literature/
├── README.md                    # This file - overview & index
├── bibliography.yaml            # BibTeX for citation management
├── papers/
│   ├── fault_injection/        # SEU/fault injection methodology
│   ├── bit_flip_attacks/       # BFA and adversarial bit flips  
│   ├── mitigation/             # Hardening & fault-tolerant training
│   └── surveys/               # Overview papers
```

## Adding Papers

1. Add paper to appropriate topic folder as `.md`
2. Update `bibliography.yaml` with BibTeX entry
3. Add entry to topic index below

## Topic Index

### Fault Injection Methodology
- [Selective Hardening for Neural Networks in FPGAs](./papers/fault_injection/selective_hardening_fpgas.md) - Libano et al. (2019)
- [FsimNNs Platform](./papers/fault_injection/fsimnns_platform.md) - Lu et al. (2025)

### Bit-Flip Attacks & Defenses
- [Survey of Bit-Flip Attacks on DNN](./papers/bit_flip_attacks/bit_flip_attack_survey.md) - Qian et al. (2023)

### Mitigation Techniques
- [Fault-Aware Training for SEU Mitigation](./papers/mitigation/fault_aware_training.md) - Vinck et al. (2025)

### Surveys & Overview
- [Fault-Tolerant Methodologies for DNNs](./papers/surveys/fault_tolerant_dnn_survey.md)

## Key Papers Summary

| Year | Paper | Key Contribution |
|------|-------|------------------|
| 2019 | Selective Hardening for NNs in FPGAs | Radiation effects on FPGA-based NNs |
| 2023 | Survey of Bit-Flip Attacks | Comprehensive defense survey |
| 2024 | Fault-Aware Training | FAT methodology (3x robustness) |
| 2025 | FsimNNs | GNN-based SEU simulation platform |

## Related

- [Research Paper (ICAART 2025)](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)
- [SEU Injection Framework Documentation](../docs/)
