# Selective Hardening for Neural Networks in FPGAs

**Libano, F., Wilson, B., Anderson, J., Wirthlin, M.J., Cazzaniga, C., Frost, C., Rech, P.**  
*IEEE Transactions on Nuclear Science, 2019*

## Summary

Evaluates radiation-induced errors on neural networks in FPGAs. Focuses on Iris Flower ANN and MNIST CNN.

## Key Contributions

1. Fault injection methodology for FPGA-based NNs
2. Selective hardening via triplication of critical neurons
3. Vulnerability analysis of different network layers

## Findings

- CNNs show different vulnerability patterns than ANNs
- Certain layers are more critical for reliability
- Selective hardening achieves fault tolerance with reduced overhead

## Relevance

- Validates layer-wise fault injection importance
- Provides benchmark data for CNN vulnerability
- Supports systematic injection approach

## BibTeX

```bibtex
@article{libano2019selective,
  title={Selective Hardening for Neural Networks in FPGAs},
  author={Libano, F. and Wilson, B. and Anderson, J. and Wirthlin, M.J. and Cazzaniga, C. and Frost, C. and Rech, P.},
  journal={IEEE Transactions on Nuclear Science},
  year={2019}
}
```

Tags: `fault_injection` `fpga` `radiation` `cnn`
