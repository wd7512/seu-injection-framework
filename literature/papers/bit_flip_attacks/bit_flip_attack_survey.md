# Survey of Bit-Flip Attacks on Deep Neural Networks

**Qian, C., Zhang, M., Nie, Y., Lu, S., Cao, H.**  
*MDPI Electronics, 12(4), 853, 2023*

## Summary

Comprehensive survey of bit-flip attacks (BFA) and defense mechanisms.

## Attack Categories

1. **Rowhammer-based BFA**: Exploits DRAM physical vulnerabilities
2. **Progressive Bit Search**: Iterative identification of critical bits
3. **Adversarial Bit Flips**: Targets model integrity

## Defense Methods

| Category | Approach |
|----------|----------|
| Detection | Anomaly detection in model parameters |
| Prevention | Secure hardware design |
| Mitigation | Fault-tolerant training |
| Redundancy | Model duplication, ensemble |

## Key Findings

- Few-bit flips can severely degrade DNN performance
- Sign bit flips often have most impact

## Verification Links

- [Paper](https://doi.org/10.3390/electronics12040853) - DOI
- [MDPI](https://www.mdpi.com/2079-9292/12/4/853) - MDPI

## BibTeX

```bibtex
@article{qian2023survey,
  title={A Survey of Bit-Flip Attacks on Deep Neural Network and Corresponding Defense Methods},
  author={Qian, Cheng and Zhang, Ming and Nie, Yuanping and Lu, Shuaibing and Cao, Huayang},
  journal={Electronics},
  volume={12},
  number={4},
  year={2023},
  doi={10.3390/electronics12040853}
}
```

Tags: `bit_flip_attack` `survey` `defense`
