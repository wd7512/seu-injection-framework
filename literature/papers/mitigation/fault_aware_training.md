# Mitigating Multiple SEUs Using Fault-Aware Training

**Vinck, T., Jonckers, N., Dekkers, G., Prinzie, J., Karsmakers, P.**  
*arXiv:2502.09374, 2025*

## Summary

Fault-Aware Training (FAT) methodology improves DNN robustness to multiple SEUs. Shows 3x improvement in fault tolerance without hardware modifications.

## Approach

1. Fault injection during backpropagation
2. Loss modification incorporating fault impact
3. Robustness evaluation on various fault rates

## Key Results

- Up to **3x improvement** in fault tolerance
- Works for multiple simultaneous bit flips
- No inference overhead - training-only cost

## Comparison

| Method | Improvement |
|--------|-------------|
| Standard Training | Baseline |
| Clipped ReLU | ~1.5x |
| **Fault-Aware Training** | **~3x** |

## Verification Links

- [Paper](https://arxiv.org/abs/2502.09374) - arXiv
- [PDF](https://arxiv.org/pdf/2502.09374.pdf) - arXiv PDF

## Relevance

- Provides training-based mitigation strategy
- Can be combined with framework's injection capabilities
- Validates importance of sign bit protection

## BibTeX

```bibtex
@article{vinck2025mitigating,
  title={Mitigating multiple single-event upsets during deep neural network inference using fault-aware training},
  author={Vinck, Toon and Jonckers, Na{\"i}n and Dekkers, Gert and Prinzie, Jeffrey and Karsmakers, Peter},
  journal={arXiv preprint arXiv:2502.09374},
  year={2025}
}
```

Tags: `mitigation` `fault_aware_training` `robustness`
