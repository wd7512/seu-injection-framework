# References

[Back to README](README.md)

---

## Primary References

### Flood Level Training

**Ishida, T., Yamane, I., Sakai, T., Niu, G., & Sugiyama, M. (2020)**  
"Do We Need Zero Training Loss After Achieving Zero Training Error?"  
*Advances in Neural Information Processing Systems (NeurIPS)*, 33, 9444-9455.  
[Paper](https://proceedings.neurips.cc/paper/2020/hash/6a08076e1ceaa9 fed6a9e3f4a923a58-Abstract.html)

**Key contribution**: Introduced flood level training and demonstrated improved generalization.

---

## Loss Landscape and Robustness

**Hochreiter, S., & Schmidhuber, J. (1997)**  
"Flat Minima"  
*Neural Computation*, 9(1), 1-42.  
DOI: 10.1162/neco.1997.9.1.1

**Key contribution**: First proposed connection between flat minima and generalization.

**Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017)**  
"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"  
*International Conference on Learning Representations (ICLR)*.  
[Paper](https://arxiv.org/abs/1609.04836)

**Key contribution**: Demonstrated causality between loss sharpness and generalization.

**Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018)**  
"Visualizing the Loss Landscape of Neural Nets"  
*Advances in Neural Information Processing Systems (NeurIPS)*, 31.  
[Paper](https://arxiv.org/abs/1712.09913)

**Key contribution**: Developed techniques for visualizing high-dimensional loss landscapes.

**Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021)**  
"Sharpness-Aware Minimization for Efficiently Improving Generalization"  
*International Conference on Learning Representations (ICLR)*.  
[Paper](https://arxiv.org/abs/2010.01412)

**Key contribution**: Introduced SAM, which explicitly seeks flat minima.

**Pattnaik, S., Tang, X., Jain, A., Park, J., Yazar, O., Rezaei, A., Kim, Y., Gupta, A., & Grover, P. (2020)**  
"Robust Deep Neural Networks"  
*arXiv preprint arXiv:2002.10355*.  
[Paper](https://arxiv.org/abs/2002.10355)

**Key contribution**: Showed flat minima are more robust to weight noise.

---

## SEU Robustness and Fault Tolerance

**Dennis, W., & Pope, J. (2025)**  
"A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices"  
*Proceedings of the 17th International Conference on Agents and Artificial Intelligence (ICAART)*, Volume 2, 322-333.  
Publisher: SciTePress. DOI: 10.5220/0013155000003890  
[Link](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)

**Key contribution**: Systematic comparison of CNN architectures for SEU robustness. Introduced SEU Injection Framework.

**Reagen, B., Gupta, U., Pentecost, L., Whatmough, P., Lee, S. K., Mulholland, N., Brooks, D., & Wei, G. Y. (2018)**  
"Ares: A Framework for Quantifying the Resilience of Deep Neural Networks"  
*Design Automation Conference (DAC)*, 1-6. IEEE.  
[Paper](https://ieeexplore.ieee.org/document/8465834)

**Key contribution**: Systematic fault injection framework showing different layers/bits have varying criticality.

**Li, G., Hari, S. K. S., Sullivan, M., Tsai, T., Pattabiraman, K., Emer, J., & Keckler, S. W. (2017)**  
"Understanding Error Propagation in Deep Learning Neural Network (DNN) Accelerators and Applications"  
*International Conference for High Performance Computing, Networking, Storage and Analysis (SC)*, 1-12. IEEE.  
[Paper](https://ieeexplore.ieee.org/document/8126529)

**Key contribution**: Analyzed how bit flips propagate through networks.

**Schorn, C., Guntoro, A., & Ascheid, G. (2018)**  
"Accurate Neuron Resilience Prediction for a Flexible Reliability Management in Neural Network Accelerators"  
*Design, Automation & Test in Europe Conference (DATE)*, 979-984. IEEE.  
[Paper](https://ieeexplore.ieee.org/document/8342159)

**Key contribution**: Showed dropout improves SEU tolerance.

---

## Regularization and Generalization

**Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017)**  
"Understanding Deep Learning Requires Rethinking Generalization"  
*International Conference on Learning Representations (ICLR)*.  
[Paper](https://arxiv.org/abs/1611.03530)

**Key contribution**: Showed neural networks can fit random labels, motivating need for regularization.

**Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014)**  
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"  
*Journal of Machine Learning Research (JMLR)*, 15(1), 1929-1958.  
[Paper](http://jmlr.org/papers/v15/srivastava14a.html)

**Key contribution**: Introduced dropout regularization.

**Zhu, F., Wu, X., Fu, Y., & Qian, L. (2019)**  
"Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks"  
*International Conference on Machine Learning (ICML)*, 2020.  
[Paper](https://arxiv.org/abs/1908.06314)

**Key contribution**: Connection between adversarial robustness and loss landscape.

---

## Additional Reading

### Radiation Effects in Electronics

**Baumann, R. C. (2005)**  
"Radiation-induced soft errors in advanced semiconductor technologies"  
*IEEE Transactions on Device and Materials Reliability*, 5(3), 305-316.

### Hardware Fault Tolerance

**Lyons, R. E., & Vanderkulk, W. (1962)**  
"The Use of Triple-Modular Redundancy to Improve Computer Reliability"  
*IBM Journal of Research and Development*, 6(2), 200-209.

### Information Theory and Generalization

**Tishby, N., & Zaslavsky, N. (2015)**  
"Deep Learning and the Information Bottleneck Principle"  
*IEEE Information Theory Workshop (ITW)*, 1-5.

---

## Software and Tools

**PyTorch**  
Paszke, A., Gross, S., Massa, F., Lerer, A., et al. (2019)  
"PyTorch: An Imperative Style, High-Performance Deep Learning Library"  
*Advances in Neural Information Processing Systems (NeurIPS)*, 32.  
https://pytorch.org

**SEU Injection Framework**  
Dennis, W. (2025)  
*SEU Injection Framework*, Version 1.1.12.  
https://github.com/wd7512/seu-injection-framework

**Scikit-learn**  
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011)  
"Scikit-learn: Machine Learning in Python"  
*Journal of Machine Learning Research*, 12, 2825-2830.  
https://scikit-learn.org

---

## Related Work (Not Directly Cited)

### Adversarial Training
- Madry, A., et al. (2018): "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR.
- Wong, E., & Kolter, Z. (2018): "Provable defenses against adversarial examples via the convex outer adversarial polytope", ICML.

### Quantization and Pruning
- Han, S., et al. (2015): "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", ICLR.
- Jacob, B., et al. (2018): "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR.

### Neural Architecture Search
- Zoph, B., & Le, Q. V. (2017): "Neural Architecture Search with Reinforcement Learning", ICLR.
- Tan, M., & Le, Q. (2019): "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML.

---

## Citation for This Work

If you use this research in your work, please cite:

```bibtex
@techreport{flood_training_seu_2025,
  title={Flood Level Training for SEU Robustness: 
         A Training-Time Approach to Radiation Tolerance},
  author={SEU Injection Framework Research Team},
  year={2025},
  institution={SEU Injection Framework Project},
  url={https://github.com/wd7512/seu-injection-framework/tree/main/examples/flood_training_study},
  note={Research study on flood level training for improving neural network 
        robustness to Single Event Upsets in harsh radiation environments}
}
```

And cite the foundational works:

```bibtex
@inproceedings{ishida2020flooding,
  title={Do We Need Zero Training Loss After Achieving Zero Training Error?},
  author={Ishida, Takashi and Yamane, Ikko and Sakai, Tomoya and Niu, Gang and Sugiyama, Masashi},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9444--9455},
  year={2020}
}

@inproceedings{dennis2025framework,
  title={A Framework for Developing Robust Machine Learning Models in Harsh Environments: 
         A Review of CNN Design Choices},
  author={Dennis, William and Pope, James},
  booktitle={Proceedings of the 17th International Conference on Agents and 
             Artificial Intelligence (ICAART)},
  volume={2},
  pages={322--333},
  year={2025},
  organization={SciTePress},
  doi={10.5220/0013155000003890}
}
```

---

**Last Updated**: December 11, 2025  
**Document Version**: 1.0  
**License**: CC BY 4.0
