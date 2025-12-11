SEU Injection Framework Documentation
======================================

A Python framework for **Single Event Upset (SEU) injection** in neural networks for robustness analysis in harsh environments.

The SEU Injection Framework provides tools for:

- Systematic SEU injection in PyTorch models
- Performance evaluation under radiation-induced bit flips
- Analysis of neural network robustness in space and nuclear environments

.. image:: https://img.shields.io/pypi/v/seu-injection-framework.svg
   :target: https://pypi.org/project/seu-injection-framework/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue
   :target: https://github.com/wd7512/seu-injection-framework
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Quick Links
-----------

* `Research Paper <https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars>`_
* `Interactive Explainer <https://wd7512.github.io/robust-ml-in-harsh-environments-explained/>`_
* `GitHub Repository <https://github.com/wd7512/seu-injection-framework>`_
* `PyPI Package <https://pypi.org/project/seu-injection-framework/>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/basic_usage
   user_guide/advanced_features
   user_guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/bitops
   api/metrics

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   known_issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

