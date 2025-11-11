# Research Examples# Research Examples



Ready-to-run examples demonstrating fault injection methodologies for neural network robustness analysis.



## Quick StartThis directory contains validated research examples demonstrating systematic fault injection methodologies for neural network robustness analysis in harsh environments.



```bash

# From PyPI installation

pip install seu-injection-framework## Getting StartedThis directory contains validated research examples demonstrating systematic fault injection methodologies for neural network robustness analysis in harsh environments.

python basic_cnn_robustness.py



# From source

git clone https://github.com/wd7512/seu-injection-framework.git```bash

cd seu-injection-framework

uv sync --all-extrasgit clone https://github.com/wd7512/seu-injection-framework.git

uv run python examples/basic_cnn_robustness.py

```cd seu-injection-framework## Getting StartedThis directory contains validated research examples demonstrating systematic fault injection methodologies for neural network robustness analysis in harsh environments.Ready-to-run examples showing how to use the SEU injection framework.



## Available Examplesuv sync --all-extras  # or pip install -e ".[dev,notebooks,extras]"



### [Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb)uv run python examples/basic_cnn_robustness.py

Interactive research notebook with comprehensive fault injection analysis and visualization tools.

```

### [basic_cnn_robustness.py](basic_cnn_robustness.py)

**Objective**: Single-architecture vulnerability analysis  ```bash

**Method**: Systematic bit-flip injection across network layers  

**Application**: Space mission deployment assessment  ## Available Studies



### [architecture_comparison.py](architecture_comparison.py)uv add seu-injection-framework

**Objective**: Comparative robustness evaluation  

**Method**: Standardized fault injection protocol across multiple architectures  ### [Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb)

**Application**: Architecture selection for critical systems  

Interactive research notebook with comprehensive fault injection analysis and visualization tools.uv run python basic_cnn_robustness.py## Getting Started## Quick Start

## Experimental Protocol



Both studies implement IEEE 754-compliant fault injection methodology:

- **Fault Model**: Single Event Upset (SEU) via targeted bit manipulation### basic_cnn_robustness.py```

- **Coverage**: Exhaustive layer-wise vulnerability mapping

- **Statistics**: Multiple-trial averaging with confidence intervals**Objective**: Single-architecture vulnerability analysis  

- **Validation**: Baseline accuracy verification and repeatability testing

**Method**: Systematic bit-flip injection across network layers  

## Customization

**Application**: Space mission deployment assessment  

Easy to adapt for your own models and data:

## Available Studies

```python

from seu_injection import SEUInjector### architecture_comparison.py  

from seu_injection.metrics import classification_accuracy

**Objective**: Comparative robustness evaluation  ```bash```bash

# Use your own model and data

injector = SEUInjector(**Method**: Standardized fault injection protocol across multiple architectures  

    trained_model=your_model,

    criterion=classification_accuracy,**Application**: Architecture selection for critical systems  ### [Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb)

    x=your_test_data,

    y=your_labels

)

## Experimental ProtocolInteractive research notebook with comprehensive fault injection analysis and visualization tools.pip install seu-injection-frameworkpip install seu-injection-framework

# Run analysis

results = injector.run_seu(bit_i=0)

```

Both studies implement the IEEE 754-compliant fault injection methodology described in our research paper. Key experimental parameters:

## Results



Each script generates detailed reports showing:

- Which layers are most vulnerable to bit flips- **Fault Model**: Single Event Upset (SEU) via targeted bit manipulation### basic_cnn_robustness.pypython basic_cnn_robustness.pypython basic_cnn_robustness.py

- How different bit positions affect accuracy

- Robustness comparison across architectures- **Coverage**: Exhaustive layer-wise vulnerability mapping  



## Citation- **Statistics**: Multiple-trial averaging with confidence intervals**Objective**: Single-architecture vulnerability analysis  



```bibtex- **Validation**: Baseline accuracy verification and repeatability testing

@software{seu_injection_framework,

  author = {William Dennis},**Method**: Systematic bit-flip injection across network layers  ``````

  title = {SEU Injection Framework},

  year = {2025},## Research Applications

  url = {https://github.com/wd7512/seu-injection-framework},

  version = {1.1.8}**Application**: Space mission deployment assessment  

}

```### Adapting for New Studies



For methodology details, see the main [repository documentation](../docs/).


```python

from seu_injection import SEUInjector### architecture_comparison.py  

from seu_injection.metrics import classification_accuracy

**Objective**: Comparative robustness evaluation  ## Available Studies## What's Here

# Configure for your research

injector = SEUInjector(your_model)**Method**: Standardized fault injection protocol across multiple architectures  

results = injector.run_systematic_analysis(

    data=research_dataset,**Application**: Architecture selection for critical systems  

    targets=ground_truth,

    criterion=classification_accuracy

)

```## Experimental Protocol### [Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb)**[Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb)** - Interactive tutorial with visualizations



### Citation



When using these examples in research, please cite:Both studies implement the IEEE 754-compliant fault injection methodology described in our research paper. Key experimental parameters:Interactive research notebook with comprehensive fault injection analysis and visualization tools.

```bibtex

@software{seu_injection_framework,

  author = {William Dennis},

  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},- **Fault Model**: Single Event Upset (SEU) via targeted bit manipulation

  year = {2025},

  url = {https://github.com/wd7512/seu-injection-framework}- **Coverage**: Exhaustive layer-wise vulnerability mapping  

}

```- **Statistics**: Multiple-trial averaging with confidence intervals### basic_cnn_robustness.py## Results



For methodology details and validation results, see the main repository documentation.- **Validation**: Baseline accuracy verification and repeatability testing

**Objective**: Single-architecture vulnerability analysis  

## Research Applications

**Method**: Systematic bit-flip injection across network layers  Both scripts generate detailed reports showing:

### Adapting for New Studies

**Application**: Space mission deployment assessment  - Which layers are most vulnerable to bit flips

```python

from seu_injection import SEUInjector- How different bit positions affect accuracy  

from seu_injection.metrics import classification_accuracy

### architecture_comparison.py  - Robustness comparison across architectures

# Configure for your research

injector = SEUInjector(your_model)**Objective**: Comparative robustness evaluation  

results = injector.run_systematic_analysis(

    data=research_dataset,**Method**: Standardized fault injection protocol across multiple architectures  ## Customization

    targets=ground_truth,

    criterion=classification_accuracy**Application**: Architecture selection for critical systems  

)

```Easy to adapt for your own models and data:



### Citation## Experimental Protocol



When using these examples in research, please cite:```python

```bibtex

@software{seu_injection_framework,Both studies implement the IEEE 754-compliant fault injection methodology described in our research paper. Key experimental parameters:# Use your own model

  author = {William Dennis},

  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},injector = SEUInjector(your_model)

  year = {2025},

  url = {https://github.com/wd7512/seu-injection-framework}- **Fault Model**: Single Event Upset (SEU) via targeted bit manipulation

}

```- **Coverage**: Exhaustive layer-wise vulnerability mapping  # Use your own data



For methodology details and validation results, see the main repository documentation.- **Statistics**: Multiple-trial averaging with confidence intervalsresults = injector.run_seu(your_data, your_labels, criterion=accuracy)

- **Validation**: Baseline accuracy verification and repeatability testing```



## Research ApplicationsFor questions, see the main README or open an issue.

### Adapting for New Studies

```python
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# Configure for your research
injector = SEUInjector(your_model)
results = injector.run_systematic_analysis(
    data=research_dataset,
    targets=ground_truth,
    criterion=classification_accuracy
)
```

### Citation

When using these examples in research, please cite:
```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework}
}
```

For methodology details and validation results, see the main repository documentation.