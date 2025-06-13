# seu-injection-framework
[A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)

# Structure

This work will be structued as a python module and ideally added to pypi to allow for installation with pip.

For now please clone this repo and place it alongside your working code to be imported.

# Code log

### v0.0.6
*date: 13/06/2025*
- added dockerfile to test if we get a performance boost using wsl/linux
  - added a benchamarking.py file in tests
- removed alternative pytorch requirements

### v0.0.5
*date: 12.06.2025* 
- merge in changes from Research/ViT branch to enable batches during inference in `framework/criterion.py`
- added a few more print statements in `framework/attack.py`
- added the ability to use dataloaders in the criterion which speed up inference

### v0.0.4
*date: 11.06.2025*
- refactored criterion.py to take inputs as (model, X, y) as this is more intuitive

### v0.0.3
*date: 08.06.2025*
- allows `layer_name__` to be specified in the `.run_seu()` function of the injector
- `.run_stochastic_seu()` function added to injector, aimed out larger models where one only wants to tests bitflips on values with probability `p`
- added module `framework/bitflip.py` so there is no reliance on legacy code

### v0.0.2
*date: 07.06.2025*

Things added 
- `attack()` is removed from framework.attack, we now have `injector`, a class to handle seu injections
- tests added for 3 types of nn, NN, CNN, RNN

### v0.0.1

*date: 05.06.2025*

I have a simple version of the code working in the Example_Attack_Notebook.ipynb. This pulls from some of my legacy code which we will want to remove and the new attack.py module. 

NOTE: This is the MVP and will only work for a NN with binary classification, similar to the example one. Other forms of NN should work, i.e. CNN so long it is binary classification. 