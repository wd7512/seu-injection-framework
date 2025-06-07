# seu-injection-framework
[A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)

# Structure

This work will be structued as a python module and ideally added to pypi to allow for installation with pip.

For now please clone this repo and place it alongside your working code to be imported.

# Code log

### v0.0.2
*date: 07.06.2025*

Things added 
- attack() now returns the value before and after the bitflip

### v0.0.1

*date: 05.06.2025*

I have a simple version of the code working in the Example_Attack_Notebook.ipynb. This pulls from some of my legacy code which we will want to remove and the new attack.py module. 

NOTE: This is the MVP and will only work for a NN with binary classification, similar to the example one. Other forms of NN should work, i.e. CNN so long it is binary classification. 