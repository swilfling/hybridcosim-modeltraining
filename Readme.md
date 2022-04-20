# Model Training and FMU Generation

This folder contains tools for model training and FMU generation. 

## Structure:
- modelTraining: Contains model training and preprocessing. The main file is run.py.

## FMU Interface:

The interface for the FMU is stored inside a CSV file with the following layout:

``<feature name>;<In_Out>;Initialization Value;<StatDyn> ``

- Feature Name: name of the feature 
- In_Out: defines whether feature is input or output of the FMU (not used in framework yet)
- Initialization Value: initialization for FMU generation
- StatDyn: defines whether a feature is static or dynamic

The possible values for entries inside the interface file are:

``<feature name>;<In>/<Out>;Initialization Value;<static>/<dynamic> ``
