# Model Training 

This folder contains tools for model training. 

## Structure:
- Training
- Feature Selection
- Training Utilities
- General Utilities
- Configuration

## Setup Instructions

The ModelTraining repository contains the submodule datamodels. 

After cloning the ModelTraining repository, please execute the following instructions:

``git submodule init``

``git submodule update``

These instructions register the datamodels submodule.

Now, in the datamodels submodule, all git commands can be executed for the datamodels submodule. 

## FMU Interface:

The interface for the FMU is stored inside a CSV file with the following layout:

``<feature name>;<In_Out>;Initialization Value;<StatDyn> ``

- Feature Name: name of the feature 
- In_Out: defines whether feature is input or output of the FMU (not used in framework yet)
- Initialization Value: initialization for FMU generation
- StatDyn: defines whether a feature is static or dynamic

The possible values for entries inside the interface file are:

``<feature name>;<In>/<Out>;Initialization Value;<static>/<dynamic> ``
