# covid-data-mining

## Overview
Data files were taken [beoutbreakprepared / nCoV2019](https://github.com/beoutbreakprepared/nCoV2019) and [CSSEGISandData /
COVID-19](https://github.com/CSSEGISandData/COVID-19) data repository

The goal of the project is to build a classifier which can predict the outcome of an individual case given the set of features from the aforementioned data sets.

## Scripts
ID | Name | Purpose | Additional Info
:---: | :---: | :---: | :---: 
1.1-1.5 | `preprocessing.py` | clean, impute, and transform data set | generated `Merged_Data_Sets.csv` is too large for github. Must be run within src directory 
2.1-2.4 | `classification.py` | generate models using `Merged_Data_Sets.csv` | Creates knn and adaboost classifiers
 N/A |`main.py` | not used currently | N/A

## Tuning Hyperparameters
Both `KNN` and `ADA` models were tuned using a grid search approach. 
`KNN` hyperparameters considered: `Leaf size, N-neighbors, k-norm`
`ADA` hyperparameters considered: `Max-depth, N-estimators, Min-samples leaf`

 
 
## Downloading Models and Data
The models and cleaned data set can be found here:
https://drive.google.com/drive/folders/1d4h1lKy9umfL5JbVnwpvyMTyTmrO6i6Y?usp=sharing


## Requirements
The project uses `Python 3.8` and the external libraries found in `requirements.txt`