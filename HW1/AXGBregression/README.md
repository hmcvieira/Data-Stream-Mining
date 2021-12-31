#Adaptive Extreme Gradient Boost Regressor for Evolving Datastreams

## Table of contents
* [General info](#general-info)
* [Dependencies](#dependencies)
* [Setup](#setup)

## General info
This project is an implementation of XGBoost regressor for evolving datastreams. It was inspired in the classifier implementation by J. Montiel et al. (2020) https://jacobmontiel.github.io/publication/2020-axgb/.
Its implementation is on python 3 and built on top of scikit-multiflow. 
The code is opensource and available on github: https://github.com/hmcvieira/Data-Stream-Mining/tree/main/HW1/AXGBregression
	
## Dependencies
This project is dependent on the following python libraries:

* skmultiflow
* xgboost
* numpy

	
## Setup

Install the required libraries.
An example execution script is provided in axbg_regression_test.py. Replace the input stream, the evaluator settings and the model parameters by the desired values. 
The model parameters are the following:

*update_strategy: the ensemble update strategy ('push' or 'replace')
*n_estimators: Number of members in the ensemble
*learning_rate: Learning rate or eta
*max_depth: Max depth for each tree in the ensemble
*max_window_size: Max window size
*min_window_size: Set to activate the dynamic window strategy
*detect_drift: Enable/disable drift detection
*threshold: pagehinkley threshold

