# Bayesian Optimization


## Description
Bayesian Optimization is a method to find maxima of blackbox functions with a relatively low number of function evaluations. This repository contains files for Bayesian Optimization that can be used by itself or be connected to [OpenDiHu](https://github.com/opendihu/opendihu) to use a simulation as a function evaluation. It has started as a [Bachelor Thesis](https://elib.uni-stuttgart.de/handle/11682/16797) with a corresponding [GitHub repository](https://github.com/opendihu/optimization/tree/Bachelor-thesis).


## Dependencies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, subprocess, sys, os, shlex, csv, time, signal

OpenDiHu (if used): Version [1.5](https://github.com/opendihu/opendihu/tree/v1.5)

## Setup
Inside "BayesianOptimization" is the setup for a 1D Bayesian Optimization that is set up to optimize an easy dummy function. In the subfolder "nD" is a similar setup for a function from R^n to R.

Inside "opendihu_examples" are the categories "isotonic_contraction", "isometric_contraction", "paired_muscles" and "prestretch_force_for_given_length". The details about these different categories can be found [here](opendihu_examples/README.md).

Inside "BayesOpt_examples" is the example "test_functions" which can be used to test different Bayesian Optimization models on several test functions. These functions have different characteristics, so that you can choose a model that works best for the kind of functions you are looking for.

## Results
The results of the mentioned Bachelor Thesis are in the [test_functions](BayesOpt_examples/test_functions/README.md) and [isotonic_cuboid_contraction](opendihu_examples/isotonic_contraction/optimize_prestretch_force/cuboid_muscle/README.md) files. A discussion of when to use which parameter setup can be found [here](BayesianOptimization/discussion_parameters.md). Other results can be found in the corresponding ReadMes of the different cases.