# Bayesian Optimization


## Description
Bayesian Optimization is a method to find maxima of blackbox functions with a relatively low number of function evaluations. For that we are using the python package [botorch](https://botorch.org/). This repository contains files for Bayesian Optimization that can be used standalone or call the neuromuscular simulation framework [OpenDiHu](https://github.com/opendihu/opendihu) to find the optimal parameter(s) of a skeletal muscle simulation. 

This project started as Lukas Bauer's bachelor thesis, that led to the [first version](https://github.com/opendihu/optimization/tree/Bachelor-thesis) of the repository. The thesis can be accessed at [OPUS](https://elib.uni-stuttgart.de/handle/11682/16797), the online publication library from the University Stuttgart. 

## Dependencies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, subprocess, sys, os, shlex, csv, time, signal

OpenDiHu (if used): Version [1.5](https://github.com/opendihu/opendihu/tree/v1.5)

## Setup
Inside [BayesianOptimization](https://github.com/opendihu/optimization/tree/main/BayesianOptimization) is the setup for a 1D Bayesian Optimization that is set up to optimize an easy dummy function. In the subfolder "nD" is a similar setup for a function from $\mathbb{R}^n$ to $\mathbb{R}$.

Inside [opendihu_examples](https://github.com/opendihu/optimization/tree/main/opendihu_examples) you can find the examples "isotonic_contraction", "isometric_contraction", "paired_muscles" and "prestretch_force_for_given_length". An overview of the different cases described is provided in the [ReadMe](opendihu_examples/README.md) file.

Inside [test_functions](https://github.com/opendihu/optimization/tree/main/test_functions) there are several made-up functions which can be used to test different Bayesian Optimization models. These functions have different characteristics, so that you can choose a model that works best for the kind of functions you are looking for.

## Results
The results of the mentioned Bachelor Thesis are in the [test_functions](BayesOpt_examples/test_functions/README.md) and [isotonic_cuboid_contraction](opendihu_examples/isotonic_contraction/optimize_prestretch_force/cuboid_muscle/README.md) files. A discussion of when to use which parameter setup can be found [here](BayesianOptimization/discussion_parameters.md). Other results can be found in the corresponding ReadMes of the different cases.