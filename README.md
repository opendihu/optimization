# Bayesian Optimization

## Description
Bayesian Optimization is a method to find maxima of blackbox functions with a relatively low number of function evaluations. This repository has started as a Bachelor Thesis (https://github.com/opendihu/optimization/tree/Bachelor-thesis). It contains files for Bayesian Optimization that can be used by itself or be connected to OpenDiHu to use a simulation as a function evaluation. 

## Dependendcies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, subprocess, sys, os, shlex, csv, time, signal

OpenDiHu (if used): Version aadd55a4 (https://github.com/opendihu/opendihu/tree/aadd55a47fde8031cc4629ba138e949d54c26661)

## Setup
Inside "BayesianOptimization" are two setups for Bayesian Optimization that are both set up to optimize an easy dummy function. Inside "1D" is the setup for a one-dimensional function, inside "nD" the setup for a n-dimensional function.

Inside "examples" are a few examples where Bayesian Optimization has been used:
- biceps: Has not been added yet
- coupled_cuboid_muscles: Has not been added yet
- cuboid_muscle: This was the main focus of the mensioned Bachelor Thesis. It uses Bayesian Optimization to find the prestretch that corresponds to the greatest range of motion of a dummy cuboid muscle.
- test_functions: This can be used to test different Bayesian Optimization models on several test functions. These functions have different characteristics, so that you can choose a model that works best for the kind of functions you are looking for.

More details can be found in the corresponding ReadMe files.

## Results
### The optimal Bayesian Optimization model over all test functions
The word "optimal" is very subjective. In this case we are looking for a model that balances high accuracy with a low number of evaluations. Comparing all optional models averaged over all test functions, we find the following model as the best one:
- Kernel: Mátern kernel with smoothness parameter 0.5
- Mean: Constant mean
- Acquisition function: Entropy search

It takes on average 7.352 evaluations and finds local maxima in 85.2% and global maxima in 79.8% of cases. There are other models with a higher accuracy and models with a lower number of evaluations, but no model balaces it as well as this one does. That has been the conclusion of the Bachelor Thesis.
### The prestretch for the best range of motion of a cuboid muscle
