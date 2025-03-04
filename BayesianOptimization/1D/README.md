# General 1D Bayesian Optimization

## Description
Bayesian Optimization is a method to find maxima of blackbox functions with a relatively low number of function evaluations. This folder contains the files to maximize a continuous function from R to R. Because infinity is hard to compute, the function actually has to be from an interval [a,b] to R. The resulting optimization process can then also be visualized afterwards.

## Dependendcies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, os, csv, time

## Setup
The details of the optimization process can be set up in "setup_BayesOpt_general_1D.py". There you can change the target function to the function you want to optimize, the bounds and the parameters that define the optimization model you want to use.

## How to run?
To optimize the target function inside "setup_BayesOpt_general_1D.py" run
```
python BayesOpt_general_1D.py
```
To visualize and print the results of the optimization process, open the file "BayesOpt_global_individuality_parameters.csv" inside "build_release" that has been created after the optimization process, and copy the corresponding individuality parameter, for example "_matern_0.5_const_fixed_noise_ES_stopping_xy_130815". Then run
```
python visualize_BayesOpt_general_1D.py _matern_0.5_const_fixed_noise_ES_stopping_xy_130815
```