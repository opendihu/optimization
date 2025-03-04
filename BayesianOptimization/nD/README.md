# General 1D Bayesian Optimization

## Description
Bayesian Optimization is a method to find maxima of blackbox functions with a relatively low number of function evaluations. This folder contains the files to maximize a continuous function from R^n to R. Because infinity is hard to compute, the function actually has to be from an interval [a_1,b_1]x[a_2,b_2]x...x[a_n,b_n] to R. 

## Dependendcies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, os, csv, time

## Setup
The details of the optimization process can be set up in "setup_BayesOpt_general_nD.py". There you can change the target function to the function you want to optimize, the corresponding bounds and the parameters that define the optimization model you want to use.

## How to run?
To optimize the target function inside "setup_BayesOpt_general_nD.py" run
```
python BayesOpt_general_nD.py
```
To view the results of the optimization process, go to "build_release" that has been created after the optimization process and open the file "BayesOpt_outputs_{individuality_parameter}.csv". The corresponding individuality parameter is the last entry in the file "BayesOpt_global_individuality_parameters.csv". There you find the number of trials, the elapsed time and the optimizer and optimum.
