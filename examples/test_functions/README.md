# Bayesian Optimization for test functions

## Dependendcies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, subprocess, sys, os, shlex, csv, time, signal


## Setup
This is the setup for Bayesian optimization for different test functions. These functions are:
- -3x(x-1.3) + 0.3
- exp(-(5x-3)^2) + 0.2 exp(-(30x-22)^2)
- exp(-(5x-5)^2) n sin(5x-1.5) +x
- exp(-(10x -2)^2 ) + exp(-(10x-6)^2/10) + 1/((10x)^2 +1)
- 0.5 - 3x(x-1)\*sin(5x)
- sin(5x)^2
- x + 0.5x^2 \*sin(18x)
- 1-|x-0.5|
- x^0.5 -exp(5(x-1))

Apart from optimizing these functions, you can evaluate different Bayesian optimization models by averaging accuracy and number of evaluations over all the test functions.

## How to build
Create a folder with the name "build_release" inside the folder "test_functions". This will contain all the results of the optimization processes.


## How to run?
To evaluate a Bayesian Optimization model by averaging the results over multiple iterations, run:
```
python Evaluate_BayesOpt_model.py matern 0.5 const fixed_noise es stopping_xy
```
To run an optimization process with Bayesian Optimization to optimize a test function, choose the BO model you want to use and run:
```
python BayesOpt_test_functions.py matern 0.5 const fixed_noise es stopping_xy 1
```

