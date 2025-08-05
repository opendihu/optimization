# Bayesian Optimization for test functions

## Dependendcies
Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, subprocess, sys, os, shlex, csv, time, signal


## Setup
This is the setup for Bayesian optimization for different test functions. These functions are:
- $-3x(x-1.3) + 0.3$
- $e^{(-(5x-3)^2)} + 0.2 e^{(-(30x-22)^2)}$
- $e^{-(5x-5)^2} \cdot sin(5x-1.5) +x$
- $e^{(-(10x -2)^2 )} + e^{-\frac{(10x-6)^2}{10}} + \frac{1}{((10x)^2 +1)}$
- $0.5 - 3x(x-1)\cdot sin(5x)$
- $sin(5x)^2$
- $x + 0.5x^2 \cdot sin(18x)$
- $1-|x-0.5|$
- $x^{\frac{1}{2}} -e^{(5(x-1))}$

Apart from optimizing these functions, you can evaluate different Bayesian optimization models by averaging accuracy and number of evaluations over all the test functions.


## How to run?
To evaluate a Bayesian Optimization model by averaging the results over multiple iterations, run:
```
python Evaluate_BayesOpt_model.py matern 0.5 const fixed_noise es stopping_xy
```
To run an optimization process with Bayesian Optimization to optimize a test function, choose the BO model you want to use and run:
```
python BayesOpt_test_functions.py matern 0.5 const fixed_noise es stopping_xy 1
```

## Results

### The optimal Bayesian Optimization model over all test functions
This was discussed in the mentioned [Bachelor Thesis](https://elib.uni-stuttgart.de/handle/11682/16797). Firstly, we should clarify what we mean by "optimal". In this case we are looking for a model that balances high accuracy with a low number of evaluations. Comparing all optional models that botorch has to offer, averaged over all test functions, we find the following model as the best one:
- Kernel: Mátern kernel with smoothness parameter 0.5
- Mean: Constant mean
- Acquisition function: Entropy search

It takes on average 7.352 evaluations and finds local maxima in 85.2% and global maxima in 79.8% of cases. There are other models with a higher accuracy and models with a lower number of evaluations, but no model balaces it as well as this one does.
