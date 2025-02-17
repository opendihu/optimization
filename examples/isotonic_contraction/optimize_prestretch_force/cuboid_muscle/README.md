# Bayesian Optimization for a cuboid muscle model

## Dependencies
OpenDiHu: Version 1.5 (https://github.com/opendihu/opendihu/tree/v1.5)

Python: Python 3.10.12

Required Python libraries: botorch, torch, numpy, matplotlib, subprocess, sys, os, shlex, csv, time, signal


## Setup
- A dummy cuboid muscle geometry. 
- The solvers for both stretching and contraction are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to `False` and add boundary conditions that simulate the muscle being fixed at one side and being pulled at from the other side. In the contraction process we set dynamic to `True` and let the ends of the muscle free. 
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to build?
Follow OpenDiHu's documentation (https://opendihu.readthedocs.io/en/latest/index.html) for installation, then run 
```
mkorn && sr
```
For a debug build, look into the documentation. 

## How to run?

### Running an OpenDiHu simulation
To run a single simulation of stretching a muscle with a certain force and then contract it, go to build_release and run:

```
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin 10.0
```
To run a single simulation of only stretching a muscle with a certain force, go to build_release and run:
```
./incompressible_mooney_rivlin_prestretch_only ../prestretch_tensile_test.py incompressible_mooney_rivlin_prestretch_only 10.0
```
### Running an optimization loop

To run an optimization process with Bayesian Optimization, choose the BO model you want to use, go to cuboid_muscle and run:
```
python BayesOpt.py matern 0.5 const fixed_noise es stopping_xy
```
To evaluate a Bayesian Optimization model by averaging the results over multiple iterations, go to cuboid_muscle and run:
```
python Evaluate_BayesOpt_model.py matern 0.5 const fixed_noise es stopping_xy
```
To run an optimization process with Bayesian Optimization to optimize a test function, choose the BO model you want to use, go to cuboid_muscle and run:
```
python BayesOpt_test_functions.py matern 0.5 const fixed_noise es stopping_xy 1
```
More detailed instructions can be found inside the respective files.

