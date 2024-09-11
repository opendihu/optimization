# A cuboid muscle model

## Setup
- A dummy cuboid muscle geometry. 
- Solver comprises hyperelasticity solver for the prestretch and both mechanics solver + fastmonodomain solver for the contraction. 
- It uses the CellML model "hodgkin_huxley-razumova".
- No preCICE involved. 

### How to build?
Follow OpenDiHu's documentation for installation, then run 
```
mkorn && sr
```
For a debug build, look into the documentation. 

### How to run?
To run a single simulation of stretching a muscle with a certain force and then contract it, go to build_release and run:

```
./incompressible_mooney_rivlin ../settings_force.py incompressible_mooney_rivlin 10.0
```
To run a single simulation of only stretching a muscle with a certain force, go to build_release and run:
```
./incompressible_mooney_rivlin_2 ../prestretch_tensile_test.py incompressible_mooney_rivlin_2 10.0
```
To run an optimization process with Bayesian Optimization, choose the BO model you want to use, go to cuboid_muscle and run:
```
python BayesOpt.py matern 1.5 zero fixed_noise ei stopping-xy
```
To evaluate a Bayesian Optimization model by averaging the results over multiple iterations, go to cuboid_muscle and run:
```
python Evaluate_BayesOpt_model.py matern 1.5 zero fixed_noise ei stopping-xy
```
