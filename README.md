# A cuboid muscle model

## Setup
- A dummy cuboid muscle geometry. 
- The solvers for both stretching and contraction are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to False and add boundary conditions that simulate the muscle being fixed at one side and being pulled at from the other side. In the contraction process we set dynamic to True and don't add any boundary conditions. 
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
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin 10.0
```
To run a single simulation of only stretching a muscle with a certain force, go to build_release and run:
```
./incompressible_mooney_rivlin_prestretch_only ../prestretch_tensile_test.py incompressible_mooney_rivlin_prestretch_only 10.0
```
To run an optimization process with Bayesian Optimization, choose the BO model you want to use, go to cuboid_muscle and run:
```
python BayesOpt.py matern 1.5 zero fixed_noise ei stopping-xy
```
To evaluate a Bayesian Optimization model by averaging the results over multiple iterations, go to cuboid_muscle and run:
```
python Evaluate_BayesOpt_model.py matern 1.5 zero fixed_noise ei stopping-xy
```
