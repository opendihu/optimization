# Bayesian Optimization for a cuboid muscle model

## Setup
- A dummy cuboid muscle geometry. 
- The solvers for both stretching and contraction are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to `False` and add boundary conditions that simulate the muscle being fixed at one side and being pulled at from the other side until it has reached a given length. In the contraction process we set dynamic to `True` and let the ends of the muscle free.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of stretching and contracting a muscle, go to build_release and run:
```
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin 10.0
```

## Optimization
To run an optimization process, choose the optimization model, modify the parameters inside "setup_BayesOpt_cuboid_muscle.py" and run
```
python BayesOpt_cuboid_muscle.py
```
With this case we can use Bayesian Optimization to optimize the contraction length (length of the muscle before contraction process - length of the muscle after contraction process). Our function f: R -> R maps a prestretch length (increase of muscle length by stretching) to the contraction length of a muscle, that has been stretched until it has reached the prestretch length before contracting. One function evaluation is one simulation of the muscle. This way the optimization process outputs the prestretch length that leads to the greatest contraction length of our given muscle. Using the Matern kernel with nu=0.5, the constant mean function and the entropy search acquisition function, the plot of the optimization process looks like the following:

```
To Do
```