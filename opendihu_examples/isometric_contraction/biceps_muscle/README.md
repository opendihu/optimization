# Bayesian Optimization for a biceps muscle model

## Setup
- A biceps muscle geometry. 
- The solvers for both stretching and contraction are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to `False` and add boundary conditions that simulate the muscle being fixed at one side and being pulled at from the other side. In the contraction process we set dynamic to `True` and fix both ends of the muscle in place by boundary conditions. 
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of stretching and contracting a muscle, go to build_release and run:
```
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py ramp.py --prestretch_force 10.0
```
This computes everything in serial and therefore will take a long time. If you can, use the alternative:
```
mpirun -n 16 ./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py ramp.py --prestretch_force 10.0
```
`-n` is the number of MPI ranks that will be used. If you choose, e.g., ,`-n 4`, the domain will be partitioned into 4 subdomains. However, not all numbers of ranks are supported, since some partitions might end up empty, which  will throw an error. 

## Optimization
To run an optimization process, choose the optimization model, modify the parameters inside "setup_BayesOpt_biceps_muscle.py" and run
```
python BayesOpt_biceps_muscle.py
```
With this case we can use Bayesian Optimization to optimize the contraction force. This contraction force in a single time step is the average traction in the direction of the fibers at the left end of the muscle. Our function f: R -> R maps a prestretch force to the maximal contraction force of a muscle, that has been stretched with the prestretch force before contracting. One function evaluation is one simulation of the muscle. This way the optimization process outputs the prestretch force that leads to the greatest contraction force of our given muscle. Using the Matern kernel with nu=0.5, the constant mean function and the entropy search acquisition function, the plot of the optimization process looks like the following:

```
To Do
```
