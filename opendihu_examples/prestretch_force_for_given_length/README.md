# Bayesian Optimization for a cuboid muscle model

## Setup
- A dummy cuboid muscle geometry. 
- The solvers for stretching are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to `False` and add boundary conditions that simulate the muscle being fixed at one side and being pulled at from the other side.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved.
  
## Goal
We want to stretch a muscle with a force, such that we reach a given length. We use a bisection method to find this force. We guess a force and check how close we are to the given length, and then change our next guess depending on that.

## How to run
To run a single simulation of stretching a muscle, go to build_release and run:
```
./incompressible_mooney_rivlin_prestretch_only ../prestretch_tensile_test.py incompressible_mooney_rivlin_prestretch_only 10.0
```
To find the prestretch force that gives you a certain prestretch extension in cm, stay in the folder prestretch_force_for_given_length and run:
```
python find_prestretch_force.py {prestretch_extension}
```
