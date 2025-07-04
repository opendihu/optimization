# Bayesian Optimization for a cuboid muscle model

## Setup
- Two dummy cuboid muscle geometries. 
- The solvers for both stretching and contraction are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to `False` and add boundary conditions that simulate the muscles being fixed at one side (the inner side) and being pulled at from the other side (the outer side). In the contraction process we set dynamic to `True` and fix the outer ends of the muscles in place and connect the inner ends together via an imaginary spring.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of stretching and contracting the muscles, go to build_release and run:
```
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin 2
```
To run a single simulation of only prestretch, go to build_release and run:
```
./muscle_prestretch ../settings_prestretch.py incompressible_mooney_rivlin 2
```To modify the prestretch force, change the input parameter `2`.
