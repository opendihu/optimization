# Bayesian Optimization for a cuboid muscle model

## Setup
- Two dummy cuboid muscle geometries. 
- The solvers for both stretching and contraction are coupled mechanics solver and fastmonodomain solver. In the prestretch process we set dynamic to `False` and add boundary conditions that simulate the muscles being fixed at one side (the inner side) and being pulled at from the other side (the outer side). In the contraction process we set dynamic to `True`, fix the outer end of the muscles and apply a Neumann boundary condition at the inner end of the muscles. The force in the Neumann boundary condition is updated at every timestep and it is different for each muscle. We compute the force such that it mimics the effect of an immaginary tendon connecting the two muscle. We assume the tendon behaves like a spring and compute the force using Hooke's law: $F_{tendon} = k_{tendon} (l − l_0 )$. The spring's constant is $10N/cm$ by default, but can be modified in ```variables/variables.py``` by changing ```tendon_spring_constant``` to the desired value.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of stretching and contracting the muscles, go to build_release and run:
```
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin 2 5
```
To run a single simulation of only prestretch, go to build_release and run:
```
./muscle_prestretch ../settings_prestretch.py incompressible_mooney_rivlin 2 5
```
To modify the prestretch forces, change the input parameters `2` and `5`. The first parameter corresponds to muscle 1, the second one to muscle 2. If only one force parameter is given, both muscles use the same prestretch force.