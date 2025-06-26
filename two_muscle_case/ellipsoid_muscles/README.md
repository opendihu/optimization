# Bayesian Optimization for two coupled cuboid muscles

## Setup
- Two ellipsoid muscle geometries. The 3d geometries are created in ```ellipsoid_mesh_generation.py```, and the fibers' geometries are given by ```mesh_left.json``` and ```mesh_right.json```.
- The solvers for the contraction are coupled mechanics solver and fastmonodomain solver. In the contraction process we set dynamic to `True`, fix the outer end of the muscles and apply a Neumann boundary condition at the inner end of the muscles. The force in the Neumann boundary condition is updated at every timestep and it is different for each muscle. We compute the force such that it mimics the effect of an immaginary tendon connecting the two muscle. We assume the tendon behaves like a spring and compute the force using Hooke's law: $F_{tendon} = k_{tendon} (l − l_0 )$. The spring's constant is $500N/cm$ by default, but can be modified in ```variables/variables.py``` by changing ```tendon_spring_constant``` to the desired value.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of contracting the muscles, go to build_release and run:
```
./two_muscles_contraction ../settings_contraction.py
```
