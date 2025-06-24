# Bayesian Optimization for two coupled cuboid muscles

## Setup
- Two ellipsoid muscle geometries. The 3d geometries are created in ```ellipsoid_mesh_generation.py```, and the fibers' geometries are given by ```mesh_left.json``` and ```mesh_right.json```.
- The solvers for the contraction are coupled mechanics solver and fastmonodomain solver. In the contraction process we set dynamic to `True`, fix one end of the muscles each and let one end of the muscles free. The contraction forces of one muscle is transfered over to the other one in every timestep via an imaginary spring. The spring's restoring force is 500N by default, but can be modified in ```variables/variables.py``` by changing ```tendon_spring_constant``` to the desired value.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of contracting the muscles, go to build_release and run:
```
./two_muscles_contraction ../settings_contraction.py
```
