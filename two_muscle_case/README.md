# Bayesian Optimization for two coupled cuboid muscles

## Setup
- Two dummy cuboid muscle geometries. 
- The solvers for the contraction are coupled mechanics solver and fastmonodomain solver. In the contraction process we set dynamic to `True`, fix one end of the muscles each and let one end of the muscles free. The contraction forces of one muscle is transfered over to the other one in every timestep.
- It uses the electrophysiology CellML model "hodgkin_huxley-razumova" and the incompressible mechanics model "Mooney-Rivlin".
- No preCICE involved. 

## How to run
To run a single simulation of contracting the muscles, go to build_release and run:
```
./two_muscles_contraction ../settings_contraction.py
```

For the simulation of a spring between the muscles, go to variables/variables.py and set ```tendon_spring_simulation = True```. For direct force transfer, set ```tendon_spring_simulation = False```.