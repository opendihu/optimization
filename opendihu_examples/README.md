# Bayesian Optimization for a cuboid muscle model

## Description
In this folder are several examples of different muscles with different solvers and boundary conditions. Inside each example is a short ReadMe file that summarizes its setup. This is a general explanation on how to build and run an example.

## How to build?
Follow OpenDiHu's documentation (https://opendihu.readthedocs.io/en/latest/index.html) for installation, go to the corresponding folder and run 
```
mkorn && sr
```
For a debug build, look into the documentation. 

## How to run?

### Running an OpenDiHu simulation
To run a single simulation of the corresponding muscle, go to build_release and run:

```
./{scenario_name} ../{settings_file}.py {muscle_material_name} {possible_other_parameters}
```
Example for "isotonic_contraction/optimize_prestretch_force/cuboid_muscle":
```
./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin 10.0
```
### Running an optimization loop

To run an optimization process with Bayesian Optimization, choose the BO model you want to use, go to the corresponding folder and run:
```
python BayesOpt_{example_name}.py
```
More detailed instructions can be found inside the respective files.

