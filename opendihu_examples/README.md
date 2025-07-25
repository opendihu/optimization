# Bayesian Optimization for a cuboid muscle model

## Description
In this folder there are several examples of different muscles with different solvers and boundary conditions. Here we provide a general explanation on the differences of the different cases and on how to build and run OpenDiHu cases.

## Case Overview

In "isotonic_contraction" we take a single muscle and simulate its contraction after activation with one end fixed and one end free. We can use Bayesian Optimization to look for optimal parameters like the prestretch force that maximizes the muscle's contraction length. 

In "isometric_contraction" we take a single muscle and simulate its contraction after activation. The difference to "isotonic_contraction" is that the muscle is fixed on both ends. We can use Bayesian Optimization to look for optimal parameters like the prestretch force that maximizes force generation within the muscle. 

In "paired_muscles" we are not looking at a single muscle, but two muscles that are connected via a simplified tendon that behaves like a spring. We can use 2D Bayesian Optimization to look for parameters like the prestretch forces that maximize the contraction lengths of the paired muscles.

In "prestretch_force_for_given_length" we use a bisection method to find the prestretch force that is required to stretch a muscle to a given length. 

## How to build?
Follow OpenDiHu's documentation for [installation](https://opendihu.readthedocs.io/en/latest/user/installation.html#), go to the corresponding folder and run 
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
