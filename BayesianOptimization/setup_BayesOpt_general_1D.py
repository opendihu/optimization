import numpy as np

def target_function(x):
    x = x.numpy()[0]
    return x**2

#Major changes:
nu = 0.5
matern = True
rbf = False

const = True
zero = False

fixed_noise = True
variable_noise = False

EI = False
PI = False
KG = False
ES = True

stopping_y = False
improvement_threshold = 1e-4
stopping_xy = True
x_range = 5e-2
num_consecutive_trials = 3

test_function_number = 0

#Minor changes:
fixed_Yvar = 1e-6
sobol_on = True
num_initial_trials = 2 #this needs to be >=2
visualize = True
add_points = False
lower_bound = 0.0
upper_bound = 1.0