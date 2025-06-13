import numpy as np

data = [[0.0,0.03874795662348074],
[0.5,0.03896244282613981],
[1.0,0.03930557118127274],
[1.5,0.039648970664906844],
[2.0,0.039964688395255396],
[2.5,0.04023916048403001],
[3.0,0.040470024060148546],
[3.5,0.04066781249739465],
[4.0,0.040834042794013814],
[4.5,0.040972935743122836],
[5.0,0.04108984752449413],
[5.5,0.04118923404544591],
[6.0,0.04127467775373131],
[6.5,0.04134900908023411],
[7.0,0.04141444567191707],
[7.5,0.04147271932276298],
[8.0,0.041525181683942244],
[8.5,0.04157288858867351],
[9.0,0.04161666592231335],
[9.5,0.04165716061433811],
[10.0,0.03911523801322557]]

def target_function(x):
    for i in range(len(data)):
        if x < data[i][0]:
            x = x - data[i-1][0]
            y = data[i-1][1] + x*(data[i][1]-data[i-1][1])/(data[i][0]-data[i-1][0])
            return y
        if x == data[-1][0]:
            return data[-1][1]

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
ES_low_uncertainty = False

stopping_y = True
improvement_threshold = 1e-5
stopping_xy = False
x_range = 5e-2
num_consecutive_trials = 3

test_function_number = 0

#Minor changes:
fixed_Yvar = 1e-6
sobol_on = True
num_initial_trials = 2 #this needs to be >=2
visualize = True
add_points = True
lower_bound = 0.0
upper_bound = 10.0