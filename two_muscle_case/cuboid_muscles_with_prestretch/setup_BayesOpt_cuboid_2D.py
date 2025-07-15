import torch
import numpy as np
import time
import shlex
import subprocess
import csv

def target_function(forces):
    force1 = forces.numpy()[0]
    force2 = forces.numpy()[1]

    print("start simulation with forces "+ str(force1) + " and",force2)
    individuality_parameter = str(force1)+str(force2)
    command = shlex.split(f"./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin {force1} {force2} {individuality_parameter}")
    subprocess.run(command)

    print("end simulation")

    f = open("muscle_length_contraction_1_"+individuality_parameter+".csv")
    reader = csv.reader(f, delimiter=",")
    muscle_length_process1 = []
    for row in reader:
        for j in row:
            muscle_length_process1.append(j)
        
    contraction1 = float(muscle_length_process1[0]) - float(muscle_length_process1[-2])
    print("The muscle contracted ", contraction1)
    f.close()

    print("start simulation with forces"+ str(force2) + " and ",force1)
    individuality_parameter = str(int(time.time()))+str(force2)+str(force1)
    command = shlex.split(f"./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py incompressible_mooney_rivlin {force2} {force1} {individuality_parameter}")
    subprocess.run(command)

    print("end simulation")

    f = open("muscle_length_contraction_2_"+individuality_parameter+".csv")
    reader = csv.reader(f, delimiter=",")
    muscle_length_process2 = []
    for row in reader:
        for j in row:
            muscle_length_process2.append(j)
    contraction2 = float(muscle_length_process2[0]) - float(muscle_length_process2[-2])
    print("The muscle contracted ", contraction2)
    f.close()

    return contraction1 + contraction2


#Major changes:
dimension = 2

bounds = torch.tensor([[0.0, 0.0], [30.0, 30.0]], dtype=torch.double)

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
add_points = False
lower_bound = 0.0
upper_bound = 1.0
