import subprocess
import sys
import os
import shlex
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, PosteriorMean
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import time


#If you want to call this file, you have two options:
#>python BayesOpt.py
#or
#>python BayesOpt.py matern 1.5 const fixed_noise ei stopping_xy
#You can change these inputs to any ones of it kind, see options below. A chosen option becomes True, every other option
#of this kind becomes False. You can leave any option out, then the current setup in here is being chosen. The order
#also doesn't matter.

########################################################################################################################
#Customize code here

stopping_y = True
improvement_threshold = 1e-4
num_consecutive_trials = 3

bounds = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],  # Lower bounds for each dimension
    [1.0, 1.0, 1.0, 1.0]   # Upper bounds for each dimension
])

num_initial_trials = 2 #this needs to be >=2

all_inputs = []
all_outputs = []
########################################################################################################################


def adjust_input(x):
    x_1 = x[0].item() #kernel: rbf, matern 0.5, 1.5, 2.5
    x_2 = x[1].item() #mean: const, zero
    x_3 = x[2].item() #acquisition function: EI, PI, KG, ES
    x_4 = x[3].item() #stopping: y, xy
    
    if x_1 < 0.25:
        x_1 = 0
    elif x_1 < 0.5:
        x_1 = 0.25
    elif x_1 < 0.75:
        x_1 = 0.5
    else:
        x_1 = 0.75

    if x_2 < 0.5:
        x_2 = 0
    else:
        x_2 = 0.5
    
    if x_3 < 0.25:
        x_3 = 0
    elif x_3 < 0.5:
        x_3 = 0.25
    elif x_3 < 0.75:
        x_3 = 0.5
    else:
        x_3 = 0.75

    if x_4 < 0.5:
        x_4 = 0
    else:
        x_4 = 0.5

    x = [x_1, x_2, x_3, x_4]
    
    return x

def simulation(x):
    x = adjust_input(x)  
    
    try:
        position = all_inputs.index(x)
        y = all_outputs[position]
    except ValueError:
        all_inputs.append(x)
        x_1 = x[0]
        x_2 = x[1] 
        x_3 = x[2] 
        x_4 = x[3]  
        y = torch.tensor([ x_1 + x_2 + x_3 + x_4 ], dtype=torch.double)
        all_outputs.append(y.item())
    
    return y

"""

def simulation(force):
    force = force.item()
    print("start simulation with force", force)
    individuality_parameter = str(int(time.time()))+str(force)
    command = shlex.split(f"./incompressible_mooney_rivlin ../settings_force.py incompressible_mooney_rivlin {force} {individuality_parameter}")
    subprocess.run(command)

    print("end simulation")

    f = open("muscle_length_prestretch"+individuality_parameter+".csv")
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        prestretch = float(row[1]) - float(row[0])
        print("The muscle was stretched ", prestretch)
    f.close()

    f = open("muscle_length_contraction"+individuality_parameter+".csv")
    reader = csv.reader(f, delimiter=",")
    muscle_length_process = []
    for row in reader:
        for j in row:
            muscle_length_process.append(j)
        
    contraction = float(muscle_length_process[0]) - float(muscle_length_process[-2])
    print("The muscle contracted ", contraction)
    f.close()

    return contraction
"""

class CustomSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        likelihood = GaussianLikelihood()
        kernel = ScaleKernel(MaternKernel(nu=1.5))

        mean = ZeroMean()

        input_transform = Normalize(d=train_X.shape[-1])
        output_transform = Standardize(m=1)

        super().__init__(train_X,
                         train_Y,
                         likelihood=likelihood,
                         covar_module=kernel,
                         mean_module=mean,
                         input_transform=input_transform,
                         outcome_transform=output_transform,
                        )


os.chdir("build_release")


starting_time = time.time()

sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True)

initial_x = sobol.draw(num_initial_trials, dtype=torch.double)

with open("Optimize_BayesOpt_outputs.csv", "w"):
    pass

initial_y = torch.tensor([])
for force in initial_x:
    y = torch.tensor([[simulation(force)]])

    with open("Optimize_BayesOpt_outputs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([force.numpy(), y.item()])

    initial_y = torch.cat([initial_y, y])
    print(initial_y)


gp = CustomSingleTaskGP(initial_x, initial_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
print("Noise:", gp.likelihood.noise.mean().item())

num_iterations = 100
best_value = -float('inf')
no_improvement_trials = 0
counter = num_initial_trials

for i in range(num_iterations):
    acq_fct = ExpectedImprovement(model=gp, best_f=initial_y.max())
    
    candidate, acq_value = optimize_acqf(
        acq_function=acq_fct,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=256,
    )

    new_y = torch.tensor([[simulation(candidate[0])]])

    with open("Optimize_BayesOpt_outputs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy(), new_y.item()])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
    print("Noise:", gp.likelihood.noise.mean().item())

    counter += 1

    current_value = new_y.item()
    if current_value > best_value + improvement_threshold:
        best_value = current_value
        no_improvement_trials = 0
    elif len(initial_x) > num_initial_trials:
        no_improvement_trials += 1
    if no_improvement_trials >= num_consecutive_trials:
        print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.numpy()}, Value = {current_value}, Best Value = {best_value}")
        print("Stopping criterion met. No significant improvement for consecutive trials.")
        print("Number of total trials: ", i+1+num_initial_trials)
        break

    print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.numpy()}, Value = {current_value}, Best Value = {best_value}")



max_index = torch.argmax(initial_y)
maximizer = initial_x[max_index]
best_y = initial_y[max_index]

with open("Optimize_BayesOpt_outputs.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([counter])
    writer.writerow([maximizer.numpy(), best_y.item()])
    writer.writerow([time.time()-starting_time])
