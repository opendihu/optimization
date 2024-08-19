import subprocess
import sys
import os
import shlex
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Standardize
from gpytorch.priors import GammaPrior
import time
import botorch
botorch.settings.debug(True)

script_path = os.path.dirname(os.path.abspath(__file__))
var_path = os.path.join(script_path, "variables")
sys.path.insert(0, var_path)

import variables

########################################################################################################################
#Customize code here

#Major changes:
nu = 1.5
matern = True
rbf = False

const = False
zero = True

EI = True
PI = True ########## still needs to be tested
KG = True
ES = True

max_stddev = False
freq = 3

ML = True ################### still needs to be added
full_Bayes = False

#Minor changes:
fixed_Yvar = 1#1e-6
lower_bound = 0.
upper_bound = 20. ########### what should this be?
sobol_on = True
improvement_threshold = 1e-4
num_initial_trials = 2 #this needs to be >=2
num_consecutive_trials = 3
visualize = True
add_points = False
########################################################################################################################

def simulation(force):
    x = force.numpy()[0]
    f = -0.001678*x**2 + 0.05034*x
    return f

"""
def simulation(force):
    force = force.numpy()[0]
    print("start simulation with force", force)
    command = shlex.split(f"./incompressible_mooney_rivlin ../settings_force.py incompressible_mooney_rivlin {force}")
    subprocess.run(command)

    print("end simulation")

    f = open("muscle_length_prestretch.csv")
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        prestretch = float(row[1]) - float(row[0])
        print("The muscle was stretched ", prestretch)
    f.close()

    f = open("muscle_length_contraction.csv")
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
        train_Yvar = torch.full_like(train_Y, fixed_Yvar, dtype=torch.double)
        likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
        if matern:
            kernel = ScaleKernel(MaternKernel(nu=nu))
        elif rbf:
            kernel = ScaleKernel(RBFKernel())
        else:
            print("Wrong input, used Matern Kernel with nu=1.5 instead")
            kernel = ScaleKernel(MaternKernel(nu=1.5))

        if const:
            mean = ConstantMean()
        elif zero:
            mean = ZeroMean()
        else:
            print("Wrong input, used Constant Mean instead")
            mean = ConstantMean()

        #input_transform = InputStandardize(d=1)
        #output_tansform = Standardize(m=1)

        super().__init__(train_X,
                         train_Y,
                         likelihood=likelihood,
                         covar_module=kernel,
                         mean_module=mean,
                         #input_transform=input_transform,
                         #outcome_transform=output_tansform,
                        )

starting_time = time.time()

os.chdir("..")
os.chdir("cuboid_muscle/build_release")

sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
if sobol_on:
    initial_x = sobol.draw(num_initial_trials, dtype=torch.double)
else:
    initial_x = torch.linspace(0, 1, num_initial_trials)

with open("BayesOpt_outputs.csv", "w"):
    pass

initial_y = torch.tensor([])
for force in initial_x:
    y = torch.tensor([[simulation(force*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)

    with open("BayesOpt_outputs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([force.numpy()[0]*(upper_bound-lower_bound)+lower_bound, y.numpy()[0,0]])

    initial_y = torch.cat([initial_y, y])
initial_yvar = torch.full_like(initial_y, fixed_Yvar, dtype=torch.double)


gp = CustomSingleTaskGP(initial_x, initial_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
if ML:
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"Failed to fit the model: {e}")
elif full_Bayes:
    pass
else:
    print("Wrong input, used Maximum Likelihood instead.")
    fit_gpytorch_mll(mll)

print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
print("Outputscale:", gp.covar_module.outputscale.item())
print("Noise:", gp.likelihood.noise.mean().item())

num_iterations = 100
best_value = -float('inf')
no_improvement_trials = 0
counter = num_initial_trials

for i in range(num_iterations):
    if EI:
        acq_fct = ExpectedImprovement(model=gp, best_f=initial_y.max())
    elif PI:
        acq_fct = ProbabilityOfImprovement(model=gp, best_f=initial_y.max())
    elif KG:
        acq_fct = qKnowledgeGradient(model=gp, num_fantasies=16)
    elif ES:
        grid_points = torch.linspace(0, 1, 1000)
        acq_fct = qMaxValueEntropy(model=gp, candidate_set=grid_points)
    else:
        print("Wrong input, used Expected Improvement instead.")
        acq_fct = ExpectedImprovement(model=gp, best_f=initial_y.max())

    if not max_stddev or counter%freq!=0:
        candidate, acq_value = optimize_acqf(
            acq_function=acq_fct,
            bounds=torch.tensor([[0], [1]], dtype=torch.double),
            q=1,
            num_restarts=20,
            raw_samples=256,
        )
    else:
        x_query = torch.linspace(0, 1, 1000).unsqueeze(-1)
        posterior = gp.posterior(x_query)
        variance = posterior.variance.squeeze(-1)
        stddev = torch.sqrt(variance).detach().numpy()
        argmax_stddev = stddev.argmax()
        candidate = torch.tensor([x_query[argmax_stddev].numpy()])
        print("max stddev used")

    new_y = torch.tensor([[simulation(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
    new_yvar = torch.full_like(new_y, fixed_Yvar, dtype=torch.double)

    with open("BayesOpt_outputs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy()[0,0]*(upper_bound-lower_bound)+lower_bound, new_y.numpy()[0,0]])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    initial_yvar = torch.cat([initial_yvar, new_yvar])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if ML:
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Failed to fit the model: {e}")
    elif full_Bayes:
        pass
    else:
        print("Wrong input, used Maximum Likelihood instead.")
        fit_gpytorch_mll(mll)

    x_query = torch.linspace(0, 1, 1000).unsqueeze(-1)
    posterior = gp.posterior(x_query)

    mean = posterior.mean.squeeze(-1).detach().numpy()
    variance = posterior.variance.squeeze(-1)
    stddev = torch.sqrt(variance).detach().numpy()

    print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
    print("Outputscale:", gp.covar_module.outputscale.item())
    print("Noise:", gp.likelihood.noise.mean().item())

    current_value = new_y.item()
    if current_value > best_value + improvement_threshold:
        best_value = current_value
        no_improvement_trials = 0
    elif len(initial_x) > num_initial_trials:
        no_improvement_trials += 1

    counter += 1

    print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.item()*(upper_bound-lower_bound)+lower_bound}, Value = {current_value}, Best Value = {best_value}")

    if no_improvement_trials >= num_consecutive_trials:
        print("Stopping criterion met. No significant improvement for consecutive trials.")
        print("Number of total trials: ", i+1+num_initial_trials)
        break

    if visualize:
        plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
        plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
        plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
        plt.scatter(candidate.numpy()*(upper_bound-lower_bound)+lower_bound, new_y.numpy(), color="green", s=30, zorder=5, label="New query point")
        plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
        plt.xlabel("x")
        plt.ylabel("Objective Value")
        plt.title("Optimization Process")
        plt.legend()
        plt.show()

x_query = torch.linspace(0, 1, 1000).unsqueeze(-1)
posterior = gp.posterior(x_query)

mean = posterior.mean.squeeze(-1).detach().numpy()
variance = posterior.variance.squeeze(-1)
stddev = torch.sqrt(variance).detach().numpy()

if visualize:
    plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
    plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
    plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
    plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
    plt.xlabel("x")
    plt.ylabel("Objective Value")
    plt.title("Optimization Results")
    plt.legend()
    plt.show()

if add_points:
    continuing = input("Do you want to add another query point? (y/n)")
else:
    continuing = "n"

while continuing == "y":
    candidate = input("Which point do you want to add?")
    candidate = torch.tensor([[float(candidate)]], dtype=torch.double)

    new_y = torch.tensor([[simulation(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
    new_yvar = torch.full_like(new_y, fixed_Yvar, dtype=torch.double)

    with open("BayesOpt_outputs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy()[0,0]*(upper_bound-lower_bound)+lower_bound, new_y.numpy()[0,0]])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    initial_yvar = torch.cat([initial_yvar, new_yvar])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if ML:
        fit_gpytorch_mll(mll)
    elif full_Bayes:
        pass
    else:
        print("Wrong input, used Maximum Likelihood instead.")
        fit_gpytorch_mll(mll)

    counter += 1

    print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.item()}, Value = {current_value}, Best Value = {best_value}")

    x_query = torch.linspace(0, 1, 1000).unsqueeze(-1)
    posterior = gp.posterior(x_query)

    mean = posterior.mean.squeeze(-1).detach().numpy()
    variance = posterior.variance.squeeze(-1)
    stddev = torch.sqrt(variance).detach().numpy()

    if visualize:
        plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
        plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
        plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
        plt.scatter(candidate.numpy()*(upper_bound-lower_bound)+lower_bound, new_y.numpy(), color="green", s=30, zorder=5, label="New query point")
        plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
        plt.xlabel("x")
        plt.ylabel("Objective Value")
        plt.title("Optimization Process")
        plt.legend()
        plt.show()

    continuing = input("Do you want to add another query point? (y/n)")

max_index = torch.argmax(initial_y)
maximizer = initial_x[max_index]
best_y = initial_y[max_index]

with open("BayesOpt_outputs.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(np.linspace(lower_bound, upper_bound, 1000))
    writer.writerow(mean)
    writer.writerow(stddev)
    writer.writerow([counter])
    writer.writerow([maximizer.numpy()[0]*(upper_bound-lower_bound)+lower_bound, best_y.numpy()[0]])
    writer.writerow([time.time()-starting_time])
