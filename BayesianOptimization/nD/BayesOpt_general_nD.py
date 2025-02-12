import os
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
import setup_BayesOpt_general_nD

#TO DO

"""
This is a file to carry out Bayesian Optimization for a dummy function.
If you want to call this file, you have two options:
>python BayesOpt.py
or
>python BayesOpt.py matern 1.5 const fixed_noise ei stopping_xy 1
You can change these inputs to any ones of it kind, see options below. A chosen option becomes True, every other option
of this kind becomes False. You can leave any option out, then the current setup in here is being chosen. The order
also doesn't matter.
The options:
For the kernel: "matern 0.5" "matern 1.5" "matern 2.5" "rbf"
For the mean: "const" "zero"
For the noise: "fixed_noise" "variable_noise"
For the acquisition function: "ei" "es" "kg" "pi"
For the stopping criterion: "stopping_xy" "stopping_y"
"""

########################################################################################################################
#Customize code here

#Major changes:
dimension = setup_BayesOpt_general_nD.dimension

nu = setup_BayesOpt_general_nD.nu
matern = setup_BayesOpt_general_nD.matern
rbf = setup_BayesOpt_general_nD.rbf

const = setup_BayesOpt_general_nD.const
zero = setup_BayesOpt_general_nD.zero

fixed_noise = setup_BayesOpt_general_nD.fixed_noise
variable_noise = setup_BayesOpt_general_nD.variable_noise

EI = setup_BayesOpt_general_nD.EI
PI = setup_BayesOpt_general_nD.PI
KG = setup_BayesOpt_general_nD.KG
ES = setup_BayesOpt_general_nD.ES

stopping_y = setup_BayesOpt_general_nD.stopping_y
improvement_threshold = setup_BayesOpt_general_nD.improvement_threshold
stopping_xy = setup_BayesOpt_general_nD.stopping_xy
x_range = setup_BayesOpt_general_nD.x_range
num_consecutive_trials = setup_BayesOpt_general_nD.num_consecutive_trials

#Minor changes:
fixed_Yvar = setup_BayesOpt_general_nD.fixed_Yvar
bounds = setup_BayesOpt_general_nD.bounds
sobol_on = setup_BayesOpt_general_nD.sobol_on
num_initial_trials = setup_BayesOpt_general_nD.num_initial_trials
add_points = setup_BayesOpt_general_nD.add_points
########################################################################################################################

#We need to write the generated data into files. To see the difference between the resulting files, we add a individuality 
#parameter in the filename, which we create here. 
global_individuality_parameter = ""
title = ""
if matern:
    global_individuality_parameter = global_individuality_parameter + "_matern_" + str(nu)
    title = title + "Matern Kernel with nu=" + str(nu) + ", "
elif rbf:
    global_individuality_parameter = global_individuality_parameter + "_rbf"
    title = title + "RBF Kernel, "
if const:
    global_individuality_parameter = global_individuality_parameter + "_const"
    title = title + "Constant Mean, "
elif zero:
    global_individuality_parameter = global_individuality_parameter + "_zero"
    title = title + "Zero Mean, "
if fixed_noise:
    global_individuality_parameter = global_individuality_parameter + "_fixed_noise"
    title = title + "Fixed Noise, "
elif variable_noise:
    global_individuality_parameter = global_individuality_parameter + "_variable_noise"
    title = title + "Variable Noise, "
if EI:
    global_individuality_parameter = global_individuality_parameter + "_EI"
    title = title + "Expected Improvement, "
elif PI:
    global_individuality_parameter = global_individuality_parameter + "_PI"
    title = title + "Probability of Improvement, "
elif KG:
    global_individuality_parameter = global_individuality_parameter + "_KG"
    title = title + "Knoledge Gradient, "
elif ES:
    global_individuality_parameter = global_individuality_parameter + "_ES"
    title = title + "Entropy Search, "
if stopping_y:
    global_individuality_parameter = global_individuality_parameter + "_stopping_y"
    title = title + "Y-Stopping"
elif stopping_xy:
    global_individuality_parameter = global_individuality_parameter + "_stopping_xy"
    title = title + "XY-Stopping"


#The BO needs a Gaussian Process as statistical model, which is being created here.
class CustomSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        train_Yvar = torch.full_like(train_Y, fixed_Yvar, dtype=torch.double)
        if fixed_noise:
            likelihood = GaussianLikelihood(noise=train_Yvar)
        elif variable_noise:
            likelihood = GaussianLikelihood()
        else:
            print("Wrong input, used variable noise instead")
            likelihood = GaussianLikelihood()
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

try:
    os.chdir("build_release")
except:
    os.mkdir("build_release")
    os.chdir("build_release")

starting_time = time.time()

#Chooses the initial query points for BO and evaluates them
sobol = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True)
if sobol_on:
    initial_x = sobol.draw(num_initial_trials, dtype=torch.double)
else:
    axes = [torch.linspace(0.0, 1.0, num_initial_trials, dtype=torch.double) for _ in range(dimension)]
    grids = torch.meshgrid(*axes, indexing='ij')
    initial_x = torch.stack(grids, dim=-1).reshape(-1, dimension)
    
with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "w"):
    pass

initial_y = torch.tensor([])
for input in initial_x:
    y = torch.tensor([[setup_BayesOpt_general_nD.target_function(input*(bounds[1]-bounds[0])+bounds[0])]], dtype=torch.double)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([input.numpy()[0]*(bounds[1]-bounds[0])+bounds[0], y.numpy()[0,0]])

    initial_y = torch.cat([initial_y, y])
initial_yvar = torch.full_like(initial_y, fixed_Yvar, dtype=torch.double)


initial_x_vals = initial_x.clone()
initial_y_vals = initial_y.clone()

#Initializes the GP and calculates its posterior distribution
gp = CustomSingleTaskGP(initial_x, initial_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)


#This starts the optimization loop. It is being carried out 100 times, unless the stopping criterion is being triggered.
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
        pass
    elif ES:
        bounds_es = torch.stack([torch.zeros(dimension, dtype=torch.double), torch.ones(dimension, dtype=torch.double)])
        candidate_set = torch.rand(1000, bounds_es.size(1))
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        acq_fct = qMaxValueEntropy(model=gp, candidate_set=candidate_set)
    else:
        print("Wrong input, used Expected Improvement instead.")
        acq_fct = ExpectedImprovement(model=gp, best_f=initial_y.max())

    if KG:
        SMOKE_TEST = os.environ.get("SMOKE_TEST")
        NUM_FANTASIES = 128 if not SMOKE_TEST else 4
        NUM_RESTARTS = 10 if not SMOKE_TEST else 2
        RAW_SAMPLES = 128
        bounds_kg = torch.stack([torch.zeros(dimension, dtype=torch.double), torch.ones(dimension, dtype=torch.double)])
        acq_fct = qKnowledgeGradient(model=gp, num_fantasies=NUM_FANTASIES)
        candidates, acq_value = optimize_acqf(
            acq_function=acq_fct,
            bounds=bounds_kg,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        argmax_pmean, max_pmean = optimize_acqf(
            acq_function=PosteriorMean(gp),
            bounds=bounds_kg,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        qKG_proper = qKnowledgeGradient(
            gp,
            num_fantasies=NUM_FANTASIES,
            sampler=acq_fct.sampler,
            current_value=max_pmean,
        )

        candidate, acq_value_proper = optimize_acqf(
            acq_function=qKG_proper,
            bounds=bounds_kg,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
    else:
        candidate, acq_value = optimize_acqf(
            acq_function=acq_fct,
            bounds=torch.stack([torch.zeros(dimension, dtype=torch.double), torch.ones(dimension, dtype=torch.double)]),
            q=1,
            num_restarts=20,
            raw_samples=256,
        )

    new_y = torch.tensor([[setup_BayesOpt_general_nD.target_function(candidate[0]*(bounds[1]-bounds[0])+bounds[0])]], dtype=torch.double)
    new_yvar = torch.full_like(new_y, fixed_Yvar, dtype=torch.double)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy()[0,0]*(bounds[1]-bounds[0])+bounds[0], new_y.numpy()[0,0]])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    initial_yvar = torch.cat([initial_yvar, new_yvar])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    counter += 1

    if stopping_y:
        current_value = new_y.item()
        if current_value > best_value + improvement_threshold:
            best_value = current_value
            no_improvement_trials = 0
        elif len(initial_x) > num_initial_trials:
            no_improvement_trials += 1
        if no_improvement_trials >= num_consecutive_trials:
            scaled_candidate = candidate * (bounds[1] - bounds[0]) + bounds[0]
            print(f"Trial {i + 1 + num_initial_trials}: x = {scaled_candidate.numpy()}, Value = {current_value}, Best Value = {best_value}")
            print("Stopping criterion met. No significant improvement for consecutive trials.")
            print(global_individuality_parameter)
            print("Number of total trials: ", i+1+num_initial_trials)
            break
    elif stopping_xy:
        max_index = torch.argmax(initial_y)
        for k in range(len(initial_x)):
            number_x_in_epsilon_neighborhood = 0
            breaking = False
            max_y_in_range = False
            for j in range(len(initial_x)):
                if np.abs(initial_x[k,0].numpy() - initial_x[j,0].numpy()) < x_range:
                    number_x_in_epsilon_neighborhood += 1
                    if initial_x[max_index,0].numpy() == initial_x[k,0].numpy() or initial_x[max_index,0].numpy() == initial_x[j,0].numpy():
                        max_y_in_range = True
            if number_x_in_epsilon_neighborhood >= num_consecutive_trials and max_y_in_range:
                print("Stopping criterion met. No significant improvement for consecutive trials.")
                print("Number of total trials: ", i+1+num_initial_trials)
                breaking = True
                break
        if breaking:
            break
    else:
        print("Wrong input, used stopping_y instead.")
        stopping_y = True

    current_value = new_y.item()
    if current_value > best_value + improvement_threshold:
        best_value = current_value

    scaled_candidate = candidate * (bounds[1] - bounds[0]) + bounds[0]
    print(f"Trial {i + 1 + num_initial_trials}: x = {scaled_candidate.numpy()}, Value = {current_value}, Best Value = {best_value}")


if add_points:
    continuing = input("Do you want to add another query point? (y/n)")
else:
    continuing = "n"

#In case you want to add another point:
while continuing == "y":
    candidate = input("Which point do you want to add?")
    candidate = torch.tensor([[float(candidate)]], dtype=torch.double)

    new_y = torch.tensor([[setup_BayesOpt_general_nD.target_function(candidate[0]*(bounds[1]-bounds[0])+bounds[0])]], dtype=torch.double)
    new_yvar = torch.full_like(new_y, fixed_Yvar, dtype=torch.double)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy()[0,0]*(bounds[1]-bounds[0])+bounds[0], new_y.numpy()[0,0]])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    initial_yvar = torch.cat([initial_yvar, new_yvar])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    counter += 1

    print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.item()}, Value = {current_value}, Best Value = {best_value}")

    continuing = input("Do you want to add another query point? (y/n)")

max_index = torch.argmax(initial_y)
maximizer = initial_x[max_index]
best_y = initial_y[max_index]

with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Number of trials",counter])
    writer.writerow(["Optimizer and Optimum",(maximizer*(bounds[1]-bounds[0])+bounds[0]).numpy(), best_y.numpy()[0]])
    writer.writerow(["Time",time.time()-starting_time])

print(global_individuality_parameter)

with open("BayesOpt_global_individuality_parameters.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([global_individuality_parameter])
