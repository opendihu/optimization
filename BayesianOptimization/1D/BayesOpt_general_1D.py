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
import setup_BayesOpt_general_1D


"""
This is a file to carry out Bayesian Optimization for a dummy function.
To call this file, run:
>python BayesOpt_general_1D.py
To customize the optimization process and the target function, change settings in the file "setup_BayesOpt_general_1D.py".
"""

nu = setup_BayesOpt_general_1D.nu
matern = setup_BayesOpt_general_1D.matern
rbf = setup_BayesOpt_general_1D.rbf

const = setup_BayesOpt_general_1D.const
zero = setup_BayesOpt_general_1D.zero

fixed_noise = setup_BayesOpt_general_1D.fixed_noise
variable_noise = setup_BayesOpt_general_1D.variable_noise

EI = setup_BayesOpt_general_1D.EI
PI = setup_BayesOpt_general_1D.PI
KG = setup_BayesOpt_general_1D.KG
ES = setup_BayesOpt_general_1D.ES

stopping_y = setup_BayesOpt_general_1D.stopping_y
improvement_threshold = setup_BayesOpt_general_1D.improvement_threshold
stopping_xy = setup_BayesOpt_general_1D.stopping_xy
x_range = setup_BayesOpt_general_1D.x_range
num_consecutive_trials = setup_BayesOpt_general_1D.num_consecutive_trials

fixed_Yvar = setup_BayesOpt_general_1D.fixed_Yvar
lower_bound = setup_BayesOpt_general_1D.lower_bound
sobol_on = setup_BayesOpt_general_1D.sobol_on
num_initial_trials = setup_BayesOpt_general_1D.num_initial_trials
visualize = setup_BayesOpt_general_1D.visualize
add_points = setup_BayesOpt_general_1D.add_points
upper_bound = setup_BayesOpt_general_1D.upper_bound

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
current_time = time.strftime("%H%M%S")
global_individuality_parameter = global_individuality_parameter + "_" + str(current_time)

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
sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
if sobol_on:
    initial_x = sobol.draw(num_initial_trials, dtype=torch.double)
else:
    initial_x = torch.linspace(0, 1, num_initial_trials, dtype=torch.double).unsqueeze(1)
    
with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "w"):
    pass

initial_y = torch.tensor([])
for input in initial_x:
    y = torch.tensor([[setup_BayesOpt_general_1D.target_function(input*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([input.numpy()[0]*(upper_bound-lower_bound)+lower_bound, y.numpy()[0,0]])

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
        bounds=torch.tensor([[0], [1]], dtype=torch.double)
        candidate_set = torch.rand(1000, bounds.size(1))
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
        bounds = torch.stack([torch.zeros(1, dtype=torch.double), torch.ones(1, dtype=torch.double)])
        acq_fct = qKnowledgeGradient(model=gp, num_fantasies=NUM_FANTASIES)
        candidates, acq_value = optimize_acqf(
            acq_function=acq_fct,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        argmax_pmean, max_pmean = optimize_acqf(
            acq_function=PosteriorMean(gp),
            bounds=bounds,
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
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
    else:
        candidate, acq_value = optimize_acqf(
            acq_function=acq_fct,
            bounds=torch.tensor([[0], [1]], dtype=torch.double),
            q=1,
            num_restarts=20,
            raw_samples=256,
        )

    new_y = torch.tensor([[setup_BayesOpt_general_1D.target_function(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
    new_yvar = torch.full_like(new_y, fixed_Yvar, dtype=torch.double)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy()[0,0]*(upper_bound-lower_bound)+lower_bound, new_y.numpy()[0,0]])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    initial_yvar = torch.cat([initial_yvar, new_yvar])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

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
        plt.scatter(initial_x_vals.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y_vals.numpy(), color="orange", label="Initial values", zorder=3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Optimization Process")
        plt.gcf().suptitle(title, fontsize=12)
        plt.legend()
        plt.show()

    counter += 1

    if stopping_y:
        current_value = new_y.item()
        if current_value > best_value + improvement_threshold:
            best_value = current_value
            no_improvement_trials = 0
        elif len(initial_x) > num_initial_trials:
            no_improvement_trials += 1
        if no_improvement_trials >= num_consecutive_trials:
            print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.item()*(upper_bound-lower_bound)+lower_bound}, Value = {current_value}, Best Value = {best_value}")
            print("Stopping criterion met. No significant improvement for consecutive trials.")
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

    print(f"Trial {i + 1 + num_initial_trials}: x = {candidate.item()*(upper_bound-lower_bound)+lower_bound}, Value = {current_value}, Best Value = {best_value}")

    
x_query = torch.linspace(0, 1, 1000).unsqueeze(-1)
posterior = gp.posterior(x_query)

mean = posterior.mean.squeeze(-1).detach().numpy()
variance = posterior.variance.squeeze(-1)
stddev = torch.sqrt(variance).detach().numpy()

if visualize:
    max_index = torch.argmax(initial_y)
    max_x = initial_x[max_index]
    max_y = initial_y[max_index]
    plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
    plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
    plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
    plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
    plt.scatter(initial_x_vals.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y_vals.numpy(), color="orange", label="Initial values", zorder=3)
    plt.scatter(max_x.numpy()*(upper_bound-lower_bound)+lower_bound, max_y.numpy(), color="green", label="Maximum", zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Optimization Results")
    plt.gcf().suptitle(title, fontsize=12)
    plt.legend()
    plt.show()

if add_points:
    continuing = input("Do you want to add another query point? (y/n)")
else:
    continuing = "n"

#In case you want to add another point:
while continuing == "y":
    candidate = input("Which point do you want to add?")
    candidate = torch.tensor([[float(candidate)]], dtype=torch.double)

    new_y = torch.tensor([[setup_BayesOpt_general_1D.target_function(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
    new_yvar = torch.full_like(new_y, fixed_Yvar, dtype=torch.double)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([candidate.numpy()[0,0]*(upper_bound-lower_bound)+lower_bound, new_y.numpy()[0,0]])

    initial_x = torch.cat([initial_x, candidate])
    initial_y = torch.cat([initial_y, new_y])
    initial_yvar = torch.cat([initial_yvar, new_yvar])
    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
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
        plt.scatter(initial_x_vals.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y_vals.numpy(), color="orange", label="Initial values", zorder=3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Optimization Process")
        plt.gcf().suptitle(title, fontsize=12)
        plt.legend()
        plt.show()

    continuing = input("Do you want to add another query point? (y/n)")

max_index = torch.argmax(initial_y)
maximizer = initial_x[max_index]
best_y = initial_y[max_index]

with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(np.linspace(lower_bound, upper_bound, 1000))
    writer.writerow(mean)
    writer.writerow(stddev)
    writer.writerow([counter])
    writer.writerow([maximizer.numpy()[0]*(upper_bound-lower_bound)+lower_bound, best_y.numpy()[0]])
    writer.writerow([time.time()-starting_time])

print(global_individuality_parameter)

with open("BayesOpt_global_individuality_parameters.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([global_individuality_parameter])
