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
import signal
import setup_BayesOpt_cuboid_muscle


#If you want to call this file, you have two options:
#>python BayesOpt.py
#or
#>python BayesOpt.py matern 1.5 const fixed_noise ei stopping_xy
#You can change these inputs to any ones of it kind, see options below. A chosen option becomes True, every other option
#of this kind becomes False. You can leave any option out, then the current setup in here is being chosen. The order
#also doesn't matter.

########################################################################################################################
#Customize code here

absolute = setup_BayesOpt_cuboid_muscle.optimize_absolute_value
relative = setup_BayesOpt_cuboid_muscle.optimize_relative_value

end_lenth = setup_BayesOpt_cuboid_muscle.end_length

nu = setup_BayesOpt_cuboid_muscle.nu
matern = setup_BayesOpt_cuboid_muscle.matern
rbf = setup_BayesOpt_cuboid_muscle.rbf

const = setup_BayesOpt_cuboid_muscle.const
zero = setup_BayesOpt_cuboid_muscle.zero

fixed_noise = setup_BayesOpt_cuboid_muscle.fixed_noise
variable_noise = setup_BayesOpt_cuboid_muscle.variable_noise

EI = setup_BayesOpt_cuboid_muscle.EI
PI = setup_BayesOpt_cuboid_muscle.PI
KG = setup_BayesOpt_cuboid_muscle.KG
ES = setup_BayesOpt_cuboid_muscle.ES

stopping_y = setup_BayesOpt_cuboid_muscle.stopping_y
improvement_threshold = setup_BayesOpt_cuboid_muscle.improvement_threshold
stopping_xy = setup_BayesOpt_cuboid_muscle.stopping_xy
x_range = setup_BayesOpt_cuboid_muscle.x_range
num_consecutive_trials = setup_BayesOpt_cuboid_muscle.num_consecutive_trials

fixed_Yvar = setup_BayesOpt_cuboid_muscle.fixed_Yvar
lower_bound = setup_BayesOpt_cuboid_muscle.lower_bound
sobol_on = setup_BayesOpt_cuboid_muscle.sobol_on
num_initial_trials = setup_BayesOpt_cuboid_muscle.num_initial_trials
visualize = setup_BayesOpt_cuboid_muscle.visualize
add_points = setup_BayesOpt_cuboid_muscle.add_points
upper_bound = setup_BayesOpt_cuboid_muscle.upper_bound
specific_relative_upper_bound = setup_BayesOpt_cuboid_muscle.specific_relative_upper_bound
max_upper_bound = setup_BayesOpt_cuboid_muscle.max_upper_bound
relative_prestretch_min = setup_BayesOpt_cuboid_muscle.relative_prestretch_min
relative_prestretch_max = setup_BayesOpt_cuboid_muscle.relative_prestretch_max
########################################################################################################################

inputs = [item.lower() for item in sys.argv]
if len(inputs) > 0:
    if "matern" in inputs:
        matern = True
        if "0.5" in inputs:
            nu = 0.5
        elif "1.5" in inputs:
            nu = 1.5
        elif "2.5" in inputs:
            nu = 2.5
    elif "rbf" in inputs:
        matern = False
        rbf = True
    if "const" in inputs:
        const = True
    elif "zero" in inputs:
        const = False
        zero = True
    if "fixed_noise" in inputs:
        fixed_noise = True
    elif "variable_noise" in inputs:
        fixed_noise = False
        variable_noise = True
    if "ei" in inputs:
        EI = True
    elif "pi" in inputs:
        EI = False
        PI = True
    elif "kg" in inputs:
        EI = False
        PI = False
        KG = True
    elif "es" in inputs:
        EI = False
        PI = False
        KG = False
        ES = True
    if "stopping_y" in inputs:
        stopping_y = True
    elif "stopping_xy" in inputs:
        stopping_y = False
        stopping_xy = True



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
if absolute:
    global_individuality_parameter = global_individuality_parameter + "_absolute_contraction_length"
elif relative:
    global_individuality_parameter = global_individuality_parameter + "_relative_contraction_length"

starting_lengths = []
relative_contraction = []


def simulation(starting_length):
    starting_length = starting_length.numpy()[0]
    print("start simulation with starting_length", starting_length)
    command = shlex.split(f"./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py --stress-free {starting_length} --after-prestretch {end_lenth}")
    subprocess.run(command)

    print("end simulation")

    f = open("muscle_contraction_" + str(starting_length) + "_starting_length.csv")
    reader = csv.reader(f, delimiter=",")
    muscle_length_process = []
    for row in reader:
        for j in row:
            muscle_length_process.append(j)
    contraction = float(muscle_length_process[0]) - float(muscle_length_process[-2])
    print("The muscle contracted ", contraction)
    f.close()

    relative_contraction.append(contraction/starting_length)

    return contraction

def simulation_relative(starting_length):
    starting_length = starting_length.numpy()[0]
    print("start simulation with starting_length", starting_length)
    command = shlex.split(f"./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py --stress-free {starting_length} --after-prestretch {end_lenth}")
    subprocess.run(command)

    print("end simulation")

    f = open("muscle_contraction_" + str(starting_length) + "_starting_length.csv")
    reader = csv.reader(f, delimiter=",")
    muscle_length_process = []
    for row in reader:
        for j in row:
            muscle_length_process.append(j)
    contraction = float(muscle_length_process[0]) - float(muscle_length_process[-2])
    print("The muscle contracted ", contraction)
    f.close()

    return contraction/starting_length

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

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()



def find_relative_prestretch(starting_length):
    individuality_parameter = str(int(time.time()))+"_"+str(starting_length)
    command = shlex.split(f"./incompressible_mooney_rivlin_prestretch_only ../prestretch_tensile_test.py incompressible_mooney_rivlin_prestretch_only {starting_length} {individuality_parameter}")
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)

    try:
        subprocess.run(command)
        f = open("muscle_length_prestretch"+individuality_parameter+".csv")
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            relative_prestretch = float(row[1]) / float(row[0])
        f.close()
        command2 = shlex.split("rm muscle_length_prestretch"+individuality_parameter+".csv")
        subprocess.run(command2)
    except TimeoutException:
        relative_prestretch = -1
        print("Muscle tore")
    finally:
        signal.alarm(0)

    return relative_prestretch

def find_max_upper_bound():
    lower_guess = 0
    upper_guess = 10
    relative_prestretch_up = find_relative_prestretch(upper_guess)

    while relative_prestretch_up >= 0 or (relative_prestretch_up < 0 and upper_guess-lower_guess > 1):
        if relative_prestretch_up >= 0:
            not_relevant = upper_guess
            upper_guess = 2*upper_guess - lower_guess
            lower_guess = not_relevant
            relative_prestretch_up = find_relative_prestretch(upper_guess)
        else:
            middle_guess = (upper_guess+lower_guess)/2
            relative_prestretch_mid = find_relative_prestretch(middle_guess)
            if relative_prestretch_mid >= 0:
                lower_guess = middle_guess
            elif relative_prestretch_mid < 0:
                upper_guess = middle_guess
                relative_prestretch_up = relative_prestretch_mid

    return lower_guess
    
def find_specific_upper_bound():
    lower_guess = 0
    upper_guess = 10
    relative_prestretch_low = 0
    relative_prestretch_up = find_relative_prestretch(upper_guess)

    while (relative_prestretch_low < relative_prestretch_min or relative_prestretch_low > relative_prestretch_max) and (relative_prestretch_up < relative_prestretch_min or relative_prestretch_up > relative_prestretch_max):
        if relative_prestretch_up < relative_prestretch_min and relative_prestretch_up >= 0:
            not_relevant = upper_guess
            upper_guess = 2*upper_guess - lower_guess
            lower_guess = not_relevant
            relative_prestretch_low = relative_prestretch_up
            relative_prestretch_up = find_relative_prestretch(upper_guess)
        else:
            middle_guess = (upper_guess+lower_guess)/2
            relative_prestretch_mid = find_relative_prestretch(middle_guess)
            if relative_prestretch_mid < relative_prestretch_min:
                lower_guess = middle_guess
                relative_prestretch_low = relative_prestretch_mid
            elif relative_prestretch_mid > relative_prestretch_max:
                upper_guess = middle_guess
                relative_prestretch_up = relative_prestretch_mid
            else:
                return middle_guess
        
    return upper_guess

os.chdir("build_release")

if specific_relative_upper_bound:
    upper_bound = find_specific_upper_bound()
elif max_upper_bound:
    upper_bound = find_max_upper_bound()

starting_time = time.time()

sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
if sobol_on:
    initial_x = sobol.draw(num_initial_trials, dtype=torch.double)
else:
    initial_x = torch.linspace(0, 1, num_initial_trials, dtype=torch.double).unsqueeze(1)  

print(initial_x)

if relative and (not absolute):
    for starting_length in initial_x:
        starting_lengths.append(starting_length)
        y = torch.tensor([[simulation_relative(starting_length*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
        relative_contraction.append(y)

if absolute:
    initial_y = torch.tensor([])
    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "w") as f:
        pass
    for starting_length in initial_x:
        starting_lengths.append(starting_length)
        y = torch.tensor([[simulation(starting_length*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)

        with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([starting_length.numpy()[0]*(upper_bound-lower_bound)+lower_bound, y.numpy()[0,0]])

        initial_y = torch.cat([initial_y, y])
    initial_yvar = torch.full_like(initial_y, fixed_Yvar, dtype=torch.double)


    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
    #print("Outputscale:", gp.covar_module.outputscale.item())
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

        starting_lengths.append(candidate[0])
        new_y = torch.tensor([[simulation(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
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

        print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
        #print("Outputscale:", gp.covar_module.outputscale.item())
        print("Noise:", gp.likelihood.noise.mean().item())

        if visualize:
            plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
            plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
            plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
            plt.scatter(candidate.numpy()*(upper_bound-lower_bound)+lower_bound, new_y.numpy(), color="green", s=30, zorder=5, label="New query point")
            plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
            plt.xlabel("prestretch starting_length")
            plt.ylabel("muscle contraction length")
            plt.title("Optimization Process")
            plt.gcf().suptitle(title, fontsize=8)
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
        plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
        plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
        plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
        plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
        plt.xlabel("prestretch starting_length")
        plt.ylabel("muscle contraction length")
        plt.title("Optimization Results")
        plt.gcf().suptitle(title, fontsize=8)
        plt.legend()
        plt.show()

    if add_points:
        continuing = input("Do you want to add another query point? (y/n)")
    else:
        continuing = "n"

    while continuing == "y":
        candidate = input("Which point do you want to add?")
        candidate = torch.tensor([[float(candidate)]], dtype=torch.double)

        starting_lengths.append(candidate[0])

        new_y = torch.tensor([[simulation(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
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
            plt.xlabel("prestretch starting_length")
            plt.ylabel("muscle contraction length")
            plt.title("Optimization Process")
            plt.gcf().suptitle(title, fontsize=8)
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








# Same process for relative contraction length using the information from above

if relative:
    if absolute:
        global_individuality_parameter = global_individuality_parameter[:-28] + "_relative_contraction_length"
    

    initial_x = torch.tensor(starting_lengths).unsqueeze(1)
    initial_y = torch.tensor(relative_contraction).unsqueeze(1)
    print(initial_x)
    print(initial_y)

    with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "w") as f:
        pass

    for i in range(len(initial_x)):
        with open("BayesOpt_outputs"+global_individuality_parameter+".csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([initial_x[i].numpy()[0]*(upper_bound-lower_bound)+lower_bound, initial_y[i].numpy()[0]])

    initial_yvar = torch.full_like(initial_y, fixed_Yvar, dtype=torch.double)

    gp = CustomSingleTaskGP(initial_x, initial_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
    #print("Outputscale:", gp.covar_module.outputscale.item())
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

        new_y = torch.tensor([[simulation_relative(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
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

        print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
        #print("Outputscale:", gp.covar_module.outputscale.item())
        print("Noise:", gp.likelihood.noise.mean().item())

        if visualize:
            plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
            plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
            plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
            plt.scatter(candidate.numpy()*(upper_bound-lower_bound)+lower_bound, new_y.numpy(), color="green", s=30, zorder=5, label="New query point")
            plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
            plt.xlabel("prestretch starting_length")
            plt.ylabel("muscle contraction length")
            plt.title("Optimization Process")
            plt.gcf().suptitle(title, fontsize=8)
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
        plt.scatter(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", label="Trials", zorder=3)
        plt.plot(initial_x.numpy()*(upper_bound-lower_bound)+lower_bound, initial_y.numpy(), color="red", linestyle="", markersize=3)
        plt.plot(x_query*(upper_bound-lower_bound)+lower_bound, mean)
        plt.fill_between(x_query.numpy().squeeze()*(upper_bound-lower_bound)+lower_bound, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI")
        plt.xlabel("prestretch starting_length")
        plt.ylabel("muscle contraction length")
        plt.title("Optimization Results")
        plt.gcf().suptitle(title, fontsize=8)
        plt.legend()
        plt.show()

    if add_points:
        continuing = input("Do you want to add another query point? (y/n)")
    else:
        continuing = "n"

    while continuing == "y":
        candidate = input("Which point do you want to add?")
        candidate = torch.tensor([[float(candidate)]], dtype=torch.double)

        new_y = torch.tensor([[simulation_relative(candidate[0]*(upper_bound-lower_bound)+lower_bound)]], dtype=torch.double)
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
            plt.xlabel("prestretch starting_length")
            plt.ylabel("muscle contraction length")
            plt.title("Optimization Process")
            plt.gcf().suptitle(title, fontsize=8)
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
