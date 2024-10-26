import subprocess
import shlex
import csv
import sys
import numpy as np

"""
This evaluates BO models by trying it on the test functions from BayesOpt_test_functions.py. 
It evaluates all possible combinations of the three lists means, kernels and acqu_fcts.
This does 100 optimization processes of each test function, averages the result and saves it in build_release/BayesOpt_evaluations.csv.
"""

means = ["const", "zero"]
kernels = ["matern 0.5", "matern 1.5", "matern 2.5", "rbf"]
acqu_fcts = ["ei", "es"]

for mean in means:
    for kernel in kernels:
        for acqu_fct in acqu_fcts:
            avg_number_trials = 0
            avg_perc_local_maxima = 0
            avg_perc_global_maxima = 0

            for i in range(9):
                number_of_iterations = 100

                number_of_trials = 0
                maximizers = []
                percentage_local_maxima_found = 0
                percentage_global_maxima_found = 0
                best_f = []
                time = 0

                input_string = mean + " " + kernel + " " + acqu_fct + " stopping_xy fixed_noise " + str(i+1)

                for j in range(number_of_iterations):
                    subprocess.run(shlex.split("python3 BayesOpt.py "+ input_string))

                    with open("build_release/BayesOpt_global_individuality_parameters.csv", "r") as f:
                        reader = csv.reader(f)
                        rows = [row for row in reader]
                        individuality_parameter = str(rows[-1][0])
                    
                    with open("build_release/BayesOpt_outputs"+individuality_parameter+".csv", "r") as f:
                        reader = csv.reader(f, delimiter=",")
                        rows = [row for row in reader]

                    number_of_trials += float(rows[-3][0])
                    maximizers.append(float(rows[-2][0]))
                    best_f.append(float(rows[-2][1]))
                    time += float(rows[-1][0])

                    if i+1 == 1:
                        if np.abs(float(rows[-2][0]) - 0.65) < 3e-2 and np.abs(float(rows[-2][1]) - 1.5675) < 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                    elif i+1 == 2:
                        if np.abs(float(rows[-2][0]) - 0.6) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 0.7333) < 3e-2 and np.abs(float(rows[-2][1]) - 0.8413)< 1e-2:
                            percentage_local_maxima_found += 1
                    elif i+1 == 3:
                        if np.abs(float(rows[-2][0]) - 0.8471) < 3e-2 and np.abs(float(rows[-2][1]) - 1.0673)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                    elif i+1 == 4:
                        if np.abs(float(rows[-2][0]) - 0.2) < 3e-2 and np.abs(float(rows[-2][1]) - 1.4019)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 0.0) < 3e-2 and np.abs(float(rows[-2][1]) - 1.0456)< 1e-2:
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 0.6) < 3e-2 and np.abs(float(rows[-2][1]) - 1.0270)< 1e-2:
                            percentage_local_maxima_found += 1
                    elif i+1 == 5:
                        if np.abs(float(rows[-2][0]) - 0.3591) < 3e-2 and np.abs(float(rows[-2][1]) - 1.1731)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 1) < 3e-2 and np.abs(float(rows[-2][1]) - 0.5)< 1e-2:
                            percentage_local_maxima_found += 1
                    elif i+1 == 6:
                        if np.abs(float(rows[-2][0]) - 0.3143) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 0.943) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 1e-2:
                            percentage_local_maxima_found += 1
                            percentage_global_maxima_found += 1
                    elif i+1 == 7:
                        if np.abs(float(rows[-2][0]) - 0.8028) < 3e-2 and np.abs(float(rows[-2][1]) - 1.1093)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 1) < 3e-2 and np.abs(float(rows[-2][1]) - 0.624)< 1e-2:
                            percentage_local_maxima_found += 1
                        if np.abs(float(rows[-2][0]) - 0.477) < 3e-2 and np.abs(float(rows[-2][1]) - 0.561)< 1e-2:
                            percentage_local_maxima_found += 1
                    elif i+1 == 8:
                        if np.abs(float(rows[-2][0]) - 0.5) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1
                    elif i+1 == 9:
                        if np.abs(float(rows[-2][0]) - 0.5916) < 3e-2 and np.abs(float(rows[-2][1]) - 0.6393)< 1e-2:
                            percentage_global_maxima_found += 1
                            percentage_local_maxima_found += 1

                with open("build_release/BayesOpt_evaluations_detailed.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Inputs: ", input_string])
                    writer.writerow(["Average number of trials: ", number_of_trials/number_of_iterations])
                    writer.writerow(["Percentage of local maxima found: ", 100*percentage_local_maxima_found/number_of_iterations])
                    writer.writerow(["Percentage of global maxima found: ", 100*percentage_global_maxima_found/number_of_iterations])

                avg_perc_local_maxima += percentage_local_maxima_found/(5*number_of_iterations)
                avg_perc_global_maxima += percentage_global_maxima_found/(5*number_of_iterations)
                avg_number_trials += number_of_trials/(5*number_of_iterations)

            with open("build_release/BayesOpt_evaluations.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(["Inputs: ", mean + " " + kernel + " " + acqu_fct + " stopping_xy fixed_noise"])
                writer.writerow(["Average number of trials: ", avg_number_trials])
                writer.writerow(["Percentage of local maxima found: ", 100*avg_perc_local_maxima])
                writer.writerow(["Percentage of global maxima found: ", 100*avg_perc_global_maxima])
