import subprocess
import shlex
import csv
import sys
import numpy as np

"""
This evaluates a BO model by trying it on the test functions from BayesOpt_test_functions.py. 
To evaluate a certain model, call this file as: >Evaluate_BayesOpt_model.py matern 0.5 const es stopping_xy fixed_noise
This does 100 optimization processes of each test function, averages the result and saves it in build_release/BayesOpt_evaluations.csv.
"""

for i in range(9):
    number_of_iterations = 100

    number_of_trials = 0
    maximizers = []
    percentage_local_maxima_found = 0
    percentage_global_maxima_found = 0
    best_f = []
    time = 0

    inputs = sys.argv
    input_string = ""
    for item in inputs:
        input_string = input_string + item + " "
    input_string = input_string + str(i+1)

    for j in range(number_of_iterations):
        subprocess.run(shlex.split("python3 BayesOpt_test_functions.py "+ input_string))

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
            if np.abs(float(rows[-2][0]) - 0.65) < 3e-2 and np.abs(float(rows[-2][1]) - 1.5675) < 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
        elif i+1 == 2:
            if np.abs(float(rows[-2][0]) - 0.6) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 0.7333) < 3e-2 and np.abs(float(rows[-2][1]) - 0.8413)< 3e-2:
                percentage_local_maxima_found += 1
        elif i+1 == 3:
            if np.abs(float(rows[-2][0]) - 0.8471) < 3e-2 and np.abs(float(rows[-2][1]) - 1.0673)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
        elif i+1 == 4:
            if np.abs(float(rows[-2][0]) - 0.2) < 3e-2 and np.abs(float(rows[-2][1]) - 1.4019)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 0.0) < 3e-2 and np.abs(float(rows[-2][1]) - 1.0456)< 3e-2:
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 0.6) < 3e-2 and np.abs(float(rows[-2][1]) - 1.0270)< 3e-2:
                percentage_local_maxima_found += 1
        elif i+1 == 5:
            if np.abs(float(rows[-2][0]) - 0.3591) < 3e-2 and np.abs(float(rows[-2][1]) - 1.1731)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 1) < 3e-2 and np.abs(float(rows[-2][1]) - 0.5)< 3e-2:
                percentage_local_maxima_found += 1
        elif i+1 == 6:
            if np.abs(float(rows[-2][0]) - 0.3143) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 0.943) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 3e-2:
                percentage_local_maxima_found += 1
                percentage_global_maxima_found += 1
        elif i+1 == 7:
            if np.abs(float(rows[-2][0]) - 0.8028) < 3e-2 and np.abs(float(rows[-2][1]) - 1.1093)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 1) < 3e-2 and np.abs(float(rows[-2][1]) - 0.624)< 3e-2:
                percentage_local_maxima_found += 1
            if np.abs(float(rows[-2][0]) - 0.477) < 3e-2 and np.abs(float(rows[-2][1]) - 0.561)< 3e-2:
                percentage_local_maxima_found += 1
        elif i+1 == 8:
            if np.abs(float(rows[-2][0]) - 0.5) < 3e-2 and np.abs(float(rows[-2][1]) - 1)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1
        elif i+1 == 9:
            if np.abs(float(rows[-2][0]) - 0.5916) < 3e-2 and np.abs(float(rows[-2][1]) - 0.6393)< 3e-2:
                percentage_global_maxima_found += 1
                percentage_local_maxima_found += 1

    #print("Average number of trials: ", number_of_trials/number_of_iterations)
    #print("Average maximizer: ", maximizer/number_of_iterations)
    #print("Average maximum: ", best_f/number_of_iterations)
    #print("Average time elapsed: ", time/number_of_iterations, " seconds")

    with open("build_release/BayesOpt_evaluations.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["Inputs: ", input_string])
        writer.writerow(["Average number of trials: ", number_of_trials/number_of_iterations])
        writer.writerow(["Maximizers: ", maximizers])
        writer.writerow(["Maxima: ", best_f])
        writer.writerow(["Average time elapsed: ", time/number_of_iterations, " seconds"])
        writer.writerow(["Percentage of local maxima found: ", 100*percentage_local_maxima_found/number_of_iterations])
        writer.writerow(["Percentage of global maxima found: ", 100*percentage_global_maxima_found/number_of_iterations])
