import subprocess
import shlex
import csv
import sys

number_of_iterations = 5

number_of_trials = 0
maximizer = 0
best_f = 0
time = 0

inputs = sys.argv
input_string = ""
for item in inputs:
    input_string = input_string + item + " "

for i in range(number_of_iterations):
    subprocess.run(shlex.split("python3 BayesOpt.py "+ input_string))

    with open("build_release/BayesOpt_global_individuality_parameters.csv", "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        individuality_parameter = str(rows[-1][0])
    
    with open("build_release/BayesOpt_outputs"+individuality_parameter+".csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        rows = [row for row in reader]

    number_of_trials += float(rows[-3][0])
    maximizer += float(rows[-2][0])
    best_f += float(rows[-2][1])
    time += float(rows[-1][0])

print("Average number of trials: ", number_of_trials/number_of_iterations)
print("Average maximizer: ", maximizer/number_of_iterations)
print("Average maximum: ", best_f/number_of_iterations)
print("Average time elapsed: ", time/number_of_iterations, " seconds")

with open("build_release/BayesOpt_evaluations.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(["Inputs: ", input_string])
    writer.writerow(["Average number of trials: ", number_of_trials/number_of_iterations])
    writer.writerow(["Average maximizer: ", maximizer/number_of_iterations])
    writer.writerow(["Average maximum: ", best_f/number_of_iterations])
    writer.writerow(["Average time elapsed: ", time/number_of_iterations, " seconds"])
