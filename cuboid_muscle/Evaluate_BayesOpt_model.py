import subprocess
import shlex
import csv

number_of_iterations = 5

number_of_trials = 0
best_f = 0
time = 0

for i in range(number_of_iterations):
    subprocess.run(shlex.split("python3 BayesOpt.py"))
    
    with open("build_release/BayesOpt_outputs.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        rows = [row for row in reader]

    number_of_trials += float(rows[-3][0])
    best_f += float(rows[-2][1])
    time += float(rows[-1][0])

print("Average number of trials: ", number_of_trials/number_of_iterations)
print("Average maximum: ", best_f/number_of_iterations)
print("Average time elapsed: ", time/number_of_iterations, " seconds")