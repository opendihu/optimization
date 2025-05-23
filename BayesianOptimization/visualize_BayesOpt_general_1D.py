import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
When calling this file: python visualise_BayesOpt.py {individuality_parameter}
with individuality_parameter being parameter printed by BayesOpt file in the end.
It is the parameter in the filename in build_releas: "BayesOpt_outputs{individuality_parameter}.csv".
"""

individuality_parameter = sys.argv[1]

####################################################################
#Customize visualization file here:

relative_path_to_data = "build_release"
data_file_name = "BayesOpt_outputs"+individuality_parameter+".csv"
x_label = "x"
y_label = "y"
####################################################################

if len(sys.argv) == 5:
    relative_path_to_data = sys.argv[1]
    data_file_name = sys.argv[2]
    x_label = sys.argv[3]
    y_label = sys.argv[4]

os.chdir(relative_path_to_data)

with open(data_file_name, "r") as f:
    reader = csv.reader(f, delimiter=",")
    rows = [row for row in reader]

for i in range(len(rows) - 6):
    plt.scatter(float(rows[i][0]), float(rows[i][1]), color="red", label="Trials" if i == 0 else "", zorder=3)

x = list(map(float, rows[-6]))
mean = list(map(float, rows[-5]))
stddev = list(map(float, rows[-4]))
number_of_trials = float(rows[-3][0])
maximizer = float(rows[-2][0])
best_f = float(rows[-2][1])
time_elapsed = float(rows[-1][0])

print("Time elapsed: ", time_elapsed, " seconds")
print("Number of trials: ", number_of_trials)
print("Best value: ", best_f)
print("Maximizer: ", maximizer)

x = np.array(x)
mean = np.array(mean)
stddev = np.array(stddev)

plt.scatter(maximizer, best_f, color="green", label="Maximum", zorder=3)
plt.plot(x, mean, color="blue", label="GP Mean")
plt.fill_between(x, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI", color="lightblue")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title("Optimization Results")
plt.legend()
plt.show()
