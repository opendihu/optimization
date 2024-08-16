import csv
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir("build_release")

with open("BayesOpt_outputs.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    rows = [row for row in reader]

for i in range(len(rows) - 3):
    plt.scatter(float(rows[i][0]), float(rows[i][1]), color="red", label="Trials" if i == 0 else "", zorder=3)

x = list(map(float, rows[-3]))
mean = list(map(float, rows[-2]))
stddev = list(map(float, rows[-1]))

x = np.array(x)
mean = np.array(mean)
stddev = np.array(stddev)

plt.plot(x, mean, color="blue", label="GP Mean")
plt.fill_between(x, mean - 2 * stddev, mean + 2 * stddev, alpha=0.3, label="GP 95% CI", color="lightblue")
plt.xlabel("x")
plt.ylabel("Objective Value")
plt.title("Optimization Results")
plt.legend()
plt.show()
