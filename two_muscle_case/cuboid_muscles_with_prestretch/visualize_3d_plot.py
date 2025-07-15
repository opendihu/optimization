import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Wichtig für 3D
import csv
import sys
import ast

data_file_name = sys.argv[1]
data_file_name = "build_release/BayesOpt_outputs"+data_file_name+".csv"

with open(data_file_name, "r") as f:
    reader = csv.reader(f, delimiter=",")
    rows = [row for row in reader]

#for i in range(len(rows) - 6):
#    plt.scatter(float(rows[i][0]), float(rows[i][1]), color="red", label="Trials" if i == 0 else "", zorder=3)

XY = rows[-5]
XY = [list(map(float, row.strip("[]").split())) for row in XY]
XY = np.array(XY, dtype=float)

mean = list(map(float, rows[-4]))
number_of_trials = float(rows[-3][1])
maximizer = float(rows[-2][1])
best_f = float(rows[-2][1])
time_elapsed = float(rows[-1][1])

x_unique = np.unique(XY[:, 0])
y_unique = np.unique(XY[:, 1])
nx = len(x_unique)
ny = len(y_unique)
mean = np.array(mean, dtype=float).reshape(ny, nx)
X_grid = XY[:, 0].reshape(ny, nx)
Y_grid = XY[:, 1].reshape(ny, nx)


# 4. Plot erstellen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, mean, cmap='viridis')

# 5. Optional: Achsentitel
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.show()
