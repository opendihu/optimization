import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

parameter = sys.argv[1]

lengths_muscle_1 = []
lengths_muscle_2 = []
lengths_muscle0_1 = []
lengths_muscle0_2 = []



f = open("build_release/muscle_length_contraction"+parameter+"_1.csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle_1.append(float(i))
f.close()

f = open("build_release/muscle_length_contraction0"+parameter+"_1.csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle0_1.append(float(i))
f.close()


f = open("build_release/muscle_length_contraction"+parameter+"_2.csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle_2.append(float(i))
f.close()

f = open("build_release/muscle_length_contraction0"+parameter+"_2.csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle0_2.append(float(i))
f.close()

x = np.linspace(0,100,len(lengths_muscle_1))

plt.plot(x, [x - lengths_muscle_1[0] for x in lengths_muscle_1], "b--",label="muscle 1(prestretch)")
plt.plot(x, [x - lengths_muscle_2[0] for x in lengths_muscle_2], "y--",label="muscle 2 (prestretch)")
plt.plot(x, [x - lengths_muscle0_1[0] for x in lengths_muscle0_1], "b",label="muscle 1")
plt.plot(x, [x - lengths_muscle0_2[0] for x in lengths_muscle0_2], "y",label="muscle 1")
plt.xlabel("time (ms)")
plt.ylabel("length of muscles (cm)")
plt.legend()
plt.show()