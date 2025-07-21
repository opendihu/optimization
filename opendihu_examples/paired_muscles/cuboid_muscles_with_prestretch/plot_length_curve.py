import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

parameter = sys.argv[1]
if len(sys.argv)>2:
    parameter2 = sys.argv[2]

lengths_muscle_1 = []
lengths_muscle_2 = []

lengths_muscle_3 = []
lengths_muscle_4 = []

f = open("build_release/muscle_length_contraction_1_"+parameter+".csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle_1.append(float(i))
f.close()
lengths_muscle_1 -= lengths_muscle_1[0] * np.ones(len(lengths_muscle_1))

f = open("build_release/muscle_length_contraction_2_"+parameter+".csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle_2.append(float(i))
f.close()
lengths_muscle_2 -= lengths_muscle_2[0] * np.ones(len(lengths_muscle_2))

x = np.linspace(0,50,len(lengths_muscle_1))

plt.plot(x, lengths_muscle_1, color="orange", label="muscle 1 without prestretch")
plt.plot(x, lengths_muscle_2, color="b", label="muscle 2 without prestretch")
if len(sys.argv)>2:
    f = open("build_release/muscle_length_contraction_1_"+parameter2+".csv", "r")
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        for i in row:
            if i != "":
                lengths_muscle_3.append(float(i))
    f.close()
    lengths_muscle_3 -= lengths_muscle_3[0] * np.ones(len(lengths_muscle_3))

    f = open("build_release/muscle_length_contraction_2_"+parameter2+".csv", "r")
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        for i in row:
            if i != "":
                lengths_muscle_4.append(float(i))
    f.close()
    lengths_muscle_4 -= lengths_muscle_4[0] * np.ones(len(lengths_muscle_4))
    plt.plot(x, lengths_muscle_3, color="orange", ls="--", label="muscle 1 with prestretch")
    plt.plot(x, lengths_muscle_4, color="b", ls="--", label="muscle 2 with prestretch")
plt.xlabel("time (ms)")
plt.ylabel("change in length of muscles (cm)")
plt.legend()
plt.show()