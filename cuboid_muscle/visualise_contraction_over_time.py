import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

os.chdir("build_release")

with open("muscle_length_prestretch0.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_all = float(row[0])

lengths1 = []
with open("muscle_length_contraction0.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_1 = float(row[0])
        for j in row:
            if j != "":
                lengths1.append(float(j))
lengths1 = np.ones(len(lengths1))*starting_length_all - lengths1

lengths2 = []
with open("muscle_length_prestretch10.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_2 = float(row[0])

with open("muscle_length_contraction10.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_2 = float(row[0])
        for j in row:
            if j != "":
                lengths2.append(float(j))
lengths2 = np.ones(len(lengths1))*starting_length_all - lengths2

lengths3 = []
with open("muscle_length_prestretch20.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_3 = float(row[0])

with open("muscle_length_contraction20.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_3 = float(row[0])
        for j in row:
            if j != "":
                lengths3.append(float(j))
lengths3 = np.ones(len(lengths1))*starting_length_all - lengths3

lengths4 = []
with open("muscle_length_prestretch30.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_4 = float(row[0])

with open("muscle_length_contraction30.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_4 = float(row[0])
        for j in row:
            if j != "":
                lengths4.append(float(j))
lengths4 = np.ones(len(lengths1))*starting_length_all - lengths4

lengths5 = []
with open("muscle_length_prestretch40.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_5 = float(row[0])

with open("muscle_length_contraction40.0N.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        starting_length_5 = float(row[0])
        for j in row:
            if j != "":
                lengths5.append(float(j))
lengths5 = np.ones(len(lengths1))*starting_length_all - lengths5

number_elements = len(lengths1)
timesteps = np.linspace(0,50,number_elements)
plt.plot(timesteps, lengths1, label="0.0N")
plt.plot(timesteps, lengths2, label="10.0N")
plt.plot(timesteps, lengths3, label="20.0N")
plt.plot(timesteps, lengths4, label="30.0N")
plt.plot(timesteps, lengths5, label="40.0N")
#plt.plot([0,50], [starting_length_all, starting_length_all], label="starting length")
plt.title("length of the muscle over time")
plt.xlabel("time in ms")
plt.ylabel("cm")
plt.legend()
plt.show()