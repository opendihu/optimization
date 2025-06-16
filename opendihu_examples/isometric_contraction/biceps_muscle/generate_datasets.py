import numpy as np
import shlex
import subprocess
import csv
import os

os.chdir("build_release")

forces = np.linspace(9.5,10,6)

for force in forces:

    command = shlex.split(f"mpirun -n 16 ./muscle_contraction_with_prestretch ../settings_contraction_with_prestretch.py ramp.py --prestretch_force {force} --scenario_name {force}")
    subprocess.run(command)

    tractionz_rank0 = []
    tractionz_rank1 = []
    tractionz_rank2 = []
    tractionz_rank3 = []

    f = open("out/prestretch" + str(force) + "/muscle_contraction_rank0.csv")
    reader = csv.reader(f)
    for row in reader:
        tractionz_rank0.extend([float(value) if value.strip() else 0 for value in row])
    f.close()

    f = open("out/prestretch" + str(force) + "/muscle_contraction_rank1.csv")
    reader = csv.reader(f)
    for row in reader:
        tractionz_rank1.extend([float(value) if value.strip() else 0 for value in row])
    f.close()

    f = open("out/prestretch" + str(force) + "/muscle_contraction_rank2.csv")
    reader = csv.reader(f)
    for row in reader:
        tractionz_rank2.extend([float(value) if value.strip() else 0 for value in row])
    f.close()

    f = open("out/prestretch" + str(force) + "/muscle_contraction_rank3.csv")
    reader = csv.reader(f)
    for row in reader:
        tractionz_rank3.extend([float(value) if value.strip() else 0 for value in row])
    f.close()

    f = open("out/prestretch" + str(force) + "/muscle_prestretch_rank0.csv")
    reader = csv.reader(f)
    for row in reader:
        length_after_prestretch = row[1]
    f.close()

    # we look at the max in absolute value, and we assume it will be negative
    maxtraction = max((np.array(tractionz_rank0)+np.array(tractionz_rank1)+np.array(tractionz_rank2)+np.array(tractionz_rank3))/4)

    print("The maximum traction was ", maxtraction)

    f = open("biceps_dataset.csv", "a")
    f.write(str(force) + "," + str(maxtraction))
    f.write("\n")
    f.close()
