import subprocess
import os
import shlex
import csv


def simulation(force):
    os.chdir("..")
    os.chdir("cuboid_muscle/build_release")

    command = shlex.split(f"./incompressible_mooney_rivlin ../settings_force.py incompressible_mooney_rivlin {force}")
    subprocess.run(command)

    f = open("muscle_length_prestretch.csv")
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        prestretch = float(row[1]) - float(row[0])
        print("The muscle was stretched ", prestretch)
    f.close()

    f = open("muscle_length_contraction.csv")
    reader = csv.reader(f, delimiter=",")
    muscle_length_process = []
    for row in reader:
        for j in row:
            muscle_length_process.append(j)
        
    contraction = float(muscle_length_process[0]) - float(muscle_length_process[-2])
    print("The muscle contracted ", contraction)
    f.close()

    return contraction


simulation(4.5)