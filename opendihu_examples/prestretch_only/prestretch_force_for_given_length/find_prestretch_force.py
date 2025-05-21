import sys
import os
import shlex
import subprocess
import csv

prestretch_length = float(sys.argv[1])
error_tolerance = 0.01

def find_prestretch_length(force):
    individuality_parameter = str(force)
    command = shlex.split(f"./incompressible_mooney_rivlin_prestretch_only ../prestretch_tensile_test.py incompressible_mooney_rivlin_prestretch_only {force} {individuality_parameter}")
    subprocess.run(command)
    f = open("muscle_length_prestretch"+individuality_parameter+".csv")
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        prestretch_extension = float(row[1]) - float(row[0])
        print("The muscle was stretched ", prestretch_extension)
    f.close()
    return prestretch_extension

def find_prestretch_force():
    lower_guess = 0
    upper_guess = 10
    prestretch_extension_low = 0
    prestretch_extension_up = find_prestretch_length(upper_guess)

    while (prestretch_extension_low < prestretch_length-error_tolerance or prestretch_extension_low > prestretch_length+error_tolerance) and (prestretch_extension_up < prestretch_length-error_tolerance or prestretch_extension_up > prestretch_length+error_tolerance):
        if prestretch_extension_up < prestretch_length-error_tolerance and prestretch_extension_up >= 0:
            temp = upper_guess
            upper_guess = 2*upper_guess - lower_guess
            lower_guess = temp
            prestretch_extension_low = prestretch_extension_up
            prestretch_extension_up = find_prestretch_length(upper_guess)
        else:
            middle_guess = (upper_guess+lower_guess)/2
            prestretch_extension_mid = find_prestretch_length(middle_guess)
            if prestretch_extension_mid < prestretch_length-error_tolerance:
                lower_guess = middle_guess
                prestretch_extension_low = prestretch_extension_mid
            elif prestretch_extension_mid > prestretch_length+error_tolerance:
                upper_guess = middle_guess
                prestretch_extension_up = prestretch_extension_mid
            else:
                return middle_guess
        
    return upper_guess

os.chdir("build_release")
prestretch_force = find_prestretch_force()

print("###########################################################################")
print("The prestretch force to get a prestretch extension of ", prestretch_length, " is ",prestretch_force)
print("###########################################################################")

f = open("prestretch_force_for_given_length.csv", "a")
reader = csv.reader(f, delimiter=",")
f.write("Prestretch length: {}, prestretch force: {}\n".format(prestretch_length, prestretch_force))
f.close()