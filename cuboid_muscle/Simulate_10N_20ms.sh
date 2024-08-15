#!/bin/bash
#
# Job name:
#SBATCH -J Simulate_10N_20ms
#
# Error and Output files
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#
# Working directory:
#SBATCH -D ./
#
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=LCBa@web.de
#
# Wall clock limit:
#SBATCH --time=3:00:00
#
# Compute resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "SLURM_NNODES"=$SLURM_NNODES
echo "working directory="$SLURM_SUBMIT_DIR

echo "Configuring enviroment variables"

echo "Launching muscle"
mpirun -n 1 python Simulate_10N_20ms.py  &> Simulate_10N_20ms.log
echo "Simulation completed."



~                                                                                                                                                                                                           
~                                         
