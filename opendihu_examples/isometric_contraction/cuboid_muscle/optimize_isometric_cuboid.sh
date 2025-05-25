#!/bin/bash
#
# Job name: #SBATCH -J Optimize_isometric_cuboid
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
#SBATCH --mail-user=bauerls2@ipvs.uni-stuttgart.de
#
# Wall clock limit:#SBATCH --time=60:00:00
## Compute resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "SLURM_NNODES"=$SLURM_NNODES

echo "working directory="$SLURM_SUBMIT_DIR

echo "Configuring enviroment variables"

echo "Launching muscle"

srun python BayesOpt_cuboid_muscle.py &> optimize_isometric_cuboid.log

echo "Simulation completed."
