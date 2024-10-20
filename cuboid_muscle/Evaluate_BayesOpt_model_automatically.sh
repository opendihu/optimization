#!/bin/bash
#
# Job name:
#SBATCH -J Eval_BayesOpt
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
#SBATCH --time=24:00:00
#
# Compute resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "SLURM_NNODES"=$SLURM_NNODES
echo "working directory="$SLURM_SUBMIT_DIR

echo "Configuring enviroment variables"

echo "Running Optimization Loop"
python Evaluate_BayesOpt_model_automatically.py  &> Evaluate_BayesOpt_model_automatically.log
echo "Simulation completed."



