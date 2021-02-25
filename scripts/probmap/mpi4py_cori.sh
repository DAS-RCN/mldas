#!/bin/bash
#SBATCH -N 3
#SBATCH -q regular
#SBATCH -t 05:00:00
#SBATCH -C haswell
#SBATCH -J prob-multiproc
#SBATCH -o logs/%x-%j.out
module load pytorch/v1.6.0
srun -n 96 -c 2 python $HOME/mldas/mldas/assess.py probmap -c $HOME/mldas/configs/assess.yaml -o $SCRATCH/probmaps --mpi
