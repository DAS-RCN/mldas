#!/bin/bash
#SBATCH -n 10
#SBATCH -p lr3
#SBATCH -q lr_normal
#SBATCH -A ac_ciftres
#SBATCH -t 01:00:00
#SBATCH -J probmpi
#SBATCH -o logs/%x-%j.out
module purge
module load gcc/6.3.0
module load openmpi/3.0.1-gcc
module load python/3.6
export PYTHONPATH=/global/scratch/vdumont/myenv/lib/python3.6/site-packages/:$PYTHONPATH
mpirun -np 10 python $HOME/mldas/mldas/assess.py probmap -c $HOME/mldas/configs/assess.yaml -o /global/scratch/vdumont/probmaps --mpi
