#!/bin/sh
#SBATCH -N 2 -c 64
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -C haswell
#SBATCH -J prob-taskfarm
#SBATCH -o logs/%x-%j.out

cd $SCRATCH/probmaps
export THREADS=32

runcommands.sh $HOME/mldas/scripts/probmap/tasks.txt
