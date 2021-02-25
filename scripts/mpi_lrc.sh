#!/bin/bash
#SBATCH -n ntasks
#SBATCH -p lr3
#SBATCH -q lr_normal
#SBATCH -A ac_ciftres
#SBATCH -t 01:30:00
#SBATCH -J probmpi
#SBATCH -o logs/%x-%j.out
module purge
module load gcc/7.4.0 hdf5/1.10.5-gcc-p fftw boost python/3.6
if [[ ( $@ == *xcorr* ) || ( $@ == *pws* ) || ( $@ == *weight* ) ]]
then
    mpirun -np ntasks sh $HOME/mldas/scripts/mldas.sh $@ -m 0
else
    export PYTHONPATH=/global/scratch/vdumont/myenv/lib/python3.6/site-packages/:$PYTHONPATH
    mpirun -np ntasks python $HOME/mldas/mldas/assess.py $@
fi
