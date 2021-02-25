#! /bin/bash
#SBATCH --ntasks 146
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J set-creation
#SBATCH -o %x-%j.out

# Configuration
#export OMP_NUM_THREADS=32
#export KMP_AFFINITY="granularity=fine,compact,1,0"
#export KMP_BLOCKTIME=1

module load python
#for file in /global/cscratch1/sd/vdumont/30min_files_*/*.mat
#do
#    srun -n1 --exclusive python bin/make_set.py $file &
#done

srun -N1 --exclusive python bin/make_set.py /global/cscratch1/sd/vdumont/30min_files_NoTrain/Dsi_30min_170804123007_170804130007_ch5500_6000.mat  &
srun -N1 --exclusive python bin/make_set.py /global/cscratch1/sd/vdumont/30min_files_Train/Dsi_30min_171029203015_171029210015_ch5500_6000_NS.mat &
srun -N1 --exclusive python bin/make_set.py /global/cscratch1/sd/vdumont/30min_files_NoTrain/Dsi_30min_170804113007_170804120007_ch5500_6000.mat  &
srun -N1 --exclusive python bin/make_set.py /global/cscratch1/sd/vdumont/30min_files_Train/Dsi_30min_171028210015_171028213015_ch5500_6000_NS.mat &
wait
