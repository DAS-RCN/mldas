#!/bin/bash

#for day in {03..12}; do
#    sbatch -n 1 scripts/prob_hsw.sh --day $day
#done

#==============================================================================
#       Scaling on Cori-CPU Haswell nodes
#==============================================================================

#------------------------------------------------------------------------------
# Multiclass version 2 - 1 job
#------------------------------------------------------------------------------
#sbatch -N 5 scripts/das_hsw.sh --depth 20 --lr 0.1 --epochs 20 --sample-size 10 --batch-size 256

#------------------------------------------------------------------------------
# Multilabel version 1 - 3 jobs
#------------------------------------------------------------------------------
#sbatch -N  1 scripts/das_hsw.sh --depth 14 --lr 0.1 --epochs 10 --sample-size  10 --batch-size 256 --mode multilabel
#sbatch -N 10 scripts/das_hsw.sh --depth 20 --lr 0.1 --epochs 50 --sample-size  10 --batch-size 256 --mode multilabel
#sbatch -N  5 scripts/das_hsw.sh --depth 20 --lr 0.1 --epochs 50 --sample-size 100 --batch-size 512 --mode multilabel

#sbatch -N 10 scripts/run_hsw.sh --depth 14 --lr 0.1 --epochs 20 --sample-size 10 --batch-size 256

#------------------------------------------------------------------------------
# Single neuron 
#------------------------------------------------------------------------------

#sbatch -N 3 scripts/run_hsw.sh --depth 8 --lr 0.01 --epochs 10 --sample-size 1 --batch-size 128 --mode singleneuron
#sbatch -N 3 scripts/run_hsw.sh --depth 14 --lr 0.001 --epochs 50 --sample-size 1 --batch-size 128 --mode multiclass --num-channels 3

#==============================================================================
#       Scaling on Cori-GPU with NCCL
#==============================================================================

module purge
module load esslurm

#------------------------------------------------------------------------------
# Multiclass version 1 - 54 jobs
#------------------------------------------------------------------------------

#for depth in {2,8,14}; do for lr in {0.001,0.05,0.01}; do sbatch -n  8 scripts/das_cgpu.sh --depth $depth --lr $lr --epochs 50 --sample-size   1 --batch-size 128; done; done
#for depth in {2,8,14}; do for lr in {0.001,0.05,0.01}; do sbatch -n  8 scripts/das_cgpu.sh --depth $depth --lr $lr --epochs 50 --sample-size   5 --batch-size 128; done; done
#for depth in {2,8,14}; do for lr in {0.001,0.05,0.01}; do sbatch -n  8 scripts/das_cgpu.sh --depth $depth --lr $lr --epochs 10 --sample-size  10 --batch-size 256; done; done
#for depth in {2,8,14}; do for lr in {0.001,0.05,0.01}; do sbatch -n  8 scripts/das_cgpu.sh --depth $depth --lr $lr --epochs 10 --sample-size  50 --batch-size 256; done; done
#for depth in {2,8,14}; do for lr in {0.001,0.05,0.01}; do sbatch -n 16 scripts/das_cgpu.sh --depth $depth --lr $lr --epochs  5 --sample-size 100 --batch-size 512; done; done
#for depth in {2,8,14}; do for lr in {0.001,0.05,0.01}; do sbatch -n 16 scripts/das_cgpu.sh --depth $depth --lr $lr --epochs  5 --sample-size 150 --batch-size 512; done; done

#for depth in {2,8,14,20,26,32}; do for lr in {0.001,0.005,0.01,0.05,0.1,0.5}; do sbatch -n 4 scripts/train_cgpu.sh --depth $depth --lr $lr --epochs 40 --sample-size 10 --batch-size 128 --mode multiclass; done; done
#for depth in {2,8,14,20,26,32}; do for lr in {0.001,0.005,0.01,0.05,0.1,0.5}; do sbatch -n 4 scripts/train_cgpu.sh --depth $depth --lr $lr --epochs 40 --sample-size 100 --batch-size 128 --mode multiclass; done; done

#for ngpu in {2,4,8,16,32}; do for dp in {8,14,20,26,32}; do sbatch -n $ngpu scripts/train_cgpu.sh --depth $dp --epochs 10 --sample-size 10 --batch-size 256 --mode multiclass; done; done
#for ngpu in {2,4,8,16,32}; do for bs in {64,128,256,512,1024}; do sbatch -n $ngpu scripts/train_cgpu.sh --depth 8 --epochs 10 --sample-size 10 --batch-size $bs --mode multiclass; done; done
#for ngpu in {2,4,8,16,32}; do for ds in {1,5,10,50,100}; do sbatch -n $ngpu scripts/train_cgpu.sh --depth 8 --epochs 10 --sample-size $ds --batch-size 256 --mode multiclass; done; done

#------------------------------------------------------------------------------
# Multilabel version 1 - 24 jobs
#------------------------------------------------------------------------------

#for depth in {2,8,14,20}; do for lr in {0.05,0.1,0.5}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr $lr --epochs 10 --sample-size 5  --batch-size 256 --mode multilabel; done; done
#for depth in {2,8,14,20}; do for lr in {0.05,0.1,0.5}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr $lr --epochs 10 --sample-size 10 --batch-size 256 --mode multilabel; done; done

#------------------------------------------------------------------------------
# Multilabel version 2 - 1 job
#------------------------------------------------------------------------------

#sbatch -n 2  scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 50 --sample-size 100 --batch-size 512
#sbatch -n 4  scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 50 --sample-size 100 --batch-size 512
#sbatch -n 8  scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 50 --sample-size 100 --batch-size 512
#sbatch -n 16 scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 50 --sample-size 100 --batch-size 512
#sbatch -n 32 scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 15 --sample-size 100 --batch-size 512

#------------------------------------------------------------------------------
# Multilabel version 3 - 36 jobs
#------------------------------------------------------------------------------

#for depth in {2,8,14}; do for lr in {0.01,0.05,0.1}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr $lr --epochs 50 --sample-size 1  --batch-size 128 --mode multilabel; done; done
#for depth in {2,8,14}; do for lr in {0.01,0.05,0.1}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr $lr --epochs 50 --sample-size 5  --batch-size 128 --mode multilabel; done; done
#for depth in {2,8,14}; do for lr in {0.01,0.05,0.1}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr $lr --epochs 10 --sample-size 10 --batch-size 256 --mode multilabel; done; done
#for depth in {2,8,14}; do for lr in {0.01,0.05,0.1}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr $lr --epochs 10 --sample-size 50 --batch-size 256 --mode multilabel; done; done

#sbatch -n 8 scripts/run_cgpu.sh --depth 14 --lr 0.001 --epochs 100 --sample-size 1 --batch-size 128 --mode multiclass --num-classes 2
#sbatch -n 8 scripts/run_cgpu.sh --depth 14 --lr 0.001 --epochs 100 --sample-size 10 --batch-size 256 --mode multiclass --num-channels 1
#sbatch -n 8 scripts/run_cgpu.sh --depth 14 --lr 0.001 --epochs 100 --sample-size 1 --batch-size 128 --mode multiclass --num-channels 3

#for depth in {2,8,14}; do for sample_size in {1,5};   do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr 0.01 --epochs 50 --sample-size $sample_size --batch-size 128 --mode multiclass; done; done
#for depth in {2,8,14}; do for sample_size in {10,50}; do sbatch -n 16 scripts/run_cgpu.sh --depth $depth --lr 0.01 --epochs 10 --sample-size $sample_size --batch-size 256 --mode multiclass; done; done

#sbatch scripts/prob_hsw.sh --day 03
#sbatch scripts/prob_hsw.sh --day 04
#sbatch scripts/prob_hsw.sh --day 05
#sbatch scripts/prob_hsw.sh --day 06
#sbatch scripts/prob_hsw.sh --day 07
#sbatch scripts/prob_hsw.sh --day 08
#sbatch scripts/prob_hsw.sh --day 09
#sbatch scripts/prob_hsw.sh --day 10
#sbatch scripts/prob_hsw.sh --day 11
#sbatch scripts/prob_hsw.sh --day 12

#sbatch -n 15 scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 50 --sample-size 100 --batch-size 256 --mode multiclass --num-classes 1
#sbatch -n 15 scripts/train_cgpu.sh --depth 8 --lr 0.01 --epochs 50 --sample-size 100 --batch-size 256 --mode multiclass --num-classes 2



#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  1 --batch-size  64 --mode multiclass

#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size   64 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  128 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  256 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  512 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size 1024 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size   64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  128 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  512 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size 1024 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size   64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  128 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  512 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size 1024 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size   64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  128 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  512 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size 1024 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size   64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  128 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size  512 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  5 --batch-size 1024 --mode multiclass

#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 14 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 20 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 26 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth 32 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass

#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   1 --batch-size   64 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   5 --batch-size  128 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  50 --batch-size  512 --mode multiclass
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size 100 --batch-size 1024 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   1 --batch-size   64 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   5 --batch-size  128 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  50 --batch-size  512 --mode multiclass
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size 100 --batch-size 1024 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   1 --batch-size   64 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   5 --batch-size  128 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  50 --batch-size  512 --mode multiclass
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size 100 --batch-size 1024 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   1 --batch-size   64 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   5 --batch-size  128 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  50 --batch-size  512 --mode multiclass
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size 100 --batch-size 1024 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   1 --batch-size   64 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size   5 --batch-size  128 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  10 --batch-size  256 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size  50 --batch-size  512 --mode multiclass
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 10 --sample-size 100 --batch-size 1024 --mode multiclass

# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  512
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size 1024
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size 1024
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size 1024
# sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size 1024
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size 1024
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size 1024
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size 1024
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
# sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size 1024
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size 1024
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size 1024
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size 1024
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size   64
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size 1024
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  128
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size 1024
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size   64
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size 1024
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size   64
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  128
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
# sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size 1024

# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  128
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
# sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
# sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256

#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  512
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size   64
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  128
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size   64
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  128
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  128
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  256
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size   64
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  512
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size   64
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  128
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512

#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  512
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  512
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  512
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size   64
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  128
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  256
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size  512
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  256
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size  512
#sbatch -n 32 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size  512

#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size   64
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  256
#sbatch -n  2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  512
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size  128
#sbatch -n  4 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  256
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size   64
#sbatch -n  8 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size  128
#sbatch -n 16 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size   64

#sbatch -n 2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   1 --batch-size 1024
#sbatch -n 2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size   5 --batch-size 1024
sbatch -n 2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  10 --batch-size 1024
#sbatch -n 2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size  50 --batch-size 1024
#sbatch -n 2 scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size 100 --batch-size 1024
