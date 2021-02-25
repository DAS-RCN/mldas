#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -d singleton
#SBATCH -c 10
#SBATCH -t 120
#SBATCH -J pytorch-bm-cgpu
#SBATCH -o logs/%x-%j.out

set -e

# Options
version='v1.5.0'
batch_size=128
depth=2
epochs=1
lr=0.01
mode='multiclass'
num_channels=1
num_classes=2
sample_size=1
usage="$0 --mode MODE --batch-size BATCH_SIZE --num-classes NUM_CLASSES --depth DEPTH --epochs EPOCHS --lr LEARNING_RATE --sample-size SAMPLE_SIZE"

# Parse command line options
while (( "$#" )); do
    case "$1" in
        --batch-size)   batch_size=$2;   shift 2;;
        --depth)        depth=$2;        shift 2;;
        --epochs)       epochs=$2;       shift 2;;
        --lr)           lr=$2;           shift 2;;
        --mode)         mode=$2;         shift 2;;
        --num-channels) num_channels=$2; shift 2;;
        --num-classes)  num_classes=$2;  shift 2;;
        --sample-size)  sample_size=$2;  shift 2;;
        *)
            echo "Usage: $usage"; exit 1;;
    esac
done

# Configuration
export JOB_NAME=gpu-n$SLURM_NTASKS-ds$sample_size-bs$batch_size-ep$epochs-dp$depth-lr$lr
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-ml4das/$mode/$num_classes-neuron/$num_channels-channel/$JOB_NAME
[ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH

# Print settings
echo "writing outputs to $BENCHMARK_RESULTS_PATH"

# Load software
module load pytorch/${version}-gpu

# Prepare custom dataset for unary and multilabel classification methods
if [[ "multilabel unary" =~ "${mode}" ]]; then
    python mldas/selection.py configs/${mode}.yaml --sample-size $sample_size -o $BENCHMARK_RESULTS_PATH
fi

# Run each model
srun -l python mldas/train.py configs/${mode}.yaml -d nccl --rank-gpu -v \
     --num-classes $num_classes --num-channels $num_channels \
     --sample-size $sample_size --batch-size $batch_size \
     --epochs $epochs --depth $depth --lr $lr \
     --output-dir $BENCHMARK_RESULTS_PATH

echo "Collecting benchmark results..."
python mldas/parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt -v $version
