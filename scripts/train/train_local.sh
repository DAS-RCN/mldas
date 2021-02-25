#!/bin/bash

set -e

# Options
batch_size=128
depth=2
epochs=1
lr=0.01
mode='multilabel'
sample_size=1
usage="$0 --mode MODE --batch-size BATCH_SIZE --depth DEPTH --epochs EPOCHS --lr LEARNING_RATE --sample-size SAMPLE_SIZE"

# Parse command line options
while (( "$#" )); do
    case "$1" in
        --batch-size)
            batch_size=$2
            shift 2
            ;;
        --depth)
            depth=$2
            shift 2
            ;;
        --epochs)
            epochs=$2
            shift 2
            ;;
        --lr)
            lr=$2
            shift 2
            ;;
        --mode)
            mode=$2
            shift 2
            ;;
        --sample-size)
            sample_size=$2
            shift 2
            ;;
        *)
            echo "Usage: $usage"
            exit 1
            ;;
    esac
done

export JOB_NAME=mpirun-ds$sample_size-bs$batch_size-ep$epochs-dp$depth-lr$lr
export BENCHMARK_RESULTS_PATH=./output/$mode/$JOB_NAME

[ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH

# Print settings
echo "writing outputs to $BENCHMARK_RESULTS_PATH"

# Prepare json dataset
python mldas/selection.py configs/local/${mode}.yaml --sample-size $sample_size -o $BENCHMARK_RESULTS_PATH

# Run model
mpirun -np 1 python mldas/train.py configs/local/${mode}.yaml -v \
       --sample-size $sample_size --batch-size $batch_size --epochs $epochs --depth $depth --lr $lr \
       --output-dir $BENCHMARK_RESULTS_PATH

echo "Collecting benchmark results..."
python mldas/parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
