#!/usr/bin/env bash

cd $SCRATCH/probmaps
module load pytorch/v1.6.0
python $HOME/mldas/mldas/assess.py -f $1 -m $SCRATCH/pytorch-ml4das/multiclass/2-neuron/3-channel/gpu-n8-ds1-bs128-ep50-dp14-lr0.001/checkpoints/model_checkpoint_049.pth.tar -d 14
