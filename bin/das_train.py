#!/usr/bin/env python

"""
Main training script for DAS machine learning
"""

__copyright__ = """
Machine Learning for Distributed Acoustic Sensing data (MLDAS)
Copyright (c) 2020, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
"""
__license__ = "Modified BSD license (see LICENSE.txt)"
__maintainer__ = "Vincent Dumont"
__email__ = "vincentdumont11@gmail.com"

# System
import os
import argparse
import logging

# Externals
import yaml
import numpy as np
import torch.distributed as dist

# Locals
from mldas.datasets import get_data_loaders
from mldas.trainers import get_trainer
from mldas.utils.logging import config_logging
from mldas.utils.distributed import init_workers

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config')
    add_arg('-o', '--output-dir',
            help='Override output directory')
    add_arg('-d', '--distributed-backend', choices=['mpi', 'nccl', 'gloo'],
            help='Specify which distributed backend to use')
    add_arg('--batch-size', type=int,
            help='Choose a specific batch size')
    add_arg('--depth', type=int,
            help='Choose a specific neural net depth')
    add_arg('--epochs', type=int,
            help='Choose a specific number of epochs')
    add_arg('--lr', type=float,
            help='Choose a specific learning rate')
    add_arg('--num-channels', type=int,
            help='Choose a specific number of channels')
    add_arg('--num-classes', type=int,
            help='Choose a specific number of classes (neurons)')
    add_arg('--sample-size', type=int,
            help='Choose a size of dataset (in thousands)')
    add_arg('--gpu', type=int,
            help='Choose a specific GPU by ID')
    add_arg('--rank-gpu', action='store_true',
            help='Choose GPU according to local rank')
    add_arg('--ranks-per-node', type=int, default=8,
            help='Specify number of ranks per node')
    add_arg('-v', '--verbose', action='store_true',
            help='Enable verbose logging')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    """Main function"""
    
    # Initialization
    args = parse_args()
    rank, n_ranks = init_workers(args.distributed_backend)

    # Load configuration
    config = load_config(args.config)
    if args.sample_size != None:
        config['data_config']['sample_size'] = args.sample_size
    if args.batch_size != None:
        config['data_config']['batch_size'] = args.batch_size
    if args.num_channels != None:
        config['data_config']['num_channels'] = args.num_channels
        config['model_config']['num_channels'] = args.num_channels
    if args.depth != None:
        config['model_config']['depth'] = args.depth
    if args.lr != None:
        config['model_config']['learning_rate'] = args.lr
    if args.num_classes != None and 'multiclass' in args.config:
        config['model_config']['num_classes'] = args.num_classes
    if 'num_classes' in config['model_config'].keys() and config['model_config']['num_classes']==1:
        config['model_config']['loss'] = 'BCE'
    if args.epochs != None:
        config['train_config']['n_epochs'] = args.epochs
        
    # Prepare output directory
    output_dir = os.path.expandvars(args.output_dir if args.output_dir is not None
                                    else config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'out_%i.log' % rank)
    config_logging(verbose=args.verbose, log_file=log_file)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    if rank == 0:
        logging.info('Configuration: %s' % config)
        
    # Load the datasets
    is_distributed = args.distributed_backend is not None
    train_data_loader, valid_data_loader, test_data_loader = get_data_loaders(
        output_dir, is_distributed, **config['data_config'])
    target_data, label_data = next(iter(train_data_loader))
    input_size = target_data.reshape(target_data.shape[0],-1).shape[1]
    output_size = label_data.reshape(label_data.shape[0],-1).shape[1]
    if 'n_layer' in config['model_config'].keys():
        config['model_config']['n_layer'] = [input_size]+config['model_config']['n_layer']+[output_size]
    
    # Load the trainer
    gpu = (rank % args.ranks_per_node) if args.rank_gpu else args.gpu
    if gpu is not None:
        logging.info('Using GPU %i', gpu)
    trainer = get_trainer(name=config['trainer'], distributed=is_distributed,
                          rank=rank, output_dir=output_dir, gpu=gpu)
    
    # Build the model
    trainer.build_model(**config['model_config'])
    if rank == 0:
        trainer.print_model_summary()
        
    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            **config['train_config'])
    trainer.write_summaries()
    
    # Print some conclusions
    logging.info('Finished training')
    logging.info('Train samples %g time %g s rate %g samples/s',
                 np.mean(summary['train_samples']),
                 np.mean(summary['train_time']),
                 np.mean(summary['train_rate']))
    if valid_data_loader is not None:
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     np.mean(summary['valid_samples']),
                     np.mean(summary['valid_time']),
                     np.mean(summary['valid_rate']))
    logging.info('All done!')

if __name__ == '__main__':
    main()
    
