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
import time
import logging

# Externals
import numpy as np
import torch

class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None,
                 distributed=False, rank=0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        if gpu is not None:
            self.device = torch.device('cuda', gpu)
            torch.cuda.set_device(gpu)
        else:
            self.device = torch.device('cpu')
        self.distributed = distributed
        self.rank = rank
        self.summaries = {}

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel()
             for p in self.model.parameters()))
        )

    def save_summary(self, summaries):
        """Save summary information"""
        for (key, val) in summaries.items():
            summary_vals = self.summaries.get(key, [])
            self.summaries[key] = summary_vals + [val]

    def write_summaries(self):
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir,
                                    'summaries_%i.npz' % self.rank)
        self.logger.info('Saving summaries to %s' % summary_file)
        np.savez(summary_file, **self.summaries)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(dict(model=self.model.state_dict()),
                   os.path.join(checkpoint_dir, checkpoint_file))

    def build_model(self):
        """Virtual method to construct the model"""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None, test_data_loader=None, **kwargs):
        """Run the model training"""
        
        # Loop over epochs
        for i in range(n_epochs):
            if i+1 in self.lr_decay_epoch:
                self.optimizer = self.exp_lr_scheduler(self.optimizer)
            self.logger.info('  EPOCH {:>3}/{:<3} | Model initial sumw: {:.5f} |'.format(i+1,n_epochs,sum(p.sum() for p in self.model.parameters())))
            summary = dict(epoch=i)
            # Train on this epoch
            start_time = time.time()
            summary.update(self.train_epoch(train_data_loader,**kwargs))
            summary['train_time'] = time.time() - start_time
            summary['train_samples'] = len(train_data_loader.sampler)
            summary['train_rate'] = summary['train_samples'] / summary['train_time']
            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                summary.update(self.evaluate(valid_data_loader,'Validation',**kwargs))
                summary['valid_time'] = time.time() - start_time
                summary['valid_samples'] = len(valid_data_loader.sampler)
                summary['valid_rate'] = summary['valid_samples'] / summary['valid_time']
            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None and self.rank==0:
                self.write_checkpoint(checkpoint_id=i)
        # Evaluate on this epoch
        if test_data_loader is not None:
            self.evaluate(test_data_loader,'Testing',**kwargs)
        return self.summaries
