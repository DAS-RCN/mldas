"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import numpy
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# Locals
from .base import BaseTrainer
from ..models import get_model

class GenericTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""
    
    def __init__(self, **kwargs):
        super(GenericTrainer, self).__init__(**kwargs)
        
    def build_model(self, model_type='resnet', loss='CE', optimizer='SGD',
                    learning_rate=0.01, lr_decay_epoch=[], lr_decay_ratio=0.5,
                    momentum=0.9, **model_args):
        """Instantiate our model"""
        self.loss = loss
        self.lr_decay_ratio = lr_decay_ratio
        self.lr_decay_epoch = lr_decay_epoch
        # Construct the model
        self.model = get_model(name=model_type, **model_args).to(self.device)
        
        # Distributed data parallelism
        if self.distributed:
            device_ids = [self.gpu] if self.gpu is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)
            
        # TODO: add support for more optimizers and loss functions here
        opt_type = dict(SGD=torch.optim.SGD)[optimizer]
        self.optimizer = opt_type(self.model.parameters(), lr=learning_rate, momentum=momentum)
        loss_type = dict(CE=torch.nn.CrossEntropyLoss,
                         BCE=torch.nn.BCEWithLogitsLoss,
                         MSE=torch.nn.MSELoss)[loss]
        self.loss_func = loss_type()


    def exp_lr_scheduler(self, optimizer):
        """
        Decay learning rate by a factor of lr_decay 
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.lr_decay_ratio
        return optimizer
        
    def train_epoch(self, data_loader, rounded=False, **kwargs):
        """Train for one epoch"""
        self.model.train()
        sum_loss = 0
        sum_correct = 0
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            if self.loss=='BCE' and batch_target.dim()==1:
                batch_target = batch_target.float().unsqueeze(1)
            batch_target = batch_target.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            if rounded: batch_output = batch_output.round()
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            loss = batch_loss.item()
            sum_loss += loss
            n_correct = self.accuracy(batch_output, batch_target, **kwargs)
            sum_correct += n_correct
            self.logger.debug(' batch {:>3}/{:<3} | {:6,} samples | Loss {:.5f} | Accuracy {:6.2f}'
                              .format(i+1, len(data_loader), len(batch_input), loss, 100*n_correct/len(batch_input)))
        train_loss = sum_loss / (i + 1)
        train_acc = sum_correct / len(data_loader.sampler)
        self.logger.debug('{:>14} | {:6,} samples | Loss {:.5f} | Accuracy {:6.2f}'
                          .format('Training', len(data_loader.sampler), train_loss, 100*train_acc))
        return dict(train_loss=train_loss)
    
    @torch.no_grad()
    def evaluate(self, data_loader, mode, rounded=False, **kwargs):
        """"Evaluate the model"""
        self.model.eval()
        sum_loss = 0
        sum_correct = 0
        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            if self.loss=='BCE' and batch_target.dim()==1:
                batch_target = batch_target.float().unsqueeze(1)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            loss = self.loss_func(batch_output, batch_target).item()
            sum_loss += loss
            n_correct = self.accuracy(batch_output, batch_target, **kwargs)
            sum_correct += n_correct
        valid_loss = sum_loss / (i + 1)
        valid_acc = sum_correct / len(data_loader.sampler)
        self.logger.debug('{:>14} | {:6,} samples | Loss {:.5f} | Accuracy {:6.2f}'
                          .format(mode, len(data_loader.sampler), valid_loss, 100*valid_acc))
        return dict(valid_loss=valid_loss, valid_acc=valid_acc)
    
    def accuracy(self, batch_output, batch_target, acc_tol=20):
        # Count number of correct predictions
        if self.loss=='MSE':
            #batch_preds = torch.round(batch_output)
            batch_preds = batch_output
            #n_correct = batch_preds.eq(batch_target).float().mean(dim=1).sum().item()
            #n_correct = batch_preds.sub(batch_target).abs().lt(acc_tol).float().mean(dim=1).sum().item()
            n_correct = batch_target.sub(batch_preds).square().div(batch_preds.square()).sqrt().mul(100).lt(acc_tol).float().mean(dim=1).sum().item()
        elif self.loss=='BCE':
            batch_preds = (torch.sigmoid(batch_output)>0.5).float()
            if batch_preds.dim()==1:
                n_correct = batch_preds.eq(batch_target).float().sum()
            else:
                n_correct = batch_preds.eq(batch_target).all(dim=1).float().sum()
        else:
            _, batch_preds = torch.max(batch_output, 1)
            n_correct = batch_preds.eq(batch_target).sum().item()
        return n_correct
    
def get_trainer(**kwargs):
    """
    Test
    """
    return GenericTrainer(**kwargs)

def _test():
    t = GenericTrainer(output_dir='./')
    t.build_model()
    
