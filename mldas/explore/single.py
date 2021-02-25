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

import torch,numpy,copy

def suplearn_simple(model,criterion,optimizer,train_loader,test_loader,epochs=1,print_every=1,save_model=False,verbose=True):
    """
    Simple, non-optimized, supervised training with validation step performed at regular
    intervals during batch iteration for single node, single processor execution.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
      Trained model
    criterion : e.g. :py:class:`torch.nn.CrossEntropyLoss`
      Loss function
    optimizer : :py:class:`torch.optim.Optimizer`
      Optimizer to perform gradient descent
    train_loader : :py:class:`torch.utils.data.DataLoader`
      Input dataset for training part 
    test_loader : :py:class:`torch.utils.data.DataLoader`
      Input dataset for validation step
    epochs : :py:class:`int`
      Number of epochs to execute the training
    print_every : :py:class:`int`
      Batch interval at which both training/validation loss and accuracy are evaluated
    save_model : :py:class:`bool`
      Save updated model in dictionary

    Returns
    -------
    loss_hist : :py:class:`numpy.ndarray`
      History of loss values
    model : :py:class:`torch.nn.Module` or :py:class:`dict`
      Final trained model or dictionary of models.
    """
    # Initialize parameters
    models,loss_hist = {},numpy.empty((0,8))
    # Loop over epochs
    for epoch in range(epochs):
        train_num,train_acc,train_loss  = 0,0,0.
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            train_loss += loss.item() * inputs.size(0)
            train_num += inputs.size(0)
            _, predicted = output.max(1)
            train_acc += predicted.eq(labels).sum().item()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % print_every == 0:
                if save_model:
                    models[len(loss_hist)] = copy.deepcopy(model)
                test_num,test_acc,test_loss  = 0,0,0.
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        output = model(inputs)
                        loss = criterion(output, labels)
                        test_loss += loss.item() * inputs.size(0)
                        test_num += inputs.size(0)
                        _, predicted = output.max(1)
                        test_acc += predicted.eq(labels).sum().item()
                loss_hist = numpy.vstack((loss_hist,[epoch,batch_idx,train_num,train_loss,train_acc,test_num,test_loss,test_acc]))
                if verbose:
                    print(f"Epoch {epoch+1:>3}/{epochs} | "+
                          f"Batch {batch_idx+1:>3}/{len(train_loader)} | "
                          f"Training loss: {train_loss/train_num:.5f} | "
                          f"Training accuracy: {100*train_acc/train_num:>7.3f} ({train_acc}/{train_num}) | "
                          f"Validation loss: {test_loss/test_num:.5f} | "
                          f"Validation accuracy: {100*test_acc/test_num:>7.3f} ({test_acc}/{test_num})")
                train_num,train_acc,train_loss = 0,0,0
    model = models if save_model else model
    return loss_hist,model
