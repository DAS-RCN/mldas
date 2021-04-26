# System
import time
import copy

# External
import numpy
import torch
from torch.nn import functional as F

# Local
from ..models import vae

class VAETrainer():

  def __init__(self):
    self.model = vae.get_model(m=50,n=50,b=2)

  # Reconstruction + KL divergence losses summed over all elements and batch
  def loss_function(self, recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

  def train_model(self,loader,epochs,verbose=False):
    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters())
    models,losses = {},numpy.empty((0,2))
    start_loop = time.time()
    start_epoch = time.time()
    for epoch in range(epochs):
      train_loss,total_num = 0,0
      for batch_idx, (data) in enumerate(loader):
        data = data[0].float() if type(data)==list else data.float()
        data = data.view(-1,numpy.prod(data.shape[-2:]))
        z, recon_batch, mu, logvar = self.model(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()
        total_num += data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx+1)%(len(loader)//20)==0 and verbose:
          print('    Batch %i/%i | Loss: %.3f'%(batch_idx+1,len(loader),train_loss/total_num))
      losses = numpy.vstack((losses,[epoch,train_loss/len(loader.dataset)]))
      models[epoch] = copy.deepcopy(self.model)
      if True: #(epoch+1)%(epochs/10)==0
        if verbose: print('\n')
        print('==> Epoch %i/%i | Duration: %.4f seconds | Loss: %.3f'%(epoch+1,epochs,time.time()-start_epoch,train_loss/len(loader.dataset)))
        if verbose: print('\n')
        start_epoch = time.time()
    print('Full loop over %i epochs processed in %.4f seconds'%(epochs,time.time()-start_loop))
    return losses, models
