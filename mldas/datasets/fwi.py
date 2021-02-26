# System
import os
import glob
import math
import random

# Externals
import scipy.io
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler

class make_dataset(Dataset):
  def __init__(self, X_train, Y_train):
    self.X_train = X_train
    self.Y_train = Y_train
    dataset, labels = self.preprocess()
    self.dataset = dataset.unsqueeze(1).float()
    self.labels = labels.float()

  def preprocess(self):
    X_train = self.X_train
    Y_train = self.Y_train
    final, labels = [], []
    for i in range(len(X_train)):
      final.append(numpy.array(X_train[i]))
      labels.append(numpy.array(Y_train[i].flatten()))
    return torch.from_numpy(numpy.array(final)), torch.from_numpy(numpy.array(labels))

  def __getitem__(self,index):
    return self.dataset[index], self.labels[index]

  def __len__(self):
    return len(self.dataset)

def refine_model(vmodel,refine,max_depth,n_dims=1):
  # Create refined depth array
  ymax = vmodel[-1,1] if n_dims==1 else max_depth
  depths = numpy.linspace(0,ymax,num=refine+1)[1:]
  velocities = []
  # Loop through depth values in refined model
  for i,depth in enumerate(depths):
    match = False
    # Loop through original model data points
    for j,(v,z) in enumerate(vmodel):
      if z==depth:
        velocities.append(v)
        match = True
      elif j==0 and 0<depth<=vmodel[j,1]:
        velocities.append(v)
        match = True
      elif vmodel[j-1,1]<depth<=vmodel[j,1]:
        velocities.append(v)
        match = True
    if match==False and vmodel[-1,1]<depth:
      velocities.append(v)
  if n_dims==1:
    vmodel = numpy.array(numpy.vstack((velocities,depths)).T,dtype=float)
  else:
    vmodel = numpy.array(velocities*refine).reshape(refine,refine).T
  return vmodel

def extract_data(dataset,input_type='field',refine=None,conv2d=None,norm=False,n_dims=1,vmax=3040.,ymax=599.,**kwargs):
  X, Y = [], []
  for fname in dataset:
    data = scipy.io.loadmat(fname)
    if input_type=='field':
      if conv2d!=None:
        tmp = torch.tensor([[data['uxt']]]).float()
        tmp = torch.nn.Conv2d(1,1,1+2*conv2d,conv2d,conv2d)(tmp)
        X.append(tmp[0,0].detach().numpy())
      else:
        X.append(data['uxt'])
    elif input_type=='spec':
      if conv2d!=None:
        tmp = torch.tensor([[data['fv']]]).float()
        tmp = torch.nn.Conv2d(1,1,1+2*conv2d,conv2d,conv2d)(tmp)
        X.append(tmp[0,0].detach().numpy())
      else:
        X.append(data['fv'])
    else:
      print('Input type not recognize (%s). Choose between "field" or "spec". Abort.'%input_type)
      quit()
    if norm:
      X[-1] = (X[-1]-X[-1].min())/(X[-1].max()-X[-1].min())
    vmodel = numpy.array([[data['vs'][i,0],sum(data['thk'][:i+1,0])] for i in range(len(data['vs']))],dtype=float)
    if n_dims==1:
      if refine==None:
        Y.append(vmodel)
      else:
        Y.append(refine_model(vmodel,refine))
      if norm:
        Y[-1][:,0] /= vmax
        Y[-1][:,1] /= ymax
    else:
      if refine!=None:
        Y.append(refine_model(vmodel,refine,max_depth=ymax,n_dims=2))
      elif input_type=='field':
        Y.append(refine_model(vmodel,refine=X[-1].shape[0],max_depth=ymax,n_dims=2))
      else:
        print('You must specify the "refine" variable for 2D velocity model when using dispersion spectrum. Abort.')
        quit()
      if norm:
        Y[-1] /= ymax
  return X, Y

def get_data_loaders(output_dir,batch_size,data_path,**kwargs):
  # List files
  file_list = sorted(glob.glob(data_path+'/*.mat'))
  assert len(file_list)>0, 'No data found, check the path. Abort.'
  if random:
    random.shuffle(file_list)
  # Extract data
  X, Y = extract_data(file_list,**kwargs)
  print('Input data of shape',X[-1].shape)
  print('Output data of shape',Y[-1].shape)
  # Create sets by splitting 20/20/60
  split = math.ceil(0.2*len(file_list))
  X_train, Y_train = X[2*split:], Y[2*split:]
  X_test,  Y_test  = X[:split], Y[:split]
  X_valid, Y_valid = X[split:2*split], Y[split:2*split]
  # Prepare dataset
  train_dataset = make_dataset(X_train, Y_train)
  test_dataset  = make_dataset(X_test,  Y_test)
  valid_dataset = make_dataset(X_valid, Y_valid)
  # Create dataloader
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
  return train_loader, valid_loader, test_loader
