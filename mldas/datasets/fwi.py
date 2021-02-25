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

def refine_model(vmodel,refine):
  # Add [0,0] reference point in original model
  vmodel = numpy.vstack(([0,0],vmodel))
  # Create refined new model
  new_model = numpy.zeros((refine,2))
  new_model[:,1] = numpy.linspace(vmodel[0,1],vmodel[-1,1],num=refine)
  # Loop through depth values in refined model
  for i,depth in enumerate(new_model[:,1]):
    # Loop through original model data points
    for j,(v,z) in enumerate(vmodel):
      if z==depth:
        new_model[i,0] = v
      elif vmodel[j-1,1]<depth<=vmodel[j,1]:
        new_model[i,0] = v
  return new_model

def extract_data(dataset,input_type,refine,conv2d,norm):
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
    else:
      if conv2d!=None:
        tmp = torch.tensor([[data['fv']]]).float()
        tmp = torch.nn.Conv2d(1,1,1+2*conv2d,conv2d,conv2d)(tmp)
        X.append(tmp[0,0].detach().numpy())
      else:
        X.append(data['fv'])
    if norm:
      X[-1] = (X[-1]-X[-1].min())/(X[-1].max()-X[-1].min())
    vmodel = numpy.array([[data['vs'][i,0],sum(data['thk'][:i+1,0])] for i in range(len(data['vs']))])
    if refine!=None:
      Y.append(refine_model(vmodel,refine))
    else:
      Y.append(vmodel)
    if norm:
      print(Y[-1][:,0])
      quit()
      Y[-1][:,0] = (Y[-1][:,0]-)/(Y[-1].max()-Y[-1].min())
      Y[-1] = (Y[-1]-Y[-1].min())/(Y[-1].max()-Y[-1].min())
  return X, Y

def get_data_loaders(output_dir,batch_size,data_path,input_type='field',
                     refine=None,conv2d=None,norm=False,**dataset_args):
  # List files
  file_list = glob.glob(data_path+'/*.mat')
  assert len(file_list)>0, 'No data found, check the path. Abort.'
  random.shuffle(file_list)
  # Extract data
  X, Y = extract_data(file_list,input_type,refine,conv2d,norm)
  print('Input data of shape',X[-1].shape)
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
  test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
  valid_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
  return train_loader, valid_loader, test_loader
