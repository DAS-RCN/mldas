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
import os,glob,random,time

# Externals
import torch,h5py,numpy

def set_from_file(filename,img_size=100,stride=10,adjust=False,shuffle=True,select=None,nrand=None):
  '''
  Create set of images out of single DAS data file.
  
  Parameters
  ----------
  filename : :py:class:`str`
    Path the data file
  img_size : :py:class:`int`
    Size of squared image
  stride : :py:class:`int`
    Sliding interval
  adjust : :py:class:`bool`
    Brightness adjustment
  shuffle : :py:class:`bool`
    Shuffle output dataset
  select : :py:class:`int`, :py:class:`list`
    Index(es) of specific image index
  nrand : :py:class:`int`
    Number of images to be randomly selected

  Returns
  -------
  Xtrain : :py:class:`list`
    List of individual images
  '''
  assert os.path.exists(filename), '%s: Path not found, make sure the Google Drive is loaded.'%datapath
  assert not(select!=None and nrand!=None), "Choose either random number or selected indexes. Not both."
  # Load file
  f = h5py.File(filename,'r')
  data = numpy.array(f[f.get('variable/dat')[0,0]])
  f.close()
  # Get corner indexes of every images
  idxs = numpy.array([[[i,j] for j in range(0,data.shape[1]-img_size+1,stride)] for i in range(0,data.shape[0]-img_size+1,stride)])
  idxs = idxs.reshape(idxs.shape[0]*idxs.shape[1],2)
  # If select defined, select index
  if select!=None:
    idxs = idxs[select] if type(select)==list else idxs[[select]]
  # If nrand non zero and less than actual size of idxs, randomly select indexes
  if nrand!=None and nrand<len(idxs):
    idxs = idxs[random.sample(range(0,idxs.shape[0]),nrand)]
  # Loop over images and extract data
  Xtrain = []
  for k,(i,j) in enumerate(idxs):
    img = data[i:i+img_size,j:j+img_size].copy()
    img = (img-img.min())/(img.max()-img.min())
    if adjust:
      img = mean_shift(img)
    Xtrain.append(img)
  if shuffle:
    random.shuffle(Xtrain)
  return Xtrain

def mean_shift(data,loops=1):
  '''
  Shit all values from input 2D array to match mean value of 0.5.

  Parameters
  ----------
  data : :py:class:`numpy.ndarray`
    Input 2D image
  loops : :py:class:`int`
    Number of iteration of algorithm
  
  Returns
  -------
  data : :py:class:`numpy.ndarray`
    Modified 2D data
  '''
  for i in range(loops):
    ref_mean = data.mean()
    infs = numpy.where(data<=ref_mean)
    sups = numpy.where(data>ref_mean)
    data[infs] = data[infs] / ref_mean * 0.5
    data[sups] = 1 - (1-data[sups]) / (1-ref_mean) * 0.5
  return data

def set_creation(datapath,img_size=100,stride=10,sample_size=1,adjust=False,shuffle=True,nrand=None,select=None,verbose=False):
  """
  Create PyTorch tensor training set of single-channel data images.
  
  Parameters
  ----------
  datapath : :py:class:`str`
    Path to data repository
  img_size : :py:class:`int`
    Size of squared image
  stride : :py:class:`int`
    Sliding interval
  sample_size : :py:class:`int`
    Number of input MAT data files to use
  adjust : :py:class:`bool`
    Do brightness adjustment
  shuffle : :py:class:`bool`
    Shuffling images in set
  nrand : :py:class:`int`
    Specify total number of images in final set
  select : :py:class:`int`, :py:class:`list`
    Index(es) of specific image index  
  verbose : :py:class:`bool`
    Do verbose
  
  Returns
  -------
  Xtrain : :py:class:`torch.Tensor`
    Tensor of images
  """
  assert os.path.exists(datapath), '%s: Path not found, make sure the Google Drive is loaded.'%datapath
  # List all data files available in target repository
  sample = [datapath] if datapath.endswith('.mat') else glob.glob(datapath+'/*.mat')
  # Randomly select files
  idxs = random.sample(range(0,len(sample)),sample_size)
  # Initialize list to store training images
  Xtrain = []
  # Loop over randomly selecetd files
  for i,idx in enumerate(idxs):
    if verbose: print(sample[idx])
    # If multiple file selected, wait to load all images before random selection
    n = None if nrand!=None and sample_size>1 else nrand
    # Store training data in dataset list
    Xtrain.extend(set_from_file(sample[idx],img_size,stride,adjust,shuffle,select,n))
  # Convert list to numpy array
  Xtrain = numpy.array(Xtrain,dtype=float)
  # Check if number of random images is less than dataset size
  if nrand!=None and nrand<len(Xtrain):
    # Select random images from dataset
    Xtrain = Xtrain[random.sample(range(0,len(Xtrain)),nrand)]
  # Convert numpy array to PyTorch tensor
  Xtrain = torch.from_numpy(numpy.reshape(Xtrain,(len(Xtrain),1,img_size,img_size)))
  return Xtrain

def prepare_loader(datapath,img_size,stride,sample_size,batch_size,nrand=None,adjust=False,shuffle=True,verbose=True):
  """
  Create custom data loader with unlabeled, single-channel, data images directly extracted from raw data files.

  Parameters
  ----------
  datapath : :py:class:`str`
    Path to raw data repository
  img_size : :py:class:`int`
    Size of squared image
  stride : :py:class:`int`
    Sliding interval
  sample_size : :py:class:`int`
    Number of input MAT data files to use
  batch_size : :py:class:`int`
    Batch size to use in :py:class:`~torch.utils.data.DataLoader`
  nrand : :py:class:`int`
    Specify total number of images in final set
  adjust : :py:class:`bool`
    Do brightness adjustment
  shuffle : :py:class:`bool`
    Shuffling images in set
  verbose : :py:class:`bool`
    Do verbose

  Returns
  -------
  train_loader : :py:class:`torch.utils.data.DataLoader`
    Data loader
  """
  start_load = time.time()
  Xtrain = set_creation(datapath,img_size,stride,sample_size,adjust,shuffle,nrand)
  # Select random images if nrand not None
  if nrand!=None and nrand<len(Xtrain):
    Xtrain = Xtrain[random.sample(range(0, len(Xtrain)), nrand)]
  # Check that the number of training images is larger than the batch size
  if len(Xtrain)<batch_size:
    if verbose: print('Input batch size too large (%i), reset to dataset size (%i).'%(batch_size,len(Xtrain)))
    batch_size = len(Xtrain)
  # Create Dataloader objects
  train_loader = torch.utils.data.DataLoader(dataset=Xtrain,batch_size=batch_size,shuffle=True)
  # Get time spent and final data size
  if verbose:
    print('Train loader created in {0:.2f} seconds from {1:,} files with {2:,} images split into:'.format(time.time()-start_load,sample_size,len(Xtrain)))
    print('\t{0:,} batches of {1:,} images'.format(len(Xtrain)//batch_size,batch_size))
  if len(Xtrain)//batch_size!=len(train_loader) and verbose:
    print('\t1 batch of {:,} images'.format(len(Xtrain)-(len(Xtrain)//batch_size)*batch_size))
  return train_loader
