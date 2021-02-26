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
import sys
import glob

# Externals
import numpy
import torch
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import cm
from PIL import Image

# Local
from .loading import hdf5read, load_model
from ..models import get_model

def extract_prob_map(data,model,img_size=10,channel_stride=1,sample_stride=10,
                     verbose=False,stage_plot=False,single=False,rgb=False,colormap=None):
  """
  Calculate probability map using fed raw data and training model.

  Parameters
  ----------
  data : :py:class:`numpy.ndarray`
    Input raw measurements extracted from HDF5 file
  model : :py:class:`torch.nn.Module`
    Trained model
  img_size : :py:class:`int`
    Size of squared image
  channel_stride : :py:class:`int`
    Sliding interval in channel space
  sample_stride : :py:class:`int`
    Sliding interval in sample space
  verbose : :py:class:`bool`
    Do verbose
  stage_plot : :py:class:`bool`
    Plot intermediate stages

  Returns
  -------
  prob_array : :py:class:`numpy.ndarray`
    Normalized probability map
  """
  # Set model to evalutation mode
  model.eval()
  # Initialize probability map
  prob_array = numpy.zeros((2,*data.shape))
  idxs = numpy.array([[[i,j] for j in range(0,data.shape[1]-img_size+1,sample_stride)] \
                      for i in range(0,data.shape[0]-img_size+1,channel_stride)])
  idxs = idxs.reshape(idxs.shape[0]*idxs.shape[1],2)
  if verbose: print('Processing %i regions...'%len(idxs))
  for k,(i,j) in enumerate(idxs):
    if verbose and (k+1)%(len(idxs)//10)==0:
      print('%i%% processed (%i/%i)'%(round((k+1)/len(idxs)*100),k+1,len(idxs)))
    # Create copy of square data window
    image = data[i:i+img_size,j:j+img_size].copy()
    # Normalize data
    image = (image-image.min())/(image.max()-image.min())
    if colormap=='gist_earth':
      image = cm.gist_earth(image)
    image = Image.fromarray(numpy.uint8(image*255))
    # Convert data to RGB image
    if rgb:
      image = transforms.ToTensor()(image.convert("RGB")).view(1,3,img_size,img_size)
    else:
      image = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(image).view(1,1,img_size,img_size)
    # Run trained model to image
    output = model(image)
    if single:
      wave_prob = float(torch.sigmoid(output))
    else:
      # Get probability for each class
      prob = F.softmax(output,dim=1).topk(2)
      # Check if label not found
      assert int(prob[1][0,0]) in [0,1],"Maximum probability class has an unknown label..."
      # Get surface wave probability
      wave_prob = 1-float(prob[0][0,0]) if int(prob[1][0,0])==0 else float(prob[0][0,0])
    # Increment probability to map
    prob_array[0,i:i+img_size,j:j+img_size]+=wave_prob
    # Increment scanning index to map
    prob_array[1,i:i+img_size,j:j+img_size]+=1
    # If frame requested to be plotted
    if stage_plot:
      # Ignore divide by zero warning message
      numpy.seterr(divide='ignore', invalid='ignore')
      # Plot data and probability map at current scanning stage
      plot_frame(i,j,data,im,prob_array[0]/prob_array[1],img_size,wave_prob,n+1)
  # Return weighted probability for every pixel
  return prob_array[0]/prob_array[1]

def minute_prob(data_path,model_file,depth,img_size=200,num_channels=3,num_classes=2,channel_stride=1,sample_stride=10,model_type='resnet',multilabel=False,compact=False):
  """
  Get surface wave probabilities for every consecutive square regions.
  """
  assert num_channels in [1,3], 'Number of channels must be either 1 or 3.'
  assert num_classes<3, 'More than 2 classes not implemented yet.'
  data = hdf5read(os.path.expanduser(data_path))
  model = get_model(model_type,depth=depth,num_channels=num_channels,num_classes=num_classes)
  model = load_model(model_file,model)
  model.eval()
  idxs = numpy.array([[[i,j] for j in range(0,data.shape[1]-img_size+1,sample_stride)] for i in range(0,data.shape[0]-img_size+1,channel_stride)])
  idxs = idxs.reshape(idxs.shape[0]*idxs.shape[1],2)
  prob_size1 = data.shape[0]-data.shape[0]%channel_stride
  prob_size2 = data.shape[1]-data.shape[1]%sample_stride
  prob_array = numpy.zeros((2,prob_size1,prob_size2),dtype=numpy.float16)
  for (i,j) in idxs:
    image = data[i:i+img_size,j:j+img_size].copy()
    image = (image-image.min())/(image.max()-image.min())
    image = Image.fromarray(numpy.uint8(image*255))
    if num_channels==1:
      image = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(image).view(1,1,img_size,img_size)
    if num_channels==3:
      image = transforms.ToTensor()(image.convert("RGB")).view(1,3,img_size,img_size)
    output = model(image)
    if output.dim()==1:
      wave_prob = torch.sigmoid(output).item()
    elif output.shape[1]==1:
      wave_prob = torch.sigmoid(output)[0,0].item()
    elif multilabel:
      wave_prob = torch.sigmoid(output)[0,1].item()
    else:
      wave_prob = F.softmax(output,dim=1)[0,1].item()
    # Increment probability to map
    prob_array[0,i:i+img_size,j:j+img_size]+=wave_prob
    # Increment scanning index to map
    prob_array[1,i:i+img_size,j:j+img_size]+=1
  probmap = prob_array[0]/prob_array[1]
  if compact:
    prob_array = numpy.array([probmap[i,::sample_stride] for i in range(0,probmap.shape[0],channel_stride)])
    numpy.savetxt('%s.txt'%(os.path.splitext(os.path.basename(data_path))[0]),prob_array,fmt='%s')
  else:
    numpy.savetxt('%s.txt'%(os.path.splitext(os.path.basename(data_path))[0]),probmap,fmt='%s')

def minute_prob_test(hdf5_file,model_file,depth,img_size=200,model_type='resnet',num_channels=3,num_classes=2,multilabel=False):
  """
  Get surface wave probabilities for every consecutive square regions.
  """
  assert num_channels in [1,3], 'Number of channels must be either 1 or 3.'
  assert num_classes<3, 'More than 2 classes not implemented yet.'
  data = hdf5read(os.path.expanduser(hdf5_file))
  model = get_model(model_type,depth=depth,num_channels=num_channels,num_classes=num_classes)
  model = load_model(model_file,model)
  model.eval()
  idxs = numpy.array([[[i,j] for j in range(0,data.shape[1]-img_size+1,img_size)] for i in range(0,data.shape[0]-img_size+1,img_size)])
  idxs = idxs.reshape(idxs.shape[0]*idxs.shape[1],2)
  if num_channels==1:
    imgs = torch.cat(
      [
        transforms.ToTensor()(
          transforms.Grayscale()(
            Image.fromarray(
              numpy.uint8(
                data[i:i+img_size,j:j+200].copy()*255
              )
            )
          )
        ).unsqueeze(0) for i,j in idxs
      ]
    )
  if num_channels==3:
    imgs = torch.cat(
      [
        transforms.ToTensor()(
          Image.fromarray(
            numpy.uint8(
              data[i:i+img_size,j:j+200].copy()*255
            )
          ).convert("RGB")
        ).unsqueeze(0) for i,j in idxs
      ]
    )
  output = model(imgs)
  # print(output)
  # if output.dim()==1:
  #   wave_prob = torch.sigmoid(output).item()
  # elif output.shape[1]==1:
  #   wave_prob = torch.sigmoid(output)[0,0].item()
  # elif multilabel:
  #   wave_prob = torch.sigmoid(output)[0,1].item()
  # else:
  #   wave_prob = F.softmax(output,dim=1)[0,1].item()
  # results.append(wave_prob)
  # numpy.savetxt('%s.txt'%(os.path.splitext(os.path.basename(hdf5_file))[0]),results,fmt='%s')

