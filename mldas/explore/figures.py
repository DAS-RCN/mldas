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

# Externals
import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Local
from .mapping import extract_prob_map
from .lookup import find_values

def rgb_plot(img):
  plt.style.use('seaborn')
  fig, ax = plt.subplots(1,4,figsize=[16,4],dpi=80)
  ax[0].imshow(img)
  for i_channel in range(3):
    temp = numpy.zeros(img.shape, dtype='uint8')
    temp[:,:,i_channel] = img[:,:,i_channel]
    ax[i_channel+1].imshow(temp)
  plt.tight_layout()
  plt.show()

def model_on_tab(data,NeuralNet,datapath,img_size=50,channel_stride=1,sample_stride=10,overwrite=False):
    """
    Side-by-side plotting of raw measurements and probability map using pre-saved
    trained models. This can be used to visualize how probability maps evolve during
    training. Individual trained models should be saved within the ``datapath``
    directory, along with the loss history (see returned variable from
    :py:class:`mldas.training.suplearn_simple`).

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
      Input raw measurements extracted from HDF5 file
    neuralnet : :py:class:`torch.nn.Module`
      Initialized neural net architecture
    datapath : :py:class:`str`
      Path to saved data
    img_size : :py:class:`int`
      Size of squared image
    channel_stride : :py:class:`int`
      Sliding interval in channel space
    sample_stride : :py:class:`int`
      Sliding interval in sample space
    overwrite : :py:class:`bool`
      Re-calculate probability map
    """
    from google.colab import widgets
    loss_hist = numpy.load('%s/loss_hist.npy'%datapath)
    div = len(loss_hist)
    tb = widgets.TabBar([str(i) for i in range(div)])
    for k in range(div):
      with tb.output_to(k, select=False):
        plot_map_stage(data,NeuralNet,loss_hist,datapath,k,img_size,channel_stride,sample_stride,overwrite)

def plot_prob_map(data,prob,xmin=None,xmax=None,ymin=None,ymax=None,log_scale=False):
  """
  Comparative plots between raw data and corresponding probability map.
  
  Parameters
  ----------
  data : :py:class:`numpy.ndarray`
    Input raw measurements extracted from HDF5 file
  prob : :py:class:`numpy.ndarray`
    Normalized probability map
  log_scale : :py:class:`bool`
    Use logaritmic scale for strain value color scale
  """
  # Use logarithmic scale if requested
  norm = LogNorm() if log_scale else None
  # Convert input raw data to absolute values
  if log_scale: data = abs(data)
  # Specify extent in plot
  if None in [xmin,xmax,ymin,ymax]:
    extent,xlabel = None,'Samples'
  else:
    extent,xlabel = [xmin/500,xmax/500,ymax,ymin],'Time [second]'
  # Plot results
  plt.style.use('seaborn')
  fig,ax = plt.subplots(1,2,figsize=(18,6),dpi=80,sharex=True,sharey=True)
  ax[0].imshow(data,extent=extent,aspect='auto',cmap='inferno',norm=norm)
  ax[0].set_title('Raw strain measurements')
  ax[0].set_xlabel(xlabel)
  ax[0].set_ylabel('Channels')
  ax[1].imshow(prob,extent=extent,aspect='auto',cmap='jet',vmin=0,vmax=1)
  ax[1].set_title('Probability map')
  ax[1].set_xlabel(xlabel)
  plt.tight_layout()
  plt.show()

def plot_map_stage(data,NeuralNet,loss_hist,datapath,k,img_size=50,channel_stride=1,sample_stride=10,overwrite=False,savefig=False):
  """
  Plot stage of probability map.
  
  Parameters
  ----------
  data : :py:class:`numpy.ndarray`
    Input raw measurements extracted from HDF5 file
  NeuralNet : :py:class:`torch.nn.Module`
    Initialized neural net architecture
  loss_hist : :py:class:`numpy.ndarray`
    History of loss values
  datapath : :py:class:`str`
    Path to saved data
  k : :py:class:`int`
    Index from loss/model history.
  img_size : :py:class:`int`
    Size of squared image
  channel_stride : :py:class:`int`
    Sliding interval in channel space
  sample_stride : :py:class:`int`
    Sliding interval in sample space
  overwrite : :py:class:`bool`
    Re-calculate probability map  
  savefig : :py:class:`bool`
    Save figure as PNG file
  """
  model = NeuralNet
  state_dict = torch.load('%s/model%02i.pt'%(datapath,k))
  model.load_state_dict(state_dict)
  model.eval()
  if os.path.exists('%s/model%02i.npy'%(datapath,k)) and overwrite==False:
    prob_map = numpy.load('%s/model%02i.npy'%(datapath,k))
  else:
    prob_map = extract_prob_map(data,model,img_size,channel_stride,sample_stride)
    numpy.save('%s/model%02i'%(datapath,k),prob_map)
  plt.style.use('seaborn')
  fig,ax = plt.subplots(2,2,figsize=(18,10),dpi=80)
  ax[0][0].plot(loss_hist[:,3]/loss_hist[:,2],label='Training loss')
  ax[0][0].plot(loss_hist[:,6]/loss_hist[:,5],label='Validation loss')
  ax[0][0].axvline(k,color='black',lw=0.8)
  ax[0][0].scatter([k,k],[loss_hist[k,3]/loss_hist[k,2],loss_hist[k,6]/loss_hist[k,5]],color='black',zorder=3)
  ax[0][0].set_xlabel('Batch Iterations')
  ax[0][0].set_ylabel('Loss')
  ax[0][0].legend(frameon=False)
  ax[0][1].plot(100*loss_hist[:,4]/loss_hist[:,2],label='Training accuracy')
  ax[0][1].plot(100*loss_hist[:,7]/loss_hist[:,5],label='Validation accuracy')
  ax[0][1].axvline(k,color='black',lw=0.8)
  ax[0][1].scatter([k,k],[100*loss_hist[k,4]/loss_hist[k,2],100*loss_hist[k,7]/loss_hist[k,5]],color='black',zorder=3)
  ax[0][1].set_xlabel('Batch Iterations')
  ax[0][1].set_ylabel('Accuracy')
  ax[0][1].legend(frameon=False)
  ax[1][0].imshow(data,aspect='auto',cmap='inferno')
  ax[1][0].set_title('Raw strain measurements')
  ax[1][0].set_xlabel('Samples')
  ax[1][0].set_ylabel('Channels')
  ax[1][1].imshow(prob_map,aspect='auto',cmap='jet',vmin=0,vmax=1)
  ax[1][1].set_title('Probability map')
  ax[1][1].set_xlabel('Samples')
  plt.tight_layout()
  if savefig:
    plt.savefig('%s/model%02i.png'%(datapath,k),dpi=200)
    plt.close()
  else:
    plt.show()

def plot_loss(input_data,log=False,**kwargs):
  train_loss, train_acc, valid_loss, valid_acc = find_values(input_data[0])
  plt.style.use('seaborn')
  fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,5),dpi=100,sharex=True)
  if log:
    ax1.semilogy(train_loss,label='Training')
    ax1.semilogy(valid_loss,label='Validation')
  else:
    ax1.plot(train_loss,label='Training')
    ax1.plot(valid_loss,label='Validation')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.legend()
  ax2.plot(train_acc,label='Training')
  ax2.plot(valid_acc,label='Validation')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy Percentage')
  ax2.set_ylim([-5,105])
  ax2.legend()
  plt.tight_layout()
  plt.savefig('plot_loss.png')
  plt.close()
