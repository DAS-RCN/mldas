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
import yaml

# Externals
import numpy
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

# Local
from .mapping import extract_prob_map
from .lookup import find_values
from ..models import mlp
from ..datasets import fwi

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

def plot_fwi_mlp(input_data,config,vmax=3040,ymax=599,**kwargs):
  # Load yaml file
  with open(config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  # Load validation dataset
  train_loader, valid_loader, test_loader = fwi.get_data_loaders(config['output_dir'],**config['data_config'])
  target_data, label_data = next(iter(train_loader))
  input_size = target_data.reshape(target_data.shape[0],-1).shape[1]
  output_size = label_data.reshape(label_data.shape[0],-1).shape[1]
  # Load model and saved parameters
  model = mlp.MLP([input_size]+config['model_config']['n_layer']+[output_size])
  checkpoint = torch.load(input_data[0],map_location=lambda storage, loc: storage)
  model.load_state_dict(checkpoint['model'])
  model.eval()
  # Make
  plt.style.use('seaborn')
  plt.figure(figsize=(8,16),dpi=200)
  for targets,labels in train_loader:
    outs = model(targets.float())
    for i,(label,out) in enumerate(zip(labels,outs)):
      if i==5: break
      if config['data_config']['n_dims']==1:
        label = label.reshape(-1,2).detach().numpy()
        out = out.reshape(-1,2).detach().numpy()
        ax = plt.subplot(3,5,i+1)
        #ax.step(exp[:,0],exp[:,1],color='blue',lw=1,where='post',zorder=1,ls='dotted')
        #ax.step(out[:,0],out[:,1],color='red',lw=1,where='post',zorder=2,ls='dotted')
        ax.scatter(label[:,0]*vmax,label[:,1]*ymax,color='white',edgecolors='blue',lw=1,s=10,label='True',zorder=3)
        ax.scatter(out[:,0]*vmax,out[:,1]*ymax,color='white',edgecolors='red',lw=5,s=10,label='ML',zorder=4)
        #if i>4: ax.set_xlabel('V$_\mathrm{S}$ (m/s)')
        #if i%5==0: ax.set_ylabel('Depth (m)')
        #if i==0: ax.legend(bbox_to_anchor=(-0.2,1))
        #ax.xlim(0,vmax)
        #ax.ylim(0,ymax)
        ax.invert_yaxis()
      else:
        cmap = 'gist_ncar'
        length = int(numpy.sqrt(label.shape[0]))
        label = label.reshape(length,length).detach().numpy()
        out = out.reshape(length,length).detach().numpy()
        ax = plt.subplot(5,2,i*2+1)
        ax.imshow(label,extent=[0,1,ymax,0],cmap=cmap,vmin=0,vmax=1,aspect='auto')
        ax.get_xaxis().set_visible(False)
        ax = plt.subplot(5,2,i*2+2)
        ax.imshow(out,extent=[0,1,ymax,0],cmap=cmap,vmin=0,vmax=1,aspect='auto')#,interpolation='bicubic')
        ax.get_xaxis().set_visible(False)
    break
  plt.tight_layout()
  plt.savefig('learning')
  plt.close()

def plot_test_2d(input_data,output_data,fname='learning',vmax=3040,ymax=599):
  cmap = 'gist_ncar'
  length = int(numpy.sqrt(input_data.shape[0]))
  plt.style.use('seaborn')
  plt.figure(figsize=(8,4),dpi=200)
  for i,data in enumerate([input_data,output_data]):
    ax = plt.axes([0.1 if i==0 else 0.5, 0.1, 0.39, 0.8])
    data = data.reshape(length,length).detach().numpy()
    im = ax.imshow(data,extent=[0,1,ymax,0],cmap=cmap,vmin=0,vmax=1,aspect='auto')
    ax.get_xaxis().set_ticks([])
    ax.set_xlabel('Receiver location')
    if i==0:
      ax.set_ylabel('Depth (m)')
    else:
      ax.yaxis.set_ticklabels([])
  cax = plt.axes([0.9, 0.1, 0.02, 0.8])
  cbar = plt.colorbar(im,cax=cax).set_label('Velocity (m/s)')
  #for t, y in zip( cbar.get_ticklabels( ), cbar.get_ticks( ) ):
  #  t.set_y( y*vmax )
  #plt.tight_layout()
  plt.savefig(fname)
  plt.close()

def plot_params(input_data,region,**kwargs):
  """
  Example
  -------
  >>> das_quickrun.py plot_params -i ML-mkshots-fcheng/info.txt
  >>> das_quickrun.py plot_params -i ML-mkshots-fcheng/info.txt -r 0.5
  """
  data = numpy.loadtxt(input_data[0])
  new_data = numpy.empty((0,3))
  x, y = data[:,2], data[:,1]
  print('Mean and Standard Deviation:')
  print('Velocity ..',numpy.mean(x),numpy.std(x))
  print('Depth .....',numpy.mean(y),numpy.std(y))
  #for y, x in data[:,1:]:
  #  match = False
  #  for i in range(len(new_data)):
  #    if x==new_data[i,0] and y==new_data[i,1]:
  #      new_data[i,2]+=1
  #      match = True
  #  if match==False:
  #    new_data = numpy.vstack((new_data,[x,y,1]))
  #x,y,c = new_data[:,0], new_data[:,1], new_data[:,2]
  plt.style.use('seaborn')
  fig = plt.figure(figsize=(10,10),dpi=200)
  gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                        left=0.07, right=0.97, bottom=0.07, top=0.97,
                        wspace=0.05, hspace=0.05)
  ax = fig.add_subplot(gs[1, 0])
  ax.set_xlim(min(x),max(x))
  ax.set_ylim(min(y),max(y))
  ax.set_xlabel('Maximum Velocity [m/s]')
  ax.set_ylabel('Maximum Depth [m]')
  ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
  ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
  ax_histx.tick_params(axis="x", labelbottom=False)
  ax_histy.tick_params(axis="y", labelleft=False)
  #ax.scatter(x,y,c=c,s=7,cmap="magma")
  ax.scatter(x,y,s=5)
  if region!=None:
    rect = patches.Rectangle((numpy.mean(x)-region*numpy.std(x),
                              numpy.mean(y)-region*numpy.std(y)),
                             2*region*numpy.std(x),
                             2*region*numpy.std(y), linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    n=0
    for v,z in zip(x,y):
      if numpy.mean(x)-region*numpy.std(x)<v<numpy.mean(x)+region*numpy.std(x) and \
         numpy.mean(y)-region*numpy.std(y)<z<numpy.mean(y)+region*numpy.std(y):
        n+=1
    print(n,'models found within range.')
  ax_histx.hist(x, bins=50)
  ax_histy.hist(y, bins=50, orientation='horizontal')
  plt.savefig('params')
  plt.close()
