# External
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.offsetbox import AnchoredText

# Local
from .utils import gaussian_fit, freq_content

plt.style.use('seaborn')
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

def full_fiber(data):
  # Initialize the figure
  fig,ax = plt.subplots(figsize=(9,6))
  ax.grid(False)
  # Plot original image
  plt.imshow(abs(numpy.array(data).T),extent=[0,data.shape[0],data.shape[1]/500,0],cmap='inferno',aspect='auto',norm=LogNorm())
  ax.axvline(4650,color='cyan',lw=3)
  ax.axvline(4850,color='cyan',lw=3)
  ax.axvline(5500,color='yellow',lw=3)
  ax.axvline(6000,color='yellow',lw=3)
  ax.xaxis.set_ticks_position('top')
  ax.xaxis.set_label_position('top')
  plt.xlabel('Channels',labelpad=10)
  plt.ylabel('Time [second]')
  plt.colorbar(pad=0.02,orientation="horizontal").set_label('DAS units (proportional to strain rate)')
  plt.tight_layout()
  plt.savefig('abs_data.png')
  plt.show()
  plt.close()

def regions(data1,data2):
  # Initialize figure
  fig,ax = plt.subplots(1,2,figsize=(18,5.5))
  # Plot coherent surface wave patterns
  im = ax[0].imshow(data1,extent=[0,data1.shape[1],200,0],cmap='seismic',aspect='auto',vmin=-1000,vmax=1000,interpolation='bicubic')
  ax[0].xaxis.set_ticks_position('top')
  ax[0].xaxis.set_label_position('top')
  ax[0].set_xlabel('Samples',labelpad=10)
  ax[0].set_ylabel('Channels')
  # Display colorbar
  divider = make_axes_locatable(ax[0])
  cax = divider.append_axes('bottom', size='5%', pad=0.05)
  plt.colorbar(im, pad=0.02, cax=cax, orientation='horizontal').set_label('Raw measurement amplitude')
  # Plot non-coherent signals
  im = ax[1].imshow(data2,extent=[0,data2.shape[1],200,0],cmap='seismic',aspect='auto',vmin=-1000,vmax=1000,interpolation='bicubic')
  ax[1].xaxis.set_ticks_position('top')
  ax[1].xaxis.set_label_position('top')
  ax[1].set_xlabel('Samples',labelpad=10)
  ax[1].set_ylabel('Channels')
  # Display colorbar
  divider = make_axes_locatable(ax[1])
  cax = divider.append_axes('bottom', size='5%', pad=0.05)
  plt.colorbar(im, pad=0.02, cax=cax, orientation='horizontal').set_label('Raw measurement amplitude')
  # Save and show figure
  plt.tight_layout()
  plt.savefig('raw_data.pdf')
  plt.show()
  plt.close()

def plot_dist(data,bins=400,xlim=[-1000,1000]):
  fig,ax = plt.subplots(2,1,figsize=(9,8),sharey=True,sharex=True)
  for i,order in enumerate([1,2]):
    hist = ax[i].hist(data.reshape(numpy.prod(data.shape)),bins=bins,range=xlim,color='white',histtype='stepfilled',edgecolor='black',lw=0.5)
    # Fit double gaussian
    x = numpy.array([0.5 * (hist[1][i] + hist[1][i+1]) for i in range(len(hist[1])-1)])
    y = hist[0]
    x, y, chisq, aic, popt = gaussian_fit(x,y,order)
    if order==1:
      ax[i].plot(x, y[0], lw=2,label='Single-gaussian fit\n$\chi^2_\mathrm{red}=$%.1e / $\mathrm{AIC}=%i$\n$\mu=%.2f, \sigma=%.3f$'%(chisq,aic,popt[1],abs(popt[2])))
    if order==2:
      ax[i].plot(x, y[0], lw=2,label='Double-gaussian fit\n$\chi^2_\mathrm{red}=$%.1e / $\mathrm{AIC}=%i$'%(chisq,aic))
      # Plot first gaussian
      # y = gauss_single(x, *popt[:3])
      ax[i].plot(x, y[1], lw=2,label=r'$\mu=%.2f, \sigma=%.3f$'%(popt[1],abs(popt[2])))
      # Plot second gaussian
      # y = gauss_single(x, *popt[3:])
      ax[i].plot(x, y[2], lw=2,label=r'$\mu=%.2f, \sigma=%.3f$'%(popt[4],abs(popt[5])))
      ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[i].set_xlim(-1000,1000)
    ax[i].legend(loc='upper left')
    ax[i].set_ylabel('Density')
  plt.xlabel('Raw measurement amplitude')
  plt.tight_layout()
  plt.savefig('distribution.pdf')

def plot_freq_content(data,img_size=200,sample_rate=500):
  plt.rc('font', size=12)
  plt.rc('axes', labelsize=12)
  plt.rc('legend', fontsize=12)
  plt.rc('xtick', labelsize=12)
  plt.rc('ytick', labelsize=12)
  fig,ax = plt.subplots(4,4,figsize=(12,12))
  for n,img in enumerate(data):
    ffts, freqs, avg_fft = freq_content(img,img_size,sample_rate)
    img_max = abs(img).max()
    # Show larger image
    ax[0][n].imshow(img,cmap='seismic',extent=[0,img_size,img_size,0],vmin=-img_max,vmax=img_max,interpolation='bicubic')
    ax[0][n].set_xlabel('Sample')
    if n==0: ax[0][n].set_ylabel('Channel')
    # Plotting data distribution
    ax[1][n].hist(img.reshape(numpy.prod(img.shape)),bins=50)
    at = AnchoredText('$\sigma=%i$'%numpy.std(img),prop=dict(size=12),loc='upper left')
    ax[1][n].add_artist(at)
    ax[1][n].set_xlabel('Strain Measurement')
    if n==0: ax[1][n].set_ylabel('Density')
    # D2 and plot FFT for each channel
    ax[2][n].imshow(ffts,extent=[0,sample_rate//2,img.shape[0],0],aspect='auto',norm=LogNorm(vmin=ffts.min(),vmax=ffts.max()),cmap='jet')
    ax[2][n].set_xlabel('Frequency (Hz)')
    if n==0: ax[2][n].set_ylabel('Channels')\
    # Plot average amplitude for each frequency
    ax[3][n].plot(freqs,avg_fft)
    ax[3][n].set_xlabel('Frequency (Hz)')
    ax[3][n].set_xlim(0,sample_rate//2)
    ax[3][n].axvline(40,ls='--',color='black',lw=1.3)
    ax[3][n].set_ylabel('Average Spectral Amplitude')
  plt.tight_layout(h_pad=0,w_pad=0)
  plt.savefig('signal_types.pdf')
  plt.show()

def latent_plot(models,loader):
  fig, ax = plt.subplots(3,2,figsize=(10,12),sharex=True,sharey=True)
  for n,(i,j) in enumerate([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]):
    model_epoch = models[n]
    model_epoch.eval()
    for batch_idx, (data,target) in enumerate(loader):
      data = data.float()
      z, recon_batch, mu, logvar = model_epoch(data.view(-1,numpy.prod(data.shape[-2:])))
      z = z.data.cpu().numpy()
      ax[i][j].scatter(z[:,0],z[:,1],s=10,c=target,cmap='cool',alpha=0.5)
      ax[i][j].set_title('Epoch %i'%(n+1),fontsize=15)
      if i==2: ax[i][j].set_xlabel('Latent variable 1')
      if j==0: ax[i][j].set_ylabel('Latent variable 2')
  plt.tight_layout()
  plt.savefig('clustering.pdf')
  plt.show()