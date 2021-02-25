# Internals
import os
import re
import glob

# Externals
import h5py
import yaml
import numpy
import scipy
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

# Locals
from .utils import apply_weight,Xcorr2mat
from .mapping import minute_prob

def xcorr_convert(target,output,flag):
  os.makedirs(output, exist_ok=True)
  mat = hdf5storage.loadmat(target[0])
  fname = re.split('[/.]',target[0])[-2]
  if flag=='pre':
    data = mat['variable'][0][2][0,0].T
    f = h5py.File(output+'/'+fname+'.h5','w')
    f.create_dataset("data",data=data,dtype="i2")
    f.close()
  if flag=='post':
    Xcorr2mat(mat,output+'/'+fname+'_corr.h5')
  
def make_alias(target,output,flag):
    os.makedirs(output, exist_ok=True)
    os.chdir(output)
    for fname in glob.glob(target[0]):
        os.system('ln -s %s'%fname)
        
def probmap(target,output,flag):
    targets = target[0].split(':')
    with open(targets[0]) as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
    if len(targets)>1:
      config['data_path']=targets[1]
    os.makedirs(output, exist_ok=True)
    os.chdir(output)
    for fname in sorted(glob.glob(config['data_path'])):
      print(fname)
      config['data_path'] = fname
      minute_prob(**config)
    
def weighting(target,output,flag):
    scale = 'scale' in flag
    filtering = 'filter' in flag
    binary = 'binary' in flag
    select = [0,1]
    for flag in flag.split(','):
        threshold = flag.replace('+','').replace('-','')
        if '+' in flag: select[0] = float(threshold)
        if '-' in flag: select[1] = float(threshold)
    os.makedirs(output, exist_ok=True)
    if target[0].endswith('.mat') and target[1].endswith('.txt'):
        print(target[0])
        name = re.split('[/.]',target[0])[-2]
        assert name==re.split('[/.]',target[1])[-2], "Data and map filenames don't match. Abort."
        os.system('cp %s %s'%(target[0],output))
        apply_weight('%s/%s.mat'%(output,name), target[1], scale, filtering, binary, select)
    else:
        for fname in sorted(glob.glob(target[0]+'/*.mat')):
            print(fname)
            name = re.split('[/.]',fname)[-2]
            os.system('cp %s %s'%(fname,output))
            apply_weight('%s/%s.mat'%(output,name), '%s/%s.txt'%(target[1],name), scale, filtering, binary, select)

def get_single_prob(target,output,flag):
    results = numpy.empty((0,2))
    all_files = glob.glob(target[0]+'/*.txt')
    for file_path in sorted(all_files):
        if 'sum' in flag:
          file_prob = numpy.sum(numpy.loadtxt(file_path))
        else:
          file_prob = numpy.average(numpy.loadtxt(file_path))
        results = numpy.vstack((results,[os.path.basename(file_path),file_prob]))
    numpy.savetxt(output,results,fmt='%s')

def prob_plot(target,output,flag):
    for fname in target[1:]:
        print(fname)
        split = re.split('[/.]',fname)
        f = h5py.File(fname,'r')
        data = f[f['dsi30/dat'][0,0]]
        probs = target[0]+'/'+split[-2]+'.txt'
        probmap = numpy.loadtxt(probs,dtype=numpy.float16)
        full_map = numpy.zeros((500,930000))
        for i in range(probmap.shape[0]):
            for j in range(probmap.shape[1]):
                full_map[100*i:100*(i+1),200*j:200*(j+1)] = probmap[i,j]
        plt.style.use('seaborn')
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,14),dpi=80)
        plt.title(split[-2])
        pos = ax1.imshow(data,aspect='auto',cmap='seismic',vmin=-10,vmax=10)
        fig.colorbar(pos, ax=ax1)
        pos = ax2.imshow(full_map,aspect='auto',cmap='jet',vmin=0,vmax=1)
        fig.colorbar(pos, ax=ax2)
        plt.tight_layout()
        plt.savefig(output+'/'+split[-2])
        f.close()

def single_prob(target,output,flag):
    results = numpy.empty((0,2))
    all_files = glob.glob(target[0]+'/*.txt')
    for file_path in sorted(all_files):
        file_prob = numpy.average(numpy.loadtxt(file_path))
        results = numpy.vstack((results,[os.path.basename(file_path),file_prob]))
    numpy.savetxt(output+'/probmaps.out',results,fmt='%s')
    
def down_select(target,output,flag):
    data = numpy.loadtxt(target,dtype=str)
    data = data[numpy.argsort(data[:,1])][:300]
    data = data[numpy.argsort(data[:,0])]
    numpy.savetxt('/global/scratch/vdumont/probmaps_600.txt',data,fmt='%s')
    for fname in data[:,0]:
        os.system('cp %s/%s.mat mat/'%(output,fname.split('.')[0]))

def raw_xcorr(target,output,flag):
    file_list = numpy.loadtxt(target[0],dtype=str)
    plt.style.use('seaborn')
    fig,ax = plt.subplots(4,2,figsize=(10,10),dpi=80,sharey='row')
    for n,i in enumerate([0,4]):
        # Plot example raw data
        f = h5py.File(file_list[i+0],'r')
        data = numpy.array(f['data'])
        im = ax[0,n].imshow(abs(data),extent=[0,data.shape[1]/500,data.shape[0],0],
                            cmap='plasma',aspect='auto',norm=LogNorm())
        #fig.colorbar(im, ax=ax[0], pad=0.01)
        ax[0,n].set_xlabel('Time [seconds]')
        if n==0: ax[0,n].set_ylabel('Channels')
        f.close()
        # Plot corresponding cross-correlation
        f = h5py.File(file_list[i+1],'r')
        data = numpy.array(f['Xcorr'])
        cent = data.shape[1]//2
        im = ax[1,n].imshow(data[:,cent-500:cent+500],extent=[-0.992,0.992,data.shape[0],0],
                            cmap='seismic',aspect='auto',vmin=-1e-2,vmax=1e-2)
        #fig.colorbar(im, ax=ax[1], pad=0.01)
        ax[1,n].set_xlabel('Time lag [seconds]')
        if n==0: ax[1,n].set_ylabel('Channels')
        f.close()
        # Plot stacked cross-correlated data
        f = h5py.File(file_list[i+2],'r')
        data = numpy.array(f['data'])
        cent = data.shape[1]//2
        im = ax[2,n].imshow(data[:,cent-500:cent+500],extent=[-0.992,0.992,data.shape[0],0],
                            cmap='seismic',aspect='auto',vmin=-1e-2,vmax=1e-2)
        #fig.colorbar(im, ax=ax[2], pad=0.01)
        ax[2,n].set_xlabel('Time lag [seconds]')
        if n==0: ax[2,n].set_ylabel('Channels')
        f.close()  
        # Plot stacked cross-correlated data
        data = hdf5storage.loadmat(file_list[i+3])
        data = data['rmsd'][0][1:]
        ax[3,n].plot(2+numpy.arange(len(data)),data)
        ax[3,n].set_xlabel('nstacks')
        if n==0: ax[3,n].set_ylabel('RMSD')
        f.close()  
    plt.tight_layout()
    plt.savefig(output+'/raw_xcorr')

def map_to_xcorr(target,output,flag):
  file_list = numpy.loadtxt(target[0],dtype=str)
  plt.style.use('seaborn')
  ncols = len(file_list)//4
  fig,ax = plt.subplots(5,ncols,figsize=(3*ncols,15),dpi=80,sharey='row')
  for n,i in enumerate(range(0,4*ncols,4)):
    # Plot example raw data
    print(file_list[i+0])
    if os.path.exists(file_list[i+0]):
      f = h5py.File(file_list[i+0],'r')
      data = numpy.array(f[f['variable/dat'][0,0]])
      im = ax[0,n].imshow(abs(data),extent=[0,data.shape[1]/500,data.shape[0],0],
                          cmap='jet',aspect='auto',norm=LogNorm(vmax=100))
      ax[0,n].set_xlabel('Time [seconds]')
      ax[0,n].set_title('/'.join(file_list[i+0].split('/')[1:3]).replace('/','\n'))
      if n==0: ax[0,n].set_ylabel('Channels')
      f.close()
    # Plot single cross-correlation data
    print(file_list[i+1])
    if os.path.exists(file_list[i+1]):
      if file_list[i+1].endswith('.mat'):
        data = hdf5storage.loadmat(file_list[i+1])['dsi_xcorr'][0,0][2][0,0].T
      else:
        f = h5py.File(file_list[i+1],'r')
        data = f['xcorr']
      cent = data.shape[1]//2
      im = ax[1,n].imshow(data[:,cent-500:cent+500],extent=[-0.992,0.992,data.shape[0],0],
                          cmap='seismic',aspect='auto',vmin=-1e-2,vmax=1e-2)
      ax[1,n].set_xlabel('Time lag [seconds]')
      if n==0: ax[1,n].set_ylabel('Channels')
    # Plot stacked cross-correlated data
    print(file_list[i+2])
    if os.path.exists(file_list[i+2]):
      if file_list[i+1].endswith('.mat'):
        data = hdf5storage.loadmat(file_list[i+2])['Dsi_pwstack'][0,0][2][0,0].T
      else:
        f = h5py.File(file_list[i+2],'r')
        data = numpy.array(f['data'])
      cent = data.shape[1]//2
      im = ax[2,n].imshow(data[:,cent-500:cent+500],extent=[-0.992,0.992,data.shape[0],0],
                          cmap='seismic',aspect='auto',vmin=-1e-2,vmax=1e-2)
      ax[2,n].set_xlabel('Time lag [seconds]')
      if n==0: ax[2,n].set_ylabel('Channels')
      f.close()
      # Plot average frequency content
      ffts = numpy.array([2.0/len(ts)*numpy.abs(scipy.fft.fft(ts)[:len(ts)//2]) for ts in data])

      ffts = numpy.average(ffts,axis=0)
      freqs = scipy.fft.fftfreq(len(scipy.fft.fft(data[0])),d=1/500.)[:len(scipy.fft.fft(data[0]))//2]
      ax[3,n].plot(freqs,ffts)
      ax[3,n].set_xlabel('Frequency [Hertz]')
      if n==0: ax[3,n].set_ylabel('Average amplitude')
    # Plot stacked cross-correlated data
    print(file_list[i+3])
    if os.path.exists(file_list[i+3]):
      data = hdf5storage.loadmat(file_list[i+3])
      data = data['rmsd'][0][1:]
      ax[4,n].plot(2+numpy.arange(len(data)),data)
      if n==0:
        ref = data
      else:
        ax[4,n].plot(2+numpy.arange(len(data)),ref,color='red')
        ax[4,n].set_xlabel('nstacks')
      if n==0: ax[4,n].set_ylabel('RMSD')
      f.close()
  plt.tight_layout()
  plt.savefig(output+'/map_to_xcorr')

def all_plot(target,output,flag):
  pdf_pages = PdfPages(output+'/all_plot.pdf')
  for fname in sorted(glob.glob(target[0])):
    print(fname)
    fig = plt.figure(figsize=(12,8),dpi=80)
    if 'Dsi_mstack' in target[0]:
      key = 'Dsi_mstack'
    if 'Dsi_pwstack' in target[0]:
      key = 'Dsi_pwstack'
    if 'xcorr' in target[0]:
      key = 'dsi_xcorr'
    data = hdf5storage.loadmat(fname)[key][0,0][2][0,0].T
    cent = data.shape[1]//2
    plt.title(fname)
    plt.imshow(data[:,cent-500:cent+500],extent=[-0.992,0.992,data.shape[0],0],
               cmap='seismic',aspect='auto',vmin=-1e-2,vmax=1e-2)
    plt.colorbar()
    plt.tight_layout()
    pdf_pages.savefig(fig)
  pdf_pages.close()
