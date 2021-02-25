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

# Internals
import os
import re

#Externals
import h5py
import numpy
import hdf5storage
import matplotlib.pyplot as plt

def phase_weight_stack(fpath,idx,weight=False,path=None):
    fname = re.split('[/.]',fpath)[-2]
    xcorr = fname.replace('westSac','1minXcorr')
    # Copy raw data to temporary HDF5 file
    mat = hdf5storage.loadmat(fpath)
    data = mat['variable'][0][2][0,0].T
    if weight:
        assert os.exists(path), 'Path to probability maps not found.'
        data = ts_weighting(fname,data,path)
    f = h5py.File(fname+'.h5','w')
    f.create_dataset("DataTimeChannel",data=data,dtype="i2")
    f.close()
    # Execute cross-correlation using ArrayUDF
    os.system('mpirun --allow-run-as-root -n 1 /content/ArrayUDF/examples/das/das-fft-full -i %s.h5 -o %s.h5 -g / -t /DataTimeChannel -x /Xcorr'%(fname,xcorr))
    # Convert cross-correlated data from h5 to mat format
    Xcorr2mat(mat,xcorr+'.h5')
    # Move mat file to xcorr2stack folder
    os.makedirs('xcorr2stack', exist_ok=True)
    os.system('mv *.mat xcorr2stack && rm *.h5')
    # Execute stacking
    os.makedirs('stack_files', exist_ok=True)
    os.system('octave -W /content/SCRIPT_run_Stacking.m > out && rm out')
    # Rename stacking stage results
    os.system('mv stack_files/Dsi_mstack.mat stack_files/Dsi_mstack_nstack%s.mat'%idx)
    os.system('mv stack_files/Dsi_pwstack.mat stack_files/Dsi_pwstack_nstack%s.mat'%idx)

def ts_weighting(fname,data,map_path):
  probs = numpy.loadtxt('%s/%s.txt'%(map_path,fname))
  if len(probs.shape)==1:
    probs = numpy.array([[[prob]*(data.shape[1]//len(probs)) for prob in probs] \
                         for i in range(data.shape[0])]).reshape(data.shape)
  return data * probs

def comp_stack(file_list,output='./'):
    os.makedirs(output, exist_ok=True)
    plt.style.use('seaborn')
    fig,ax = plt.subplots(3,3,figsize=(20,15),dpi=80)
    all_data = []
    for i,fname in enumerate(file_list):
        split = re.split('[/.]',fname)
        f = h5py.File(fname,'r')
        data = numpy.array(f['data'])
        f.close()
        threshold = 5e-2
        imin = data.shape[1]//2-500
        imax = data.shape[1]//2+500
        ax[i,0].imshow(data[:,imin:imax],aspect='auto',cmap='seismic',vmin=-threshold,vmax=threshold)
        ax[i,0].set_title('/'.join(split[-4:]))
        all_data.append(data[:,imin:imax])
    ax[1,1].imshow(all_data[1]-all_data[0],aspect='auto',cmap='seismic')
    ax[2,1].imshow(all_data[2]-all_data[0],aspect='auto',cmap='seismic')
    ax[2,2].imshow(all_data[2]-all_data[1],aspect='auto',cmap='seismic')
    plt.tight_layout()
    plt.savefig(output+'/'+split[-2])
