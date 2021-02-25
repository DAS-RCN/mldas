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

# Externals
import numpy
import scipy
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Local
from .loading import hdf5read

def avg_fft(data,fs=500.):
  N = data.shape[1]
  freqs = numpy.linspace(fs/N,fs/2,N//2)
  ffts = numpy.array([2.0/N*numpy.abs(scipy.fft.fft(ts)[:N//2]) for ts in data])
  fft = numpy.average(ffts,axis=0)
  return freqs, fft
  
def xcorr_freq(data,xcorr,lag_range=500,threshold=5e-2):
  plt.style.use('seaborn')
  fig,ax = plt.subplots(2,2,figsize=(18,8),dpi=80)
  ax[0][0].imshow(abs(data),extent=[0,data.shape[1]/500,200,0],cmap='plasma',aspect='auto',norm=LogNorm())
  ax[0][0].set_title('Raw strain measurements')
  ax[0][0].set_xlabel('Time [sec]')
  ax[0][0].set_ylabel('Channels')
  ax[0][1].imshow(xcorr[:,xcorr.shape[1]//2-lag_range:xcorr.shape[1]//2+lag_range],
                  extent=[-lag_range*59/xcorr.shape[1],lag_range*59/xcorr.shape[1],200,0],
                  aspect ='auto',cmap='seismic',vmin=-threshold,vmax=threshold)
  ax[0][1].set_title('Cross-correlation')
  ax[0][1].set_xlabel('Time lag [sec]')
  ax[1][0].plot(*avg_fft(data))
  ax[1][0].set_title('Average FFT from raw data')
  ax[1][0].set_xlabel('Frequency [Hz]')
  ax[1][0].set_ylabel('Amplitude')
  ax[1][1].plot(*avg_fft(xcorr))
  ax[1][1].set_title('Average FFT from cross-correlated data')
  ax[1][1].set_xlabel('Frequency [Hz]')
  plt.tight_layout()
  plt.show()
