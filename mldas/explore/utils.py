# Externals
import h5py
import numpy
import scipy.io as sio

def Xcorr2mat(mat,h5Xcorr):
  # Copy data and metadata into dictionary
  dsi_xcorr = {}
  dsi_xcorr['fh']=mat['variable'][0][0]
  dsi_xcorr['th']=mat['variable'][0][1]
  dsi_xcorr['dat']=mat['variable'][0][2]
  # Open file in read mode
  xfile = h5py.File(h5Xcorr,'r')
  # Load cross-correlated HDF5 data
  xdata = xfile.get('xcorr')[:,:].T
  # Close HDF5 file
  xfile.close()
  # Extract sample time from file header
  dt = mat['variable'][0][0][0][7][0][0]
  # Define new sample time
  dt_new = 0.008
  # Estimate new relative samping rate 
  R = round(dt_new/dt)
  # Length of resampledcross-correlated data array
  lres = mat['variable'][0][2][0][0].shape[0]/R
  # Maximum duration of new data
  tmax = round((lres-1)*dt_new,6)
  # Update data and file header in dictionary
  dsi_xcorr['dat'][0][0]=numpy.array(xdata[:,:],dtype=numpy.double)
  dsi_xcorr['fh'][0][6]=len(xdata)
  dsi_xcorr['fh'][0][7]=dt_new
  dsi_xcorr['fh'][0][8]=-tmax
  dsi_xcorr['fh'][0][9]=tmax
  dsi_xcorr['fh'][0][10]=[]
  # Save MAT cross-correlated file
  sio.savemat(h5Xcorr.replace('h5','mat'), {'dsi_xcorr': dsi_xcorr})

def apply_weight(fname,probmap,scaling=False,filtering=False,binary=False,select=[0,1]):
  mat = h5py.File(fname,'r+')
  data = mat[mat['variable/dat'][0,0]]
  prob = numpy.loadtxt(probmap,ndmin=2)
  if scaling:
    prob = 1+99*prob
  if filtering or binary:
    prob[(prob<select[0])|(prob>select[1])]=1e-5
  if binary:
    prob[(select[0]<=prob)&(prob<=select[1])]=1
  channel_stride = data.shape[0]//prob.shape[0]
  sample_stride = data.shape[1]//prob.shape[1]
  for i in range(prob.shape[0]):
    for j in range(prob.shape[1]):
      imin = channel_stride*i
      imax = channel_stride*(i+1)
      jmin = sample_stride*j
      jmax = sample_stride*(j+1)
      mat[mat['variable/dat'][0,0]][imin:imax,jmin:jmax] *= prob[i,j]
  mat.close()
