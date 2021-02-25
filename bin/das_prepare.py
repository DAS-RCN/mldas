#!/usr/bin/env python

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

import numpy,glob,random,os,h5py,sys,argparse
from numpy.fft import fft,fftfreq
from PIL import Image

def get_label(data,xwave=2,xhigh=1000,img_size=200,sample_rate=500):
  ffts = numpy.array([2.0/len(time_serie)*numpy.abs(fft(time_serie)[:len(time_serie)//2]) for time_serie in data])
  # freq = numpy.linspace(0,sample_rate/2,img_size/2,endpoint=False)
  freq = fftfreq(img_size,d=1/sample_rate)[:img_size//2]
  avg1 = [numpy.average(freq) for freq in ffts.T]
  label = None
  if numpy.std(data)>xhigh:
    label = 'high'
  elif max(avg1[:len(avg1)//6])<min(avg1[len(avg1)//6:2*len(avg1)//6]):
    label = 'noise'
  elif max(avg1[:len(avg1)//5])>xwave*numpy.average(avg1[len(avg1)//5:]):
    label = 'waves'
  return label

def find_target(fname):
  filename = fname.split('/')[-1][:-4]
  img_size = 200
  tmin,tmax = filename.split('_')[2],filename.split('_')[3]
  f = h5py.File(fname,'r')
  for key in list(f.keys()):
    if key in ['dsi30','data']:
      data = f[f.get('%s/dat'%key)[0,0]]
      break
  idxs, iskip = [], 0
  while True:
    i = random.randint(0,data.shape[0]-img_size)
    j = random.randint(0,data.shape[1]-img_size)
    if any([(x<=i<=x+img_size and y<=j<=y+img_size) for (x,y) in idxs]):
      iskip+=1
      if iskip==100:
        break
    else:
      iskip = 0
      img = data[i:i+img_size,j:j+img_size].copy()
      label = get_label(img,img_size=img_size)
      if label in ['noise','waves']:
        os.system('mkdir -p /global/cscratch1/sd/vdumont/30min/%s/%s'%(filename,label))
        img = (img-img.min())/(img.max()-img.min())
        img = Image.fromarray(numpy.uint8(img*255))
        img.save('/global/cscratch1/sd/vdumont/30min/%s/%s/%s_%s_%03i_%06i.jpg'%(filename,label,tmin,tmax,i,j))
      idxs.append([i,j])
  f.close()  

def set_creation(fname,sizes=[],dest='/global/cscratch1/sd/vdumont/'):
  assert len(sizes)>0, 'Size of training set not specified. Abort.'
  print('List images in %s...'%fname)
  all_noise = glob.glob(fname+'*/noise/*.jpg')
  all_waves = glob.glob(fname+'*/waves/*.jpg')
  print('Done!')
  for size in sizes:
    true_size = 1000*size
    noise = random.sample(all_noise,true_size)
    waves = random.sample(all_waves,true_size)
    for i in range(true_size):
      set_name = 'train' if i<0.70*true_size else 'test' if 0.85*true_size<=i else 'validation'
      for label in ['noise','waves']:
        fpath = noise[i] if label=='noise' else waves[i]
        dpath = '%s/set_%ik_200x200_class2/%s/%s'%(dest,size,set_name,label)
        os.makedirs(dpath, exist_ok=True)
        os.system('ln -s %s %s/%s'%(fpath,dpath,os.path.basename(fpath)))
  
if __name__ == "__main__":

  parser = argparse.ArgumentParser('prepare.py')
  add_arg = parser.add_argument
  add_arg('fname', default='Input MAT file or folder')
  add_arg('-s','--size',type=int,default=[],nargs='*',help='Size of training dataset (in thousands)')
  args = parser.parse_args()
  if args.fname.endswith('.mat'):
    find_target(args.fname)
  else:
    set_creation(args.fname,args.size)
