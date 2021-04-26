# System
import os
import random

# External
import h5py
import numpy
from PIL import Image
from scipy.fft import fft,fftfreq

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

def find_target(filelist,img_size=200,dir_path='/content'):
  inoise_total, iwaves_total = 0, 0
  for i,fname in enumerate(filelist):
    inoise, iwaves = 0, 0
    filename = fname.split('/')[-1][:-4]
    tmin,tmax = filename.split('_')[2],filename.split('_')[3]
    print('%i/%i - %s'%(i+1,len(filelist),filename))
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
          os.system('cd %s && tar -zcvf %s.tar.gz %s'%(dir_path,filename,filename))
          os.system('mv %s/%s.tar.gz 30min_files/'%(dir_path,filename))
          os.system('rm -rf %s/%s'%(dir_path,filename))
          break
      else:
        iskip = 0
        img = data[i:i+img_size,j:j+img_size].copy()
        label = get_label(img,img_size=img_size)
        if label in ['noise','waves']:
          os.system('mkdir -p %s/%s/%s'%(dir_path,filename,label))
          img = (img-img.min())/(img.max()-img.min())
          img = Image.fromarray(numpy.uint8(img*255))
          if label=='noise': inoise+=1
          if label=='waves': iwaves+=1
          n = inoise if label=='noise' else iwaves
          img.save('%s/%s/%s/%s_%s_%03i_%06i.jpg'%(dir_path,filename,label,tmin,tmax,i,j))
        idxs.append([i,j])
    f.close()
    inoise_total += inoise
    iwaves_total += iwaves
  print('\t%i noise images (Total: %i) | %i waves images (Total: %i)'%(inoise,inoise_total,iwaves,iwaves_total))

def create_set(path,dir_path='/content',imax=174000):
  for label in ['waves','noise']:
    dataset = random.sample(glob.glob('%s/%s/*.jpg'%(path,label)),imax)
    for i,fname in enumerate(dataset):
      set_name = 'train' if i<0.70*imax else 'test' if 0.85*imax<=i else 'validation'
      os.system('mkdir -p %s/%s/%s'%(dir_path,set_name,label))
      os.system('mv %s %s/%s/%s/'%(fname,dir_path,set_name,label))