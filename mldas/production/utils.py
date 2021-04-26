# External
import math
import numpy
from scipy.optimize import curve_fit
from scipy.fft import fft,fftfreq

def gauss_single(x, amp, cen, wid):
  return amp * numpy.exp(-(x - cen)**2.0 / (2 * wid**2))

def gauss_double(x, a, b, c, d, e, f):
  return a * numpy.exp(-(x - b)**2.0 / (2 * c**2)) + d * numpy.exp(-(x - e)**2.0 / (2 * f**2))

def gaussian_fit(x,y,order):
  fct = {1:gauss_single,2:gauss_double}[order]
  popt, pcov = curve_fit(fct,x,y,p0=[1e5,0,100]*order)
  # Calculate residual sum squares
  res = (y - fct(x, *popt))
  chisq = numpy.sum(res**2)
  aic = 2*len(popt)+len(x)*math.log(chisq/len(x))
  df = len(x) - len(popt)
  # Plot double gaussian
  x = numpy.arange(-1000,1000,0.001)
  y = fct(x, *popt)
  y = numpy.vstack((y,[gauss_single(x, *popt[i:i+3]) for i in range(0,len(popt),3)]))
  return x,y,chisq/df,aic,popt

def freq_content(data,img_size=200,sample_rate=500):
  ffts = numpy.array([2.0/len(time_serie)*numpy.abs(fft(time_serie)[:len(time_serie)//2]) for time_serie in data])
  freqs = fftfreq(img_size,d=1/sample_rate)[:img_size//2]
  avg_fft = [numpy.average(freq) for freq in ffts.T]
  return ffts,freqs,avg_fft
