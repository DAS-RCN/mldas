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

import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss_single(x, a, b, c):
    return a * numpy.exp(-(x - b)**2.0 / (2 * c**2))

def gauss_double(x, a, b, c, d, e, f):
    return a * numpy.exp(-(x - b)**2.0 / (2 * c**2)) + d * numpy.exp(-(x - e)**2.0 / (2 * f**2))

def distfit(ax,data,ngauss,amp=1e5,mu=0.,sig=100,bins=400,xmin=-1000,xmax=1000):
    """
    Fit gaussian to 2D patch distribution.
    """
    hist = ax.hist(data.reshape(numpy.prod(data.shape)),bins=bins,range=[xmin,xmax])
    # Fit double gaussian
    x = numpy.array([0.5 * (hist[1][i] + hist[1][i+1]) for i in range(len(hist[1])-1)])
    y = hist[0]
    fct = gauss_single if ngauss==1 else gauss_double
    popt, pcov = curve_fit(fct,x,y,p0=[amp,mu,sig]*ngauss)
    # Calculate chi-square
    r = (y - fct(x, *popt))
    sigma = numpy.sqrt(numpy.diag(pcov))
    chisq = numpy.sum((r/numpy.average(sigma))**2)
    df = len(x) - len(popt)
    # Plot double gaussian
    x = numpy.arange(xmin,xmax,0.001)
    y = fct(x, *popt)
    if ngauss==1:
        ax.plot(x, y, lw=2,color='salmon',label='Single-gaussian fit $\chi^2=%.4f$\n$\mu=%.2f, \sigma=%.3f$'%(chisq/df,popt[1],abs(popt[2])))
    if ngauss==2:
        ax.plot(x, y, lw=2,color='salmon',label='Double-gaussian fit $\chi^2=%.4f$'%(chisq/df))
        # Plot first gaussian
        y = gauss_single(x, *popt[:3])
        ax.plot(x, y, lw=2,color='yellow',label=r'$\mu=%.2f, \sigma=%.3f$'%(popt[1],abs(popt[2])))
        # Plot second gaussian
        y = gauss_single(x, *popt[3:]) 
        ax.plot(x, y, lw=2,color='palegreen',label=r'$\mu=%.2f, \sigma=%.3f$'%(popt[4],abs(popt[5])))
    ax.set_xlim(xmin,xmax)
    ax.legend(loc='best')
