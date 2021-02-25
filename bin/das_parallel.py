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

# System
import os
import re
import glob
import argparse

# Externals
import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

# Local
from explore import colormap as custom
from explore import lookup

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-a', '--accuracy', action='store_true',
            help='Accuracy normalization')
    add_arg('-c', '--colorbar', action='store_true',
            help='User color bar')
    add_arg('-i', '--interpolation', action='store_true',
            help='Use interpolation')
    add_arg('-e', '--equidistant', action='store_true',
            help='Put ticks in equidistance')
    add_arg('-p', '--padding', type=float, default=0.2,
            help='Set padding')
    add_arg('-r', '--repository', default='/global/cscratch1/sd/vdumont/',
            help='Specify data repository')
    return parser.parse_args()

def get_file_results(path,equidistant=True):
    dict = {'processor':4,'nodes':5,'dataset_size':6,'channels':3,
            'batch_size':7,'depth':9,'neurons':2,'learning_rate':10}
    proc_dict = {'hsw':0,'gpu':1}
    data = numpy.empty((0,10),dtype=object)
    data = numpy.vstack((data,[key for key in dict.keys()]+['epochs','accuracy']))
    for outfile in glob.glob(path+'/*/*/*/*'):
        if os.path.exists('%s/out_0.log'%outfile)==False: continue
        train_loss, train_acc, valid_loss, valid_acc = lookup.find_values('%s/out_0.log'%outfile)
        if len(valid_acc)==0 or min(valid_acc)<45: continue
        o = '/'.join(outfile.split('/')[-4:])
        m = re.match('(\w+)/(\d+)-neuron/(\d+)-channel/(\w+)-n(\d+)-ds(\d+)-bs(\d+)-ep(\d+)-dp(\d+)-lr(\d+\.\d+)', o)
        for i, acc in enumerate(valid_acc):
            data = numpy.vstack((data,[m.group(dict[key]) for key in dict.keys()]+[i+1,acc]))
            if o=='multiclass/2-neuron/1-channel/gpu-n4-ds1-bs128-ep40-dp14-lr0.001' and i==25:
                idx = len(data)-2
    data[1:,0] = (data[1:,0]=='gpu').astype('int8')
    data = data[:,[0,1,2,3,4,5,6,8,7,9]]
    numpy.savetxt('parallel_coordinates_raw.csv', data, delimiter=',', fmt='%s')
    data[1:] = data[1:].astype('float64')
    dict = {}
    for i in range(data.shape[1]-1):
        if data[0,i]!='epochs':
            vals = sorted([float(k) for k in numpy.unique(data[1:,i])])
            dict[data[0,i]] = vals
            if equidistant or data[0,i]=='channels':
                for n,val in enumerate(vals):
                    data[1:,i] = numpy.where(data[1:,i]==val, n, data[1:,i])
    numpy.savetxt('parallel_coordinates.csv', data, delimiter=',', fmt='%s')
    return data[0], dict, idx

def normalize(df, cols, acc_norm):
    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        if numpy.ptp(df[col])==0: 
            min_max_range[col] = [0,1,0]
            df[col] = numpy.true_divide(df[col], df[col])
        elif col=='accuracy' and acc_norm:
            min_max_range[col] = [0, 100., 100.]
            df[col] = numpy.true_divide(df[col], 100.)
        elif col in ['processor','channels']:
            min_max_range[col] = [0, 1., 1.]
            df[col] = 0.25+numpy.true_divide(df[col], 2.)
        else:
            min_max_range[col] = [df[col].min(), df[col].max(), numpy.ptp(df[col])]
            df[col] = numpy.true_divide(df[col] - df[col].min(), numpy.ptp(df[col]))
    return min_max_range, df

# Set the tick positions and labels on y axis for each plot
# Tick positions based on normalised data
# Tick labels are based on original data
def set_ticks_for_axis(dim, ax, min_max_range, cols, df, dict, padding, equidistant=False):
    min_val, max_val, val_range = min_max_range[cols[dim]]
    if cols[dim]=='processor':
        ticks,labels = [0.25,0.75],['CPU','GPU']
    elif cols[dim]=='channels':
        ticks,labels = [0.25,0.75],['B&W','RGB']
    elif cols[dim]=='epochs':
        assert max_val>75, 'Maximum epoch index found (%i) below highest user define epoch tick label (75)'%max_val
        labels = numpy.array([min_val,25,50,75,max_val],dtype=int)
        ticks = (labels-min_val)/val_range
    elif cols[dim]=='accuracy':
        labels = numpy.linspace(min_val, max_val, num=10, endpoint=True, dtype=int)
        ticks = (labels-min_val)/val_range
    else:
        ticks = sorted(numpy.unique(df[cols[dim]]))
        if equidistant:
            labels = dict[cols[dim]]
        else:
            labels = [i*val_range+min_val for i in ticks]
        if cols[dim] not in ['learning_rate']:
            labels = [int(i) for i in labels]
        else:
            labels = [round(i,4) for i in labels]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(labels,color='black')
    for label in ax.get_yticklabels():
        label.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),path_effects.Normal()])
    ax.tick_params(axis="y",direction='inout',pad=0)
    plt.setp(ax.get_yticklabels(), ha="center")
    ax.set_ylim(0-padding,1+padding)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def main():
    """
    ~/Documents/mldas/mldas/parallel.py -r pytorch-ml4das/ -ec -p 0.1
    """
    args = parse_args()
    cols, dict, best = get_file_results(os.path.abspath(os.path.expanduser(args.repository)), args.equidistant)
    df = pandas.read_csv('parallel_coordinates.csv')
    colormap = cm.viridis #mcolors.ListedColormap(custom.custom_cm()/255.0)
    color_normalize = mcolors.Normalize(vmin=0, vmax=1)
    min_max_range, df = normalize(df,cols,args.accuracy)
    print(len(df),'jobs in total.')
    x = [i for i, _ in enumerate(cols)]
    # Create (X-1) sublots along x axis
    plt.style.use('seaborn')
    plt.rc('font', size=15)
    plt.rc('axes', labelsize=15, titlesize=10)
    plt.rc('legend', fontsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,7), dpi=200)
    # Remove space between subplots
    plt.subplots_adjust(wspace=0,bottom=0.1,left=0.05,right=0.92,top=0.95)
    # Loop over each parameter corresponding subplot axe
    groups = ['Hardware','Hardware','Data','Data','Data','Architecture','Architecture','Training','Training']
    for i, ax in enumerate(axes):
        print('Processing column',i+1,'out of',len(axes))
        ax.axhspan(1,1+args.padding,color='white',zorder=-3)
        ax.axhspan(0-args.padding,0,color='white',zorder=-3)
        ax.axvline(i,ymin=0,ymax=args.padding/(1+2*args.padding),color='0.7',lw=1,ls='dashed')
        # Loop over all rows in CSV file
        for idx in df.index:
            color, width, order = ('red',5,-1) if idx==best else ('gray',0.1,-2)
            xplot,yplot = x,df.loc[idx, cols]
            if args.interpolation:
                f = interp1d(xplot, yplot, kind='quadratic')
                xplot = numpy.linspace(xplot[0], xplot[-1], num=1000, endpoint=True)
                yplot = f(xplot)
            if args.colorbar and idx!=best:
                color = colormap(color_normalize(yplot[-1]))
            ax.plot(xplot, yplot, c=color, lw=width, zorder=order)
            #if idx>50: break
        # Rescale X axis to only show the connection from target parameter
        ax.set_xlim([x[i], x[i+1]])
        ax.xaxis.set_major_locator(ticker.FixedLocator([i]))
        set_ticks_for_axis(i, ax, min_max_range, cols, df, dict, args.padding, args.equidistant)
        label = cols[i]+'\n(in thousands)' if cols[i]=='dataset_size' else cols[i]
        ax.set_xticklabels([label])
        ax.text(i,1.1,groups[i].upper(),ha='center',va='bottom',fontsize=10,weight='bold')
        ax.axvline(i,ymin=1-args.padding/(1+2*args.padding),ymax=1,color='0.7',lw=1,ls='dashed')
    if args.colorbar:
        pad = args.padding*0.85/(1+2*args.padding)
        cbar = fig.add_axes([0.922,0.1+pad,0.01,0.85-2*pad])
        s_map = cm.ScalarMappable(norm=color_normalize, cmap=colormap)
        s_map.set_array(numpy.unique(df['accuracy']))
        min_val, max_val, val_range = min_max_range['accuracy']
        labels = numpy.linspace(min_val, max_val, num=10, endpoint=True, dtype=int)
        ticks = (labels-min_val)/val_range
        plt.colorbar(s_map, cax=cbar).set_label('Validation accuracy (in percent)')
        cbar.yaxis.set_ticks(ticks)
        cbar.set_yticklabels(labels)
    else:
        # Move the final axis' ticks to the right-hand side
        ax = plt.twinx(axes[-1])
        ax.plot([0,0], [0,1], 'gray', lw=0.1)
        ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        ax.axvline(0,ymin=0,ymax=args.padding/(1+2*args.padding),color='0.7',lw=1,ls='dashed')
        set_ticks_for_axis(len(axes), ax, min_max_range, cols, df, dict, args.padding, args.equidistant)
        ax.set_xticklabels([cols[-2], cols[-1]])
        ax.grid(False)
    plt.savefig('parallel_coordinates.png')
    os.system('rm *.csv')

if __name__ == '__main__':
    main()
