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
import operator

# Externals
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Local
from explore import lookup
from explore import colormap as custom
from gather import load_data, get_model, get_prob

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('operation', choices=['assess','scaling','tuning','hyper'],
            help='Operation to be executed')
    add_arg('-d', '--data-dir',
            help='Path to DAS data')
    add_arg('-e', '--epoch', type=int,
            help='Epoch to extract probability')
    add_arg('--extension', default='png',
            help='Output image file extension')
    add_arg('-i', '--interpolation', action='store_true',
            help='Epoch to extract probability')
    add_arg('-t', '--target',
            help='Specify target result directories')
    add_arg('-u', '--uniform', action='store_true',
            help='Uniform color across all subplots')
    return parser.parse_args()

def prob_view(params):
    assert params.data_dir!=None, 'Path to DAS data not specify.'
    if params.target.endswith('txt')==False:
        assert os.path.exists('record.txt')==False, 'record.txt already exists.'
        params.target = get_prob_results(params.target, params.data_dir)
    das_data = load_data(path=params.data_dir)
    data, idxs, epochs = load_record(params.target)
    colormap = 'tab10' #mcolors.ListedColormap(custom.custom_cm()/255.0)
    plt.style.use('seaborn')
    plt.rc('font', size=15)
    plt.rc('axes', labelsize=15, titlesize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots(2,4,figsize=(18,9.1), dpi=100, sharey='row') 
    plt.subplots_adjust(wspace=0.02,hspace=0.02,bottom=0.08,left=0.04,right=0.94,top=0.96)
    for i,(key,title) in enumerate([['noise','White noise signal'],
                                    ['waves','Coherent waves'],
                                    ['nonco','Non-coherent waves'],
                                    ['high','Saturated signals']]):
        ax[0][i].imshow(das_data[key],
                        cmap='binary',aspect='equal')
        ax[0][i].axis(False)
        ax[0][i].set_title(title)
    ax[1][0].set_ylabel('Model index')
    for i in range(4):
        im = ax[1][i].imshow(data[i],extent=[0,data[i].shape[1],data[i].shape[0],0],
	                  interpolation='bicubic' if params.interpolation else None,
	                  cmap=colormap,aspect='auto',vmin=0,vmax=1)
        ax[1][i].set_xlabel('Training epoch')
        for j in idxs[0]:
            y = j//epochs
            if y==j/epochs:
                y+=1
            x = j-y*epochs-1
            ax[1][i].add_patch(Rectangle((x,y),1,1,fill=False,edgecolor='white',lw=2))
        ax[1][i].xaxis.set_ticks(numpy.arange(0,epochs,10))
        ax[1][i].set_xticklabels(numpy.arange(0,epochs,10))
    cbar = fig.add_axes([0.944,0.08,0.01,0.435])
    plt.colorbar(im, cax=cbar).set_label('Probability of usable energy')
    plt.savefig('plot.'+params.extension)

def get_prob_results(target, data_path):
    data = load_data(path=data_path)
    record = numpy.empty((0,6),dtype=object)
    for outfile in glob.glob(target):
        if os.path.exists('%s/results.txt'%outfile)==False:
            continue
        o = '/'.join(outfile.split('/')[-4:])
        m = re.match('(\w+)/(\d+)-neuron/(\d+)-channel/(\w+)-n(\d+)-ds(\d+)-bs(\d+)-ep(\d+)-dp(\d+)-lr(\d+\.\d+)', o)
        (class_type,classes,channels,epochs,depth) = [m.group(i) for i in [1,2,3,8,9]]
        activation = 'softmax' if class_type=='multiclass' and int(classes)>1 else 'sigmoid'
        for n_epoch in range(int(epochs)):
            model_path = '%s/checkpoints/model_checkpoint_%03i.pth.tar'%(outfile,n_epoch)
            if os.path.exists(model_path):
                model = get_model(model_path,int(depth),int(classes),int(channels))
                probs,_,_ = get_prob(model,data,activation,int(channels))
                print(o,n_epoch+1,*probs)
                record = numpy.vstack((record,[o,n_epoch+1,*probs[:-1]]))
    numpy.savetxt('record.txt', record, fmt='%s')
    return 'record.txt'

def load_record(target):
    # Load recorded probability values and extract total number of jobs
    all_record = numpy.loadtxt(target,dtype=str)
    jobs = numpy.unique(all_record[:,0])
    depth_lr = numpy.array([(int(job.split('-')[-2][2:]),float(job.split('-')[-1][2:])) for job in jobs],
                           dtype=numpy.dtype([('depth', int), ('lr', float)]))
    jobs = jobs[numpy.argsort(depth_lr, order=('depth', 'lr'))]
    print(jobs)
    #print(jobs)
    njobs = len(jobs)
    # Remove job name column and extract maximum number of epochs
    record = numpy.empty(all_record.shape[1]-1,dtype=float)
    job_names = numpy.empty((0,1))
    for job_name in jobs:
        for line in all_record:
            if line[0]==job_name:
                record = numpy.vstack((record,line[1:].astype(float)))
                job_names = numpy.vstack((job_names,line[0]))
    #record = all_record[:,1:].astype(float)
    epochs = int(max(record[:,0]))
    # Identify indexes with trustworthy probabilities
    idxs = numpy.where((record[:,1]<0.1) & (record[:,2]>0.9) &
                       (0.7<record[:,3]) & (record[:,3]<0.9) &
                       (0.2<record[:,4]) & (record[:,4]<0.3))
    print(record[idxs])
    print(job_names[idxs])
    # Initialize probability arrays with NaN values
    data = numpy.empty((4,njobs,epochs))
    data[:] = numpy.nan
    # Loop over recorded values and store them in data array
    njob = -1
    for epoch,prob1,prob2,prob3,prob4 in record:
        if epoch==1:
            njob+=1
        for i,prob in enumerate([prob1,prob2,prob3,prob4]):
            data[i,njob,int(epoch)-1] = prob
    return data,idxs, epochs

def hyper_tune(params):
    data,lrs,dps = get_file_results(params.target, params.epoch)
    plt.style.use('seaborn')
    plt.rc('font', size=15)
    plt.rc('axes', labelsize=15, titlesize=10)
    plt.rc('legend', fontsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.figure(figsize=(6,5), dpi=100)
    plt.imshow(data,extent=[0,data.shape[1],0,data.shape[0]],
               interpolation='bicubic' if params.interpolation else None,
               cmap='inferno',aspect='auto',vmax=100)
    for y in range(1,data.shape[0]):
        plt.axhline(y,color='white')
    for x in range(1,data.shape[1]):
        plt.axvline(x,color='white')
    plt.yticks(numpy.arange(data.shape[0])+0.5,list(lrs.keys())[::-1])
    plt.xticks(numpy.arange(data.shape[1])+0.5,list(dps.keys()))
    plt.xlabel('Neural Network Depth')
    plt.ylabel('Learning Rate')
    plt.colorbar().set_label('Validation Accuracy') 
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('hyper_%s.%s'%(params.target.replace('/','_'),params.extension))

def get_file_results(target, epoch):
    dict = {'depth':9,'learning_rate':10}
    data = numpy.empty((0,3),dtype=float)
    for outfile in glob.glob(target+'*'):
        if os.path.exists('%s/results.txt'%outfile)==False:
            continue
        train_loss, train_acc, valid_loss, valid_acc = lookup.find_values('%s/out_0.log'%outfile)
        o = '/'.join(outfile.split('/')[-4:])
        m = re.match('(\w+)/(\d+)-neuron/(\d+)-channel/(\w+)-n(\d+)-ds(\d+)-bs(\d+)-ep(\d+)-dp(\d+)-lr(\d+\.\d+)', o)
        if target in outfile:
            acc = valid_acc[-1] if epoch==None else valid_acc[epoch]
            data = numpy.vstack((data,[float(m.group(dict['learning_rate'])),int(m.group(dict['depth'])),acc]))
    lrs, dps = {}, {}
    for i,lr in enumerate(sorted(numpy.unique(data[:,0]))[::-1]):
        lrs[float(lr)] = i
    for i,dp in enumerate(sorted(numpy.unique(data[:,1]))):
        dps[int(dp)] = i
    array = numpy.array([numpy.nan]*len(lrs.keys())*len(dps.keys())).reshape(len(lrs.keys()),len(dps.keys()))
    for lr,dp,ac in data:
        array[lrs[lr],dps[dp]] = ac
    return array,lrs,dps

def gpu_scaling(params):
    # GPUs, depth, batches, sample sizes
    refs = [[2,4,8,16,32],[8,14,20,26,32],[64,128,256,512,1024],[1,5,10,50,100]]
    data = get_time_results(params.target,refs)
    plt.style.use('seaborn')
    plt.rc('font', size=15)
    plt.rc('axes', labelsize=15, titlesize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots(1,3,figsize=(13,5), dpi=100, sharey=True) 
    plt.subplots_adjust(wspace=0.02,bottom=0.12,left=0.05,right=0.92,top=0.92)
    ax[0].set_ylabel('Total number of GPUs')
    for i,times in enumerate(data):
        im = ax[i].imshow(times,extent=[0,times.shape[1],times.shape[0],0],
	                  interpolation='bicubic' if params.interpolation else None,
	                  cmap='jet',aspect='auto')
        ax[i].set_xlabel('Depth' if i==0 else 'Batch size' if i==1 else 'Sample size (in thousands)')
        # Total number of GPUs
        for y in range(1,times.shape[0]):
            ax[i].axhline(y,color='white')
        ax[i].yaxis.set_ticks(numpy.arange(times.shape[0])+0.5)
        ax[i].set_yticklabels(refs[0])
        # Lookup variables
        for x in range(1,times.shape[1]):
            ax[i].axvline(x,color='white')
        ax[i].xaxis.set_ticks(numpy.arange(times.shape[1])+0.5)
        ax[i].set_xticklabels(refs[i+1])
        ax[i].grid(False)
    cbar = fig.add_axes([0.924,0.12,0.01,0.8])
    plt.colorbar(im, cax=cbar).set_label('Training throughput [samples/s]')
    plt.savefig('times.'+params.extension)

def get_time_results(target,refs):
    data = numpy.zeros((3,5,5),dtype=float)
    #data[numpy.where(data==0)] = numpy.nan
    for outfile in glob.glob(target):
        if os.path.exists('%s/results.txt'%outfile)==False:
            print('failed\t',outfile)
            continue
        print('\t',outfile)
        summary_files = [f for f in os.listdir(outfile) if f.startswith('summaries_')]
        train_rate, inference_rate = 0, 0
        for summary_file in summary_files:
            with numpy.load(os.path.join(outfile, summary_file)) as f:
                train_rate += f['train_rate'].mean()
                inference_rate += f['valid_rate'].mean()
        #train_rate = numpy.loadtxt('%s/results.txt'%outfile,usecols=[0],skiprows=1)
        m = re.match('(\w+)-n(\d+)-ds(\d+)-bs(\d+)-ep(\d+)-dp(\d+)-lr(\d+\.\d+)', outfile.split('/')[-1])
        nodes = float(m.group(2))
        dsize = float(m.group(3))
        bsize = float(m.group(4))
        depth = float(m.group(6))
        for n,var in enumerate([depth,bsize,dsize]):
            if (n==0 and bsize==64  and dsize==5) or \
               (n==1 and depth==8   and dsize==5) or \
               (n==2 and depth==8   and bsize==256):
                data[n,refs[0].index(nodes),refs[n+1].index(var)] = train_rate
    return data

def gpu_bsds(params):
    # GPUs, depth, batches, sample sizes
    refs = [[2,4,8,16,32],[100,50,10,5,1],[64,128,256,512]]
    titles = ['2 GPUs (partial single node)','4 GPUs (half single node)',
              '8 GPUs (full single node)','16 GPUs (two full nodes)',
              '32 GPUs (four full nodes)']
    #data = numpy.zeros((5,5,5))
    data = get_time(params.target,refs)
    vmin,vmax = data.min(),data.max()
    plt.style.use('seaborn')
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12, titlesize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    fig, ax = plt.subplots(1,5,figsize=(13,5), dpi=100, sharey=True) 
    ax[0].set_ylabel('Sample size (in thousands)')
    if params.uniform:
        data[numpy.where(data==0)] = numpy.nan
        vmin,vmax = numpy.nanmin(data),numpy.nanmax(data)
        plt.subplots_adjust(wspace=0.05,hspace=0.02,bottom=0.1,left=0.05,right=0.92,top=0.9)
    for i,times in enumerate(data):
        if params.uniform==False:
            times[numpy.where(times==0)] = numpy.nan
            vmin,vmax = numpy.nanmin(times),numpy.nanmax(times)
        if params.interpolation: times = numpy.nan_to_num(times)
        im = ax[i].imshow(times,extent=[0,times.shape[1],times.shape[0],0],
                          interpolation='bicubic' if params.interpolation else None,
                          cmap='gist_stern',aspect='auto',vmin=vmin,vmax=vmax)
        ax[i].set_xlabel('Batch size')
        if params.uniform==False:
            ax_divider = make_axes_locatable(ax[i])
            cax = ax_divider.append_axes("top", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            if i==0: cax.set_ylabel('Training\nthroughput\n [sample/s]', fontsize=10)
            cax.xaxis.set_label_position('top')
            cax.xaxis.set_ticks_position("top")
        # Total number of GPUs
        for y in range(1,times.shape[0]):
            ax[i].axhline(y,color='white')
        ax[i].yaxis.set_ticks(numpy.arange(times.shape[0])+0.5)
        ax[i].set_yticklabels(refs[1])
        # Lookup variables
        for x in range(1,times.shape[1]):
            ax[i].axvline(x,color='white')
        ax[i].xaxis.set_ticks(numpy.arange(times.shape[1])+0.5)
        ax[i].set_xticklabels(refs[2])
        title = titles[i].replace('GPUs','GPUs\n') if params.uniform else titles[i]+'\n\n\n\n'
        ax[i].set_title(title,weight='bold')
        ax[i].grid(False)
    if params.uniform:
        cbar = fig.add_axes([0.928,0.1,0.01,0.8])
        plt.colorbar(im, cax=cbar).set_label('Training throughput [sample/s]')
    else:
        plt.tight_layout(w_pad=-2.5)
    plt.savefig('scaling.'+params.extension)

def get_time(target,refs):
    #for nodes in refs[0]:
    #    for ds in refs[1]:
    #        for bs in refs[2]:
    #            sample_per_iter = ds*1000*2*0.7/bs/nodes
    #            if 100<sample_per_iter<200:
    #                print('sbatch -n {:>2} scripts/train_cgpu.sh --depth  8 --epochs 1 --sample-size {:>3} --batch-size {:>4}'.format(nodes,ds,bs))
    data = numpy.zeros((5,5,4),dtype=float)
    for outfile in sorted(glob.glob(target)):
        m = re.match('(\w+)-n(\d+)-ds(\d+)-bs(\d+)-ep(\d+)-dp(\d+)-lr(\d+\.\d+)', outfile.split('/')[-1])
        nodes = int(m.group(2))
        dsize = int(m.group(3))
        bsize = int(m.group(4))
        sample_per_iter = dsize*1000*2*0.7/bsize/nodes
        if nodes not in refs[0] or dsize not in refs[1] or bsize not in refs[2]:
            continue
        elif os.path.exists('%s/results.txt'%outfile)==False:
            train_rate, inference_rate = numpy.nan, numpy.nan
        else:
            summary_files = [f for f in os.listdir(outfile) if f.startswith('summaries_')]
            train_rate, inference_rate = 0., 0.
            for summary_file in summary_files:
                with numpy.load(os.path.join(outfile, summary_file)) as f:
                    train_rate += f['train_rate'].mean()
                    inference_rate += f['valid_rate'].mean()
            assert train_rate!=0, 'Training throuput equal to 0.'
        data[refs[0].index(nodes),refs[1].index(dsize),refs[2].index(bsize)] = train_rate
    return data

def main():
    """
    # Assessing probabilities 
    scaletune.py tuning -i -t pytorch-ml4das/multiclass/2-neuron/1-channel/gpu-n4-ds1-bs128-ep40-dp
    scaletune.py hyper -i -t 'pytorch-ml4das/multiclass/2-neuron/1-channel/gpu-n*-ds*-bs*-ep10-dp*-lr0.01'
    scaletune.py scaling -i -t 'pytorch-ml4das/multiclass/2-neuron/1-channel/gpu-n*-ds*-bs*-ep1-dp8-lr0.01' 
    scaletune.py assess -i -t pytorch-ml4das/multiclass/2-neuron/1-channel/gpu-n4-ds10-bs128-ep40-dp
    scaletune.py assess -i -t record_ds1.txt -d ~/Desktop/
    """
    args = parse_args()
    if args.operation=='assess':
        prob_view(args)
    if args.operation=='tuning':
        hyper_tune(args)
    if args.operation=='hyper':
        gpu_scaling(args)
    if args.operation=='scaling':
        gpu_bsds(args)

if __name__ == '__main__':
    main()
