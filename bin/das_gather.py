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
import glob
import json
import argparse
import operator
from collections import OrderedDict

# Externals
import yaml
import h5py
import numpy
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from torchvision import transforms
from matplotlib import cm
from PIL import Image

# Locals
from models import resnet
from explore import lookup

"""
./mldas/gather.py -m multilabel -p hsw -s
./mldas/gather.py -m multilabel -p gpu -s
~/Documents/mldas/mldas/gather.py -t pytorch-ml4das/multiclass/2-neuron/3-channel/gpu-n8-ds1-bs128-ep50-dp14-lr0.001/checkpoints/model_checkpoint_049.pth.tar -r ./
"""

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-m', '--mode', choices=['multiclass','multilabel','unary','singleneuron'],
            help='Specify training mode')
    add_arg('-c', '--num-channels', type=int,
            help='Specify number of channels')
    add_arg('-i', '--image',
            help='Specify specific image')
    add_arg('-f', '--file',
            help='YAML file')
    add_arg('-l', '--lr', type=float, nargs='*', default=[],
            help='Specify learning rate')
    add_arg('-n', '--num-neurons', type=int,
            help='Specify number of classes (neurons)')
    add_arg('-p', '--processor', choices=['hsw','gpu'],
            help='Specify processors')
    add_arg('-r', '--repository', default='/global/cscratch1/sd/vdumont/',
            help='Specify data repository')
    add_arg('-s', '--sort', action='store_true',
            help='Specify whether to sort results')
    add_arg('-t', '--target',
            help='Target model')
    return parser.parse_args()
        
def load_data(path,img_size=200):
    """
    Load reference signals.
    """
    data = {}
    with h5py.File(path+'/1min_ch4650_4850/180104/westSac_180104001631_ch4650_4850.mat','r') as f:
        data['nonco']=f[f.get('variable/dat')[0,0]][:img_size,15000:15000+img_size]
    with h5py.File(path+'/30min_files_Train/Dsi_30min_170730023007_170730030007_ch5500_6000_NS.mat','r') as f:
        data['noise']=f[f.get('dsi30/dat')[0,0]][201:201+img_size,384672:384672+img_size]
        data['waves']=f[f.get('dsi30/dat')[0,0]][193:193+img_size,921372:921372+img_size]
        data['high'] =f[f.get('dsi30/dat')[0,0]][157:157+img_size,685516:685516+img_size]
    return data

def get_model(path,depth,num_classes=2,num_channels=1):
    model = resnet.ResNet(depth,num_classes,num_channels)
    checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_prob(model,data,activation,channels,verbose=False):
    probs,out = {},[]
    for label in ['noise','waves','nonco','high']:
        image = data[label]
        image = (image-image.min())/(image.max()-image.min())
        image = Image.fromarray(numpy.uint8(image*255))
        if channels==3:
            image = transforms.ToTensor()(image.convert("RGB")).view(1,3,200,200)
        if channels==1:
            image = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(image).view(1,1,200,200)
        output = model(image)
        if activation=='softmax':
            assert output.dim()>1, 'Softmax not implemented for single neuron.'
            prob = F.softmax(output,dim=1)
        if activation=='sigmoid':
            prob = torch.sigmoid(output)
        wave_prob = prob if output.dim()==1 else prob[0,0] if output.shape[1]==1 else prob[0,1]
        wave_prob = wave_prob.item()
        if verbose:
            print('{:>5}:{:>.4f} output:{} prob:{}'.format(label,wave_prob,output.data,prob.data))
        if label in ['noise','waves']:
            probs[label] = output if output.dim()==1 else output[0,0] if output.shape[1]==1 else output[0,1]
        out.append(wave_prob)
    out.append(out[1]-out[0])
    return numpy.array(out),probs['noise'],probs['waves']

def get_prob_from_file(model,img_path,activation,channels):
    image = Image.open(img_path)
    if channels==3:
        image = transforms.ToTensor()(image.convert("RGB")).view(1,3,200,200)
    if channels==1:
        image = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(image).unsqueeze(1)
    output = model(image)
    if activation=='softmax':
        assert output.dim()>1, 'Softmax not implemented for single neuron.'
        prob = F.softmax(output,dim=1)
    if activation=='sigmoid':
        prob = torch.sigmoid(output)
    wave_prob = prob if output.dim()==1 else prob[0,0] if output.shape[1]==1 else prob[0,1]
    wave_prob = wave_prob.item()
    print('{}:{:>.4f} output:{} prob:{}'.format(os.path.basename(img_path),wave_prob,output.data,prob.data))

def prob_check(gather,data,mode,activation,num_classes,channels,path,sort=True):
    results = numpy.empty((0,13))
    jobs = []
    for i,target in enumerate(gather.keys()):
        #if i not in [19,20]: continue
        jobs.append(target)
        job_path = '/'.join([path,'pytorch-ml4das',mode,'%i-neuron'%num_classes,'%i-channel'%channels,target])
        n_epochs = len(gather[target]['valid_acc'])
        acc,probs = float('nan'),[float('nan')]*5
        gather[target]['prob_noise'] = list()
        gather[target]['prob_waves'] = list()
        for n_epoch in range(n_epochs):
            model_path = '%s/checkpoints/model_checkpoint_%03i.pth.tar'%(job_path,n_epoch)
            if os.path.exists(model_path):
                acc = gather[target]['valid_acc'][-1]
                model = get_model(model_path,gather[target]['depth'],num_classes,channels)
                if model!=None:
                    probs,wav4noise,wav4waves = get_prob(model,data,activation,channels)
                    gather[target]['prob_noise'].append(wav4noise.data.item())
                    gather[target]['prob_waves'].append(wav4waves.data.item())
        results = numpy.vstack((results,[i,gather[target]['node'],gather[target]['size'],
                                         gather[target]['batch'],gather[target]['epochs'],
                                         gather[target]['depth'],gather[target]['lrate'],
                                         acc,*probs]))
        #for i,n,ds,bs,ep,dp,lr,ac,noise,waves,nonco,high,diff in [results[-1]]:
        #    print('{:>2}:{:>4}{:>5}{:>5}{:>5}{:>5}{:>8.3f}{:>8.2f}{:>12.4f}{:>10.4f}{:>10.4f}{:>10.4f}   {}'
        #          .format(i+1,int(n),int(ds),int(bs),int(ep),int(dp),lr,ac,noise,waves,nonco,high,jobs[-1]))
    if sort:
        results = results[results[:,-1].argsort()][::-1]
    else:
        results = sorted(results,key=operator.itemgetter(2,5,6))
    for k,(i,n,ds,bs,ep,dp,lr,ac,noise,waves,nonco,high,diff) in enumerate(results):
        print('{:>2}:{:>4}{:>5}{:>5}{:>5}{:>5}{:>8.3f}{:>8.2f}{:>12.4f}{:>10.4f}{:>10.4f}{:>10.4f}   {}'
              .format(k+1,int(n),int(ds),int(bs),int(ep),int(dp),lr,ac,noise,waves,nonco,high,jobs[int(i)]))
    return gather

def get_file_results(path,mode,classes,channels,proc,lrs):
    target = '%s/pytorch-ml4das/%s/%i-neuron/%i-channel/%s-*'%(path,mode,classes,channels,proc)
    assert len(glob.glob(target))>0, 'No target directories found in %s'%target
    gather = {}
    for outfile in sorted(glob.glob(target)):
        target = os.path.basename(outfile)
        if any([(lr>0 and 'lr%s'%lr not in target) or \
                (lr<0 and 'lr%s'%abs(lr) in target) for lr in lrs]):
            continue
        train_loss, train_acc, valid_loss, valid_acc = lookup.find_values('%s/out_0.log'%outfile)
        params = target.split('-')
        gather[target] = {}
        gather[target]['proc']       = str(params[0].replace(mode+'/',''))
        gather[target]['node']       = int(params[1][1:])
        gather[target]['size']       = int(params[2][2:])
        gather[target]['batch']      = int(params[3][2:])
        gather[target]['epochs']     = int(params[4][2:])
        gather[target]['depth']      = int(params[5][2:])
        gather[target]['lrate']      = float(params[6][2:])
        gather[target]['train_loss'] = list(train_loss)
        gather[target]['train_acc']  = list(train_acc)
        gather[target]['valid_loss'] = list(valid_loss)
        gather[target]['valid_acc']  = list(valid_acc)
    return gather

def make_subplot(ax,gather,sample_size,spec):
    colors = {0.001:'royalblue',0.005:'turquoise',
              0.01:'salmon',0.05:'mediumseagreen',
              0.1:'sandybrown',0.5:'mediumpurple'}
    marker = {2:'d',8:'o',14:'s',20:'<',26:'p',32:'*'}
    for run in gather.keys():
        params = gather[run]
        if params['size']==sample_size:
            ax.plot(numpy.arange(len(params[spec])),params[spec],
                    color=colors[params['lrate']],marker=marker[params['depth']],markersize=4,lw=0.5)
            #ax.set_xlim(0,params['epochs']-1)
            ax.set_title('%ik images per label - Batch size of %i'%(params['size'],params['batch']))

def add_legend(fig):
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-')
             for c in ['royalblue', 'turquoise', 'salmon', 'mediumseagreen','sandybrown','mediumpurple']]
    labels = ['0.001', '0.005', '0.01', '0.05','0.1','0.5']
    leg = fig.legend(lines, labels,loc="lower center",title="Learning rate",ncol=6,
                     bbox_to_anchor=[0.28, 0],frameon=True)
    d2  = mlines.Line2D([],[],color='grey',marker='d',linestyle='None',markersize=10,label='2')
    d8  = mlines.Line2D([],[],color='grey',marker='o',linestyle='None',markersize=10,label='8')
    d14 = mlines.Line2D([],[],color='grey',marker='s',linestyle='None',markersize=10,label='14')
    d20 = mlines.Line2D([],[],color='grey',marker='<',linestyle='None',markersize=10,label='20')
    d26 = mlines.Line2D([],[],color='grey',marker='<',linestyle='None',markersize=10,label='26')
    d32 = mlines.Line2D([],[],color='grey',marker='<',linestyle='None',markersize=10,label='32')
    leg = fig.legend(handles=[d2,d8,d14,d20,d26,d32],loc="lower center",title="Depth of neural network",ncol=6,
                     bbox_to_anchor=[0.72, 0],frameon=True)
            
def plot_results(gather,spec,dir_dst):
    os.makedirs(dir_dst, exist_ok=True)
    ylabel = dict(train_loss='Training loss',train_acc='Training accuracy',
                  valid_loss='Validation loss',valid_acc='Validation accuracy',
                  prob_noise='Wave probability on white noise data',
                  prob_waves='Wave probability on surface wave data')
    if type(gather)==str:
        with open(gather) as file:
            gather = yaml.load(file, Loader=yaml.FullLoader)
    plt.style.use('seaborn')
    plt.rc('font', size=10)
    plt.rc('axes', labelsize=10, titlesize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    fig,ax = plt.subplots(2,3,figsize=(12,8),dpi=80,sharex='col',sharey=True)
    fig.subplots_adjust(left=0.05, bottom=0.13, right=0.98, top=0.95, wspace=0.05, hspace=0.1)
    for n,sample_size in enumerate([1,5,10,50,100,150]):
        i, j = n//2, n-2*(n//2)
        make_subplot(ax[j][i],gather,sample_size,spec)
        if i==0: ax[j][i].set_ylabel(ylabel[spec])
        if j==1: ax[j][i].set_xlabel('Epochs')
    add_legend(fig)
    plt.savefig('%s/%s.pdf'%(dir_dst,spec))

def main():

    args = parse_args()
    dir_src = os.path.abspath(os.path.expanduser(args.repository))

    if args.target!=None:
        path_split   = args.target.split('/')
        num_neurons  = int(path_split[-5].split('-')[0])
        num_channels = int(path_split[-4].split('-')[0])
        depth        = int(path_split[-3].split('-')[-2][2:])
        activation   = 'softmax' if path_split[-6]=='multiclass' and num_neurons>1 else 'sigmoid'
        model = get_model(args.target,depth,num_neurons,num_channels)
        if args.image==None:
            data = load_data(path=dir_src)
            get_prob(model,data,activation,num_channels,verbose=True)
        else:
            get_prob_from_file(model,args.image,activation,num_channels)
        quit()

    if args.file==None:
        assert all([args.mode!=None,args.processor!=None,args.num_channels!=None,args.num_neurons!=None]), \
            'All 4 arguments needed: --mode (-m), --processor (-p), --num-channels (-c), --num-neurons needed (-n).'
        data = load_data(path=dir_src)
        activation = 'softmax' if args.mode=='multiclass' and args.num_neurons>1 else 'sigmoid'
        gather = get_file_results(dir_src,args.mode,args.num_neurons,args.num_channels,args.processor,args.lr)
        print(len(gather.keys()),'repositories found.')
        dir_dst = '%s/output/%s/%i-neuron/%i-channel/%s/'%(dir_src,args.mode,args.num_neurons,args.num_channels,args.processor)
        gather = prob_check(gather,data,args.mode,activation,args.num_neurons,args.num_channels,dir_src,args.sort)
        with open('%s/store_dict.yaml'%dir_dst, 'w') as file:
            documents = yaml.dump(gather, file)

    if args.file!=None: gather, dir_dst = args.file, os.path.dirname(args.file)
    for spec in ['train_loss','train_acc','valid_loss','valid_acc','prob_noise','prob_waves']:
        plot_results(gather,spec,dir_dst)

if __name__ == '__main__':
    main()
