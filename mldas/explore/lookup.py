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
import random
import glob
import math

# Externals
import yaml
import numpy
import scipy.io

# locals
from ..datasets import get_data_loaders

def find_values(fname):
    with open(fname,'r') as f:
        file_data = f.read()
    version = 1 if 'Training | Total loss' in file_data else 2
    out = open(fname,'r')
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    for line in out:
        if version==1:
            if 'Training | Total loss' in line :
                train_loss.append(float(line.split()[-1]))
            if 'Validation | Total loss' in line:
                valid_loss.append(float(line.split()[-5]))
                valid_acc.append(float(line.split()[-1]))
        if version==2:
            if 'Training' in line :
                train_loss.append(float(line.split()[-4]))
                train_acc.append(float(line.split()[-1]))
            if 'Validation' in line:
                valid_loss.append(float(line.split()[-4]))
                valid_acc.append(float(line.split()[-1]))
    out.close()
    return train_loss, train_acc, valid_loss, valid_acc

def fwi_check(input_data,stop,verbose,**kwargs):
    """
    Quick look at random data for inversion problem.

    Example
    -------
    >>> das_quickrun.py fwi_check -i '/clusterfs/bear/ML-mkshots-fcheng/mat_run0*/*.mat'
    """
    input_data = sorted(glob.glob(input_data[0]))
    random.shuffle(input_data)
    bad,info = [],numpy.empty((0,4))
    for i,fname in enumerate(input_data):
        print('%i / %i'%(i+1,len(input_data)))
        try:
            data = scipy.io.loadmat(fname)
            info = numpy.vstack((info,[fname,len(data['thk']),sum(data['thk'][:,0]),max(data['vs'][:,0])]))
            if verbose:
                print('Filename ...................... %s'%fname)
                print('Shape of wavefield ............ %s'%str(data['uxt'].shape))
                print('Shape of dispersion spectrum .. %s'%str(data['fv'].shape))
                print('Number of Layers .............. %i'%len(data['thk']))
                print('Velocities .................... %s'%data['vs'][:,0])
                print('Thicknesses ................... %s'%data['thk'][:,0])
                print('Summed thickness .............. %i'%sum(data['thk'][:,0]))
                print('\n')
            if i==stop:
                break
        except:
            bad.append(fname)
            continue
    if len(info)>0: numpy.savetxt('info.txt',info,fmt='%s')
    if len(bad)>0: numpy.savetxt('bad.txt',bad,fmt='%s')

def get_spec(config,test_index,**kwargs):
    """
    Show the velocity model from file in test dataset using its index value.
    The test dataset is not shuffled so the index from the loaded data reflects
    the position of the file within the loaded test dataset.

    Examples
    --------
    >>> das_quickrun.py get_spec -c fwi.yaml -t 1
    """
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    file_list = sorted(glob.glob(config['data_config']['data_path']))
    split = math.ceil(0.2*len(file_list))
    # Get filename of of test sample
    fname = file_list[:split][test_index]
    data = scipy.io.loadmat(fname)
    vmodel = numpy.array([[data['vs'][i,0],sum(data['thk'][:i+1,0])] for i in range(len(data['vs']))],dtype=float)
    print(vmodel)
