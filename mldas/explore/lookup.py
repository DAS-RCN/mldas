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

# Externals
import scipy.io

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

def fwi_check(input_data,**kwargs):
    """
    Quick look at random data for inversion problem.
    das_quickrun.py fwi_check -i /clusterfs/bear/ML-mkshots-fcheng/mat_run0*/*.mat
    """
    random.shuffle(input_data)
    nlayers = []
    for i,fname in enumerate(input_data):
        data = scipy.io.loadmat(fname)
        nlayer = len(data['thk'])
        print('Filename ...................... %s'%fname)
        print('Shape of wavefield ............ %s'%str(data['uxt'].shape))
        print('Shape of dispersion spectrum .. %s'%str(data['fv'].shape))
        print('Number of Layers .............. %i'%nlayer)
        print('Velocities .................... %s'%data['vs'][:,0])
        print('Thicknesses ................... %s'%data['thk'][:,0])
        print('Summed thickness .............. %i'%sum(data['thk'][:,0]))
        print('\n')
        #if nlayer not in nlayers:
        #    nlayers.append(nlayer)
        #    print(nlayer)
        if i==10:
            break
    
