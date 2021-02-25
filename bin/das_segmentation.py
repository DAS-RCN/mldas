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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
    # Original values: nChannel 100, maxIter 1000, minLabels 3, lr 0.1, nConv 2
    parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                        help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=100, type=int,
                        help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                        help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--nConv', metavar='M', default=2, type=int,
                        help='number of convolutional layers')
    parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                        help='number of superpixels')
    parser.add_argument('--compactness', metavar='C', default=100, type=float,
                        help='compactness of superpixels')
    args = parser.parse_args()
    
def main():

    args = parse_args()

if __name__ == '__main__':
    main()
