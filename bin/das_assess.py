#!/bin/env python

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
import time
import argparse

# Local
from explore.mapping import minute_prob

# Externals
import yaml
import numpy

def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser('assess.py')
  add_arg = parser.add_argument
  add_arg('-a','--action',   help='Operation to be executed', required=True)
  add_arg('-d','--data',     help='Path to input target data', nargs='+', required=True)
  add_arg('-m','--mpi',      help='Number of CPUs for parallel execution', type=int)
  add_arg('-o','--out',      help='Path to output repository where data are saved', default='./')
  add_arg('-p','--platform', help='Action to be executed')
  add_arg('-s','--software', help='Software to be used for phase-weighted stack')
  return parser.parse_args()

def load_config(params):
  """
  Prepare configuration dictionary.
  """
  config = {}
  if params.data[0].endswith('.yaml'):
    with open(params.data[0]) as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
    config['data_path'] = sorted(glob.glob(config['data_path']))
  else:
    config['data_path'] = sorted(params.data)
  return config

def multi_core_run(iterable,params):
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  start_time = time.time()
  for fname in iterable[rank::size]:
    print('Rank {:>5}/{}: {}'.format(rank+1,size,fname))
    options = option_gather(params,fname)
    dirname = os.path.dirname(os.path.abspath(__file__))
    os.system('%s/../scripts/mldas.sh %s'%(dirname,options))
  if rank == 0:
    print("Time spent with ", size, " threads in seconds")
    print("-----", int((time.time()-start_time)), "-----")

def option_gather(params,fname):
  options=""
  for arg in vars(params):
    if getattr(params, arg)!=None:
      if arg=='data':
        if getattr(args, arg)[0].endswith('.yaml'):
          options+=' -d %s:%s'%(getattr(args, arg)[0],fname)
        else:
          options+=' -d '+fname
      if arg in ['action','out','platform','software','weight']:
        options+=' -%s %s'%(arg[0],getattr(args, arg))
  return options
    
if __name__ == '__main__':
  args = parse_args()
  config = load_config(args)
  multi_core_run(config['data_path'],args)
