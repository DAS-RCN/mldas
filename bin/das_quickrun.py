#!/usr/bin/env python

# Internals
import argparse

# Externals
from mldas import explore

parser = argparse.ArgumentParser('mldas.py')
parser.add_argument('operation', help='Operation to execute')
parser.add_argument('-i','--input-data', help='Path to input data', nargs='+')
parser.add_argument('-c','--config', help='Path to configuration file')
parser.add_argument('-o','--output', help='Path to output repository', default='./')
parser.add_argument('-f','--flag', help='Additional flag', default=[])
parser.add_argument('-l','--log', action='store_true', help='Use logarithmic scale')
parser.add_argument('-r','--region', type=float, help='Selected region (in standard deviation)')
parser.add_argument('-s','--stop', type=int, help='Stopping index')
parser.add_argument('-t','--test-index', type=int, help='Index of testing result')
parser.add_argument('-v','--verbose', action='store_true', help='Do verbose')
args = parser.parse_args()

try:
    getattr(explore,args.operation)(**vars(args))
except AttributeError:
    print("Method '%s' not found in MLDAS."%args.operation)
    quit()
