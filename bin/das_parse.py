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
import argparse

# Externals
import numpy as np
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('results_dirs', nargs='*', help='Benchmark output directories')
    add_arg('-o', '--output-file', help='Text file to dump results to')
    add_arg('-v', '--version', help='Pytorch version')
    return parser.parse_args()

def load_result(path, ranks=1, **kwargs):

    summary_files = [f for f in os.listdir(path) if f.startswith('summaries_')]
    assert (ranks == len(summary_files))
    train_rate, inference_rate = 0, 0
    for summary_file in summary_files:
        with np.load(os.path.join(path, summary_file)) as f:
            train_rate += f['train_rate'].mean()
            inference_rate += f['valid_rate'].mean()
    return dict(train_rate=train_rate, inference_rate=inference_rate,
                ranks=ranks, **kwargs)

def load_results(results_dirs,sw):
    results = []

    for results_dir in results_dirs:
        
        # Extract hardware, software, from path
        m = re.match('(.*)-n(\d+)-ds(\d+)-bs(\d+)-ep(\d+)-dp(\d+)-lr(\d+\.\d+)', os.path.basename(results_dir))
        hw, ranks = m.group(1), int(m.group(2))
        backend = 'mpi' if hw=='hsw' else 'nccl'

        # Use all subdirectories as models
        results.append(load_result(results_dir, hardware=hw, version=sw,
                                   backend=backend, model='resnet', ranks=ranks))

    return pd.DataFrame(results)

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    results = load_results(args.results_dirs,args.version)
    print(results)

    if args.output_file is not None:
        print('Writing data to', args.output_file)
        results.to_csv(args.output_file, index=False, sep='\t')

if __name__ == '__main__':
    main()
