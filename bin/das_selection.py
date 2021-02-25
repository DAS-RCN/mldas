
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

import os
import re
import glob
import sys
import json
import yaml
import numpy
import random
import argparse
import collections

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config_file', help='Configuration file')
    add_arg('--sample-size', type=int, help='Size of target dataset (in thousands)')
    add_arg('-o', '--output-dir', help='Output directory')
    return parser.parse_args()

class MultiLabelDataLoader():
    def __init__(self, src_data, sample_size, num_labels, dst_dir, mode, ratios=[0.70,0.15,0.15]):
        
        # Prepare output directory
        os.makedirs(dst_dir, exist_ok=True)

        # Create data and label list
        self._MakeDictionary(sample_size, src_data, dst_dir, mode)
        
        # Split data to Train, Val, Test
        dataset = collections.defaultdict(list)
        with open(dst_dir + '/data.txt') as d:
            for i,line in enumerate(d.readlines()):
                line = json.loads(line)
                if (i+1)/(num_labels*sample_size) <= ratios[0]:
                    data_type = "Train"
                elif (i+1)/(num_labels*sample_size) <= ratios[0] + ratios[1]:
                    data_type = "Validate"
                else:
                    data_type = "Test"
                dataset[data_type].append(line)
                
        # Write to file
        self._WriteDataToFile(dataset["Train"], dst_dir + "/TrainSet/")
        self._WriteDataToFile(dataset["Validate"], dst_dir + "/ValidateSet/")
        self._WriteDataToFile(dataset["Test"], dst_dir + "/TestSet/")

    def _MakeDictionary(self, sample_size, src_data, output_dir, mode):
        """
        Create data.txt and label.txt files
        """
        data = glob.glob(src_data+'*/waves/*.jpg')
        random.shuffle(data)
        data = data[:sample_size]
        if mode!='unary':
            noise_data = glob.glob(src_data+'*/noise/*.jpg')
            random.shuffle(noise_data)
            data += noise_data[:sample_size]
        random.shuffle(data)
        f = open(output_dir+'/data.txt','w')
        for i,fname in enumerate(data):
            fpath = {'image_file':fname,'id':['waves']} if mode=='unary' else \
                    {'image_file':fname,'id':['noise','waves']} if 'waves' in fname else \
                    {'image_file':fname,'id':['noise']}
            json.dump(fpath,f)
            f.write('\n')
        f.close()
        f = open(output_dir+'/label.txt','w')
        f.write('1;label\nwaves')
        f.close()

    def _WriteDataToFile(self, src_data, dst_dir):
        """
        Write info of each objects to data.txt as predefined format
        """
        os.makedirs(dst_dir, exist_ok=True)
        with open(dst_dir + "/data.txt", 'w') as d:
            for line in src_data:
                d.write(json.dumps(line, separators=(',',':'))+'\n')
                    
if __name__ == '__main__':

    args = parse_args()
    mode = re.split('/|\.',args.config_file)[-2]

    # Extract data path and number of labels from configuration file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    src_data= config['data_config']['data_path']
    num_labels = config['data_config']['num_labels']

    # Execute dataset list creation
    MultiLabelDataLoader(src_data, 1000*args.sample_size, num_labels, args.output_dir, mode)
