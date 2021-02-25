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
import sys
import json
import torch
import random
import logging
import torch.utils.data as data
from torchvision import transforms
from .util import load_image

class BaseDataset(data.Dataset):

    def __init__(self, data_path, data_type, id2rid):
        super(BaseDataset, self).__init__()
        self.data_type = data_type
        self.dataset = self._load_data(data_path + '/' + data_type + '/data.txt')
        self.id2rid = id2rid
        self.data_size = len(self.dataset)
        self.transformer = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    def __getitem__(self, index):
        image_file, attr_ids = self.dataset[index % self.data_size]        
        input = load_image(image_file, self.transformer)
        labels = list()
        for attr_id in self.id2rid[0].keys():
            labels.append(float(attr_id in attr_ids))
        return input, torch.tensor(labels)

    def __len__(self):
        return self.data_size

    def _load_data(self, data_file):
        dataset = list()
        if not os.path.exists(data_file):
            return dataset
        with open(data_file) as d:
            for line in d.readlines():
                line = json.loads(line)
                dataset.append(self.readline(line))
        random.shuffle(dataset)
        return dataset
    
    def readline(self, line):
        data = [None, None]
        data[0] = line["image_file"]
        data[1] = line["id"]
        return data
