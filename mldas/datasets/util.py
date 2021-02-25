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

import collections
from PIL import Image

def load_image(image_file, transformer):
    img = Image.open(image_file)
    width, height = img.size
    # transform
    input = transformer(img)
    return input

def load_label(label_file):
    rid2name = list()   # rid: real id, same as the id in label.txt
    id2rid = list()     # id: number from 0 to len(rids)-1 corresponding to the order of rids
    rid2id = list()     
    with open(label_file) as l:
        rid2name_dict = collections.defaultdict(str)
        id2rid_dict = collections.defaultdict(str)
        rid2id_dict = collections.defaultdict(str)
        new_id = 0 
        for line in l.readlines():
            line = line.strip('\n\r').split(';')
            if len(line) == 3: # attr description
                if len(rid2name_dict) != 0:
                    rid2name.append(rid2name_dict)
                    id2rid.append(id2rid_dict)
                    rid2id.append(rid2id_dict)
                    rid2name_dict = collections.defaultdict(str)
                    id2rid_dict = collections.defaultdict(str)
                    rid2id_dict = collections.defaultdict(str)
                    new_id = 0
                rid2name_dict["__name__"] = line[2]
                rid2name_dict["__attr_id__"] = line[1]
            elif len(line) == 2: # attr value description
                rid2name_dict[line[0]] = line[1]
                id2rid_dict[new_id] = line[0]
                rid2id_dict[line[0]] = new_id
                new_id += 1
        if len(rid2name_dict) != 0:
            rid2name.append(rid2name_dict)
            id2rid.append(id2rid_dict)
            rid2id.append(rid2id_dict)
    return rid2name, id2rid, rid2id
