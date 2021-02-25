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

# Systems
import os
import glob
import random

# Externals
import h5py
import numpy
import torch
from collections import OrderedDict
from torchvision import transforms
from PIL import Image

def hdf5read(fname,key=['dsi30','data','variable']):
    """
    Read input raw DAS file. The file is supposed to have an HDF5 format and the
    actual data stored in a group that have one of the following names: ``dsi30``,
    ``data`` or ``variable``.

    Parameters
    ----------
    fname : :py:class:`str`
      Full path to the HDF5 file
    key : :py:class:`str` or :py:class:`list`
      Group's key name where the data are stored. If not specify, a list of keys
      will be looped over.

    Returns
    -------
    data : :py:class:`numpy.ndarray`
      Raw data
    """
    f = h5py.File(fname,'r')
    if type(key)==str:
        data = f[f[key][0,0]]
    else:
        assert any(keyname in f.keys() for keyname in key), 'No acceptable key found in input HDF5 file.'
        for keyname in f.keys():
            if keyname in key:
                data = f[f['%s/dat'%keyname][0,0]]
                break
    return data

def load_model(fname,model):
    """
    Load saved model's parameter dictionary to initialized model.
    The function will remove any ``.module`` string from parameter's name.

    Parameters
    ----------
    fname : :py:class:`str`
      Path to saved model
    model : :py:class:`torch.nn.Module`
      Initialized network network architecture

    Returns
    -------
    model : :py:class:`torch.nn.Module`
      Up-to-date neural network model
    """
    checkpoint = torch.load(fname,map_location=lambda storage, loc: storage)
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'])
    return model

def save_data(data, dir_dst, fname):
    """
    Save raw data region as JPG image.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
      Input raw data
    dir_dst : :py:class:`str`
      Path to save image
    fname : :py:class:`str`
      Output file name
    """
    img = (data-data.min())/(data.max()-data.min())
    img = Image.fromarray(numpy.uint8(img*255))
    img.save('%s/%s.jpg'%(dir_dst,fname))

def load_image(data,rgb=False,to_numpy=False,squeeze=False):
    """
    Load input data as tensor image. Input data can be either in the form
    of JPG image or raw data region and will be converted into numpy.uint8
    format, then either RGB or grayscale image.

    Parameters
    ----------
    data : :py:class:`str` or :py:class:`numpy.ndarray`
      Input data, either path saved image, or raw data array.
    rgb : :py:class:`bool`
      Convert data to RGB data image
    to_numpy : :py:class:`bool`
      Convert data to numpy array
    squeeze : :py:class:`bool`
      Squeeze data to remove any dimensions equal to 1.
    """
    if type(data)==str:
        data = os.path.expandvars(data)
        image = Image.open(data)
    else:
        image = (data-data.min())/(data.max()-data.min())
        image = Image.fromarray(numpy.uint8(image*255))
    if rgb:
        image = transforms.ToTensor()(image.convert("RGB")).view(1,3,*image.size)
    else:
        image = transforms.ToTensor()(image.convert("L")).view(1,1,*image.size)
    if squeeze:
        image = torch.squeeze(image)
    if to_numpy:
        image = image.numpy()
    return image

def load_bulk(dname,size,rgb=False,to_numpy=False,labeled=True):
    """
    Load multiple images from directory.

    Parameters
    ----------
    dname : :py:class:`str`
      Path to directory where images are saved.
    size : :py:class:`int`
      Number of images to be loaded
    rgb : :py:class:`bool`
      Convert data to RGB data image
    to_numpy : :py:class:`bool`
      Convert data to numpy array
    labeled : :py:class:`bool`
      Whether images are saved in a labeled structure (same structure than for 
      :py:class:`torchvision.datasets.ImageFolder` class) or directly in target repository.

    Returns
    -------
    tensors : :py:class:`torch.Tensor` or :py:class:`numpy.ndarray`
      Output list of loaded images either in tensor or numpy array formats.
    labels : :py:class:`torch.Tensor` or :py:class:`numpy.ndarray`
      Output list of labels either in tensor or numpy array formats.
    """
    fname = os.path.expandvars(dname)
    if labeled:
        all_labels = [label for label in os.listdir(fname+'/train/') if label!='.DS_Store']
        assert size%len(all_labels)==0, 'Requested bulk size not multiple of number of labels.'
        labels, tensors = [], []
        for i,label in enumerate(all_labels):
            file_list = glob.glob('%s/*/%s/*.jpg'%(dname,label))
            for fname in random.sample(file_list,size//len(all_labels)):
                tensors.append(load_image(fname,rgb=rgb))
            labels += ([i]*(size//len(all_labels)))
        tensors = torch.stack(tensors).squeeze(dim=1)
        if to_numpy:
            tensors = torch.squeeze(tensors).numpy()
        idxs = random.sample(range(size), size)
        return tensors[idxs], numpy.array(labels)[idxs]
    else:
        tensors = []
        file_list = glob.glob('%s/*.jpg'%dname)
        for fname in random.sample(file_list,size):
            tensors.append(load_image(fname,rgb=rgb))
        tensors = torch.stack(tensors).squeeze(dim=1)
        if to_numpy:
            tensors = torch.squeeze(tensors).numpy()
        return tensors,numpy.array([0]*(size))
