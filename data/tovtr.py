import numpy as np
import os
from glob import glob
import sys

from pyevtk.hl import gridToVTK

sys.path.append('..')

from utils import read_uni


def save_vtr(data, filepath):
    data = np.squeeze(data)
    x = np.arange(data.shape[0]+1)
    y = np.arange(data.shape[1]+1)
    z = np.arange(data.shape[2]+1)
    gridToVTK(filepath, x, y, z, cellData={'data': data.copy()})


def uni_to_vtr(prefix='test/', folder='simSimple_0001'):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    files = sorted(glob('./simSimple_0001/density_*.uni'))
    for i in range(len(files)):
        print(i, end="", flush=True)
        _, d = read_uni(files[i])
        save_vtr(d, os.path.join(prefix+'data'+str(i)))


def npz_to_vtr(prefix='test/', filepath='data.npz'):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    data = np.load(filepath)
    outputs = data['outputs']
    inputs = data['inputs']
    truth = data['truths']
    for i in range(outputs.shape[0]):
        print(i, end="", flush=True)
        save_vtr(outputs[i], os.path.join(prefix+'/outputs_'+str(i)))
        save_vtr(inputs[i], os.path.join(prefix+'/inputs_'+str(i)))
        save_vtr(truth[i], os.path.join(prefix+'/truth_'+str(i)))
