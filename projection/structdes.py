# !/usr/bin/env/python
# -*- coding: UTF-8 -*-

# Import modules
import os
import json
import functools
import numpy as np
import pandas as pd
from scipy import stats as sp
from distutils.util import strtobool


class StructDescriptor():
    '''
    This is for making structural descriptors from atomic descriptors
    The calc condition of this class is assumed to set by JSON file
    '''

    def __init__(self, json_path):
        '''
        Read json file and set calc conditon
        '''
        data_li = open(json_path, 'r')
        data = json.load(data_li)
        self._data = data

        # Set init by json file
        self.atom_num_lst = np.load(data['atom_num_list'])
        atomic_des_path = data['atomic_des_path']
        atomic_des_name = data['atomic_descriptors']
        self.with_cov = bool(strtobool(data['with_cov']))
        self.save_path = data['save_path']

        # Load atomic descriptors and make array
        data_path_li = [os.path.join(atomic_des_path, name + '.npy')
                        for name in atomic_des_name]
        for i, each_path in enumerate(data_path_li):
            if i == 0:
                atom_des = np.load(each_path)
            else:
                data = np.load(each_path)
                atom_des = np.hstack((atom_des, data))
        self.atomic_descriptor = atom_des

    def calc_descriptor(self, mom_order=2):
        '''
        Calculate descriptor from atomic descriptors
        '''
        atom_num_lst = self.atom_num_lst
        # Exclude nan value
        # conver flat_arr to pandas DataFrame to use dropna method
        df = pd.DataFrame(self.atomic_descriptor)
        # Exclude nan data
        _ex_nan = np.array(df.dropna(how='any', axis=1))

        # Sort descriptors by structure
        # conf and imag_arr are values to check whether it is ok to cast float
        all_des = []
        index = 0
        imag_arr = np.zeros((1, _ex_nan.shape[1]))
        conf = np.zeros((sum(atom_num_lst), _ex_nan.shape[1]))
        for i, atom_num in enumerate(atom_num_lst):
            fin = index + atom_num
            # Convert to float
            each_val = _ex_nan[index:fin].reshape(atom_num, -1)
            all_des.append(each_val.astype(np.complex).real)

            # Make imag_arr to check
            each_img_arr = each_val.astype(np.complex).imag
            imag_arr = np.vstack((imag_arr, each_img_arr))

            index += atom_num

        # Check whether it is okay to cast to float
        if np.allclose(conf, imag_arr[1:], atol=1e-5):
            pass
        else:
            raise Exception('This Descriptor should not be casted to float')

        # Raise error if something goes wrong
        if sum(atom_num_lst) != index:
            raise Exception('Bugggggg!!')

        # Make descriptors
        _descriptor = list(map(functools.partial(atomic_rep_to_struct_descriptor,
                                                 mom_order=mom_order,
                                                 icov=self.with_cov),
                               all_des))
        # Reshape array
        descriptor = np.array(_descriptor).reshape(len(all_des), -1)
        return descriptor


def atomic_rep_to_struct_descriptor(rep, mom_order=2, icov=False):
    '''
    convert atomic descriptors to structual descriptors
    this function is made by Seko
    '''

    d = []
    d.extend(np.mean(rep, axis=0))
    if (mom_order > 1):
        d.extend(np.std(rep, axis=0))
    if (mom_order > 2):
        d.extend(sp.stats.skew(rep, axis=0))
    if (mom_order > 3):
        d.extend(sp.stats.kurtosis(rep, axis=0))
    if (icov is True):
        if (rep.shape[0] == 1):
            cov = np.zeros((rep.shape[1], rep.shape[1]))
        else:
            cov = np.cov(rep.T)
        for i, row in enumerate(cov):
            for j, val in enumerate(row):
                if (i < j):
                    d.append(val)
    return d


if __name__ == '__main__':
    DESINS = StructDescriptor('./element.json')
    DES = DESINS.calc_descriptor()
    print(DES[0:5])
