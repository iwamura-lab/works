'''
This script convert atom_descriptors to structual descriptors
'''

# !/usr/bin/env/python
# -*- coding: UTF-8 -*-


# Import Modules
import numpy as np
import pandas as pd
from scipy import stats as sp
from itertools import chain
import functools
import sys
import os


def from_atom_des_to_struct(atom_des, atom_num_lst,
                            weighed, mom_order=2, icov=False):
    # Make atom_des flatten
    _flatten = [flatten_arr(each, weighed=weighed) for each in atom_des]
    if weighed == 'on':
        vec = np.zeros((1, 24))
        flat_arr = np.vstack((vec, *_flatten))[1:]

    elif weighed == 'off':
        fla_arr = np.zeros((1, 1))
        for a_file in _flatten:
            fla_arr = np.append(fla_arr, a_file)
        flat_arr = fla_arr[1:].reshape(sum(atom_num_lst), 1)

    # conver flat_arr to pandas DataFrame to use dropna method
    df = pd.DataFrame(flat_arr)
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

    # Check whether it is okay to check catsing to float
    if np.allclose(conf, imag_arr[1:], atol=1e-5):
        pass
    else:
        raise Exception('This Descriptor should not be casting to float')

    # Raise error if something goes wrong
    if sum(atom_num_lst) != index:
        raise Exception('Bugggggg!!')

    # Make descriptors
    _descriptor = list(map(functools.partial(atomic_rep_to_compound_descriptor,
                                             mom_order=mom_order, icov=icov),
                           all_des))
    # Reshape array
    descriptor = np.array(_descriptor).reshape(len(all_des), -1)
    return descriptor


def atomic_rep_to_compound_descriptor(rep, mom_order, icov):
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


def flatten_arr(arr, weighed):
    if weighed == 'on':
        result = np.array(list(chain.from_iterable(list(arr))))
    elif weighed == 'off':
        working = np.array(list(chain.from_iterable(list(arr))))
        result = np.array(list(chain.from_iterable(list(working))))
    return result


if __name__ == '__main__':

    atom_num_lst = np.load('/home/ryuhei/vega/projection_method/data/'
                           'working/num_list.npy')

    CALC_PATH = '/home/ryuhei/vega/projection_method/data/working/atom_des/'
    SAVE_PATH = '/home/ryuhei/vega/projection_method/data/descriptors/'
    args = sys.argv
    path_info = args[1]
    WEIGHED = args[2]
    path_of_atom = os.path.join(CALC_PATH, path_info)
    path_to_save = os.path.join(SAVE_PATH, path_info)
    all_calc_lst = os.listdir(path_of_atom)
    for a_file in all_calc_lst:
        path = os.path.join(path_of_atom, a_file)
        data = np.load(path)
        des = from_atom_des_to_struct(data, atom_num_lst, WEIGHED)
        save_path = os.path.join(path_to_save, a_file)
        np.save(save_path, des)
        del des, data

    print('Calc Path = ' + str(path_of_atom) + str(args[1]))
    print('Save Path = ' + str(path_to_save) + str(args[1]))
