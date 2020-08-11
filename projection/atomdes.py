'''
This is script for caclulating atomic descriptors
This method is based on projection method
Use Guassian Hermite Moment and spherical harmonics x GRDF values
'''

# !/usr/bin/env/python
# -*- coding: UTF-8 -*-

# Import modules
import os
import json
import functools
import itertools
import operator
import numpy as np
from pymatgen.io.vasp import Poscar

# Import modules which I made
from projection.moment import calc_ghm_value
from projection.projmatrix import ProjMatrix
from projection.spharmo import calc_product_sph_radius


class AtomDescriptors(ProjMatrix):
    '''
    This class calculates atomic descriptors
    All calculation conditon is assumed to configured by JSON file
    '''

    def __init__(self, json_path):
        '''
        Load json file
        '''
        data_li = open(json_path, 'r')
        data = json.load(data_li)
        self.__data__ = data

        # Set spherical harmonics and GuassianHerimiteMoment info
        self.l_vals = data['l_vals']
        self.exponents = data['exponents']

        # Set cut off radius info
        self.cut_off = data['cut_off']
        self.radius_params = data['radius_params']
        self.radius_type = data['radius_type']

        self.max_mom = data['max_mom']
        # self.atom_info_path = data['atom_info_path']
        # Set eigen vector data path
        self.eig_path = data['eig_path']
        self.poscar_path = data['poscar_path']

        # Set save name
        self.save_path = data['save_path']

        # Set basis function
        self.set_basis(self.l_vals, self.exponents)

        # set values which to be calced
        self.sph_val_dict = None
        self.ghm_val_dict = None

    def calc_descriptor(self, poscar_path=None):
        '''
        Calculating atomic descriptors
        '''

        # Get POSCAR data
        # You can reset POSCAR data if you want to calc different POSCAR
        # at same condition defined in JSON file
        if poscar_path is None:
            poscar = Poscar.from_file(self.poscar_path)
        else:
            poscar = Poscar.from_file(poscar_path)

        # calc spherical harmonics and GaussianHermiteMoment values
        # If basis is set to None,  unnuse the descriptor
        if self.sph_basis is None:
            self._sph_val_dict = None
        else:
            self._sph_val_dict = calc_sph(self, poscar)

        if self.ghm_basis is None:
            self._ghm_val_dict = None
        else:
            self._ghm_val_dict = calc_moment(self, poscar)

        # calc product of sph and ghm
        product_val = calc_product_of_sph_and_ghm(self)

        # Load eigen vector data
        path = self.eig_path
        data_path = os.path.join(path, str(self.l_vals) +
                                 '_' + str(self.exponents) + '.npy')
        eig_arr = np.load(data_path)

        # calc descriptor from product val and eigen vector
        for i in range(eig_arr.shape[1]):
            eig_vec = eig_arr[:, i]
            _product_li = [eig_val * each_array for eig_val, each_array
                           in zip(eig_vec, product_val)]
            descriptor = np.sum(np.array(_product_li), axis=0)

            # hsatck the result array
            if i == 0:
                result_arr = descriptor
            else:
                result_arr = np.hstack((result_arr, descriptor))
        return result_arr


def calc_sph(self, poscar):
    '''
    Calc product of spherical harmonics value and radius function
    '''
    # Calc each sph value and make dict for fast computing
    sph_to_calc = _make_calc_sph_lst(self.l_vals)

    if self.radius_type == 'gaussian':
        gamma = None
        atom_info_path = None

    elif self.radius_type == 'gaussian_weighed':
        gamma = self.__data__['gamma']
        atom_info_path = self.__data__['atom_info_path']

    else:
        raise Exception('Set radius type properly!')

    sph_val_li = [calc_product_sph_radius(poscar, each_sph,
                                          self.cut_off, self.radius_params,
                                          self.radius_type, 'cos',
                                          atom_info_path,
                                          gamma)
                  for each_sph in sph_to_calc]

    # Make dictionary of values
    each_sph_val_dict = {}
    for sph_basis, sph_val in zip(sph_to_calc, sph_val_li):
        each_sph_val_dict[str(sph_basis)] = sph_val

    # calc product of spherical harmonics values
    val_lst = [functools.reduce(operator.mul,
                                [each_sph_val_dict[str(each_basis)]
                                 for each_basis in multi_basis])
               for multi_basis in self.sph_basis]

    # Make dict of sph value
    sph_val_dict = {}
    for each_basis, each_val in zip(self.sph_basis, val_lst):
        sph_val_dict[str(each_basis)] = each_val

    return sph_val_dict


def calc_moment(self, poscar):
    '''
    Calc GaussianHermiteMoment from json file
    '''
    # Get atom info
    if self.atom_info_path is None:
        atom_info = None
    else:
        atom_info = np.load(self.atom_info_path).item()

    # Make list to calc
    ghm_to_calc = _make_calc_ghm_lst(self.exponents)
    ghm_val_li = [calc_ghm_value(each_basis, self.cut_off,
                                 self.max_mom, poscar,
                                 atom_info=atom_info)
                  for each_basis in ghm_to_calc]

    # Make ghm value dict so that fastly computing
    each_ghm_val_dict = {}
    for ghm_basis, ghm_val in zip(ghm_to_calc, ghm_val_li):
        each_ghm_val_dict[str(ghm_basis)] = ghm_val

    # calc product of ghm values
    val_lst = [functools.reduce(operator.mul,
                                [each_ghm_val_dict[str(each_basis)]
                                 for each_basis in multi_basis])
               for multi_basis in self.ghm_basis]

    # make dictionary of data
    ghm_val_dict = {}
    for each_basis, each_val in zip(self.ghm_basis, val_lst):
        ghm_val_dict[str(each_basis)] = each_val

    return ghm_val_dict


def calc_product_of_sph_and_ghm(self):
    '''
    calc direct products of sph and ghm
    If one of them is None, return other product val
    '''

    # Get value from self
    sph_val_dict = self._sph_val_dict
    ghm_val_dict = self._ghm_val_dict

    product_li = []
    if sph_val_dict is None:
        for each_basis in self.basis:
            ghm_val = ghm_val_dict[str(each_basis)]
            product_li.append(ghm_val)

    elif ghm_val_dict is None:
        for each_basis in self.basis:
            sph_val = sph_val_dict[str(each_basis)]
            product_li.append(sph_val)

    else:
        for each_basis in self.basis:
            sph_basis = each_basis[0]
            ghm_basis = each_basis[1]
            each_sph = sph_val_dict[str(sph_basis)]
            each_ghm = ghm_val_dict[str(ghm_basis)]
            product_val = each_sph * each_ghm
            product_li.append(product_val)

    return product_li


# Define function to make dict
def _make_calc_sph_lst(l_vals):
    lst_to_calc = []
    _calc_lval = list(set(l_vals))
    for each_l in _calc_lval:
        m_vals = range(-each_l, each_l + 1)
        for each_m in m_vals:
            _lst = [each_l, each_m]
            lst_to_calc.append(_lst)
    return lst_to_calc


def _make_calc_ghm_lst(exponents):
    exps = list(set(exponents))
    _lst = [list(itertools.product([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   repeat=expo)) for expo in exps]
    lst_to_calc = []
    for each in _lst:
        each_rep = [list(sum(np.array(a_file))) for a_file in each]
        for target in each_rep:
            lst_to_calc.append(target)
    return lst_to_calc


if __name__ == '__main__':
    DESINS = AtomDescriptors('../test/cond.json')
    print(DESINS.__dict__)
    import time
    START = time.time()
    RES = DESINS.calc_descriptor()
    ELPTIME = time.time() - START
    print('calc time is ' + str(ELPTIME))
    print('Descriptor : {}'.format(RES))
