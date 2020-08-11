'''
This script calculates descriptors based on moment invariants from POSCAR file
Moment is Gaussian Hermite Moment
'''


# !/usr/bin/env/python
# -*- coding: UTF-8 -*-


# Import modules
import numpy as np
import functools
import operator
import scipy.special as sp


def calc_ghm_value(basis, cut_off, max_mom, poscar,
                   sigma_li=None, atom_info=None):
    '''
    calclulates Gaussian Hermite Moment from poscar
    if you want to use atom_info, please set values
    basis : list
        The basis Moments (lenghth must be 3)
        Ex) [2, 0, 0] ------> M200
    max_mom : int
        The value which moment order is the most large
        This value effects the sigma value
    '''
    # Get sigma value to calc
    if sigma_li is None:
        _sigma_li = [1, 1, np.sqrt(2/5), (2/np.sqrt(9+np.sqrt(57))),
                          np.sqrt(2)/(np.sqrt(np.sqrt(21) + 7))]
    else:
        _sigma_li = sigma_li
    sigma = _sigma_li[max_mom] * cut_off

    # calc moment value
    moment_val = calc_ghm_from_poscar(poscar, basis, sigma,
                                      cut_off, atom_info)
    return moment_val


def calc_ghm_from_poscar(poscar, basis, sigma, cut_off, atom_info=None):
    '''
    This function calcs moment invariants from POSCAR file
    input
    ------------------------------------------------------------------
    poscar : poscar(pymatgen file)
        The poscar from which generate descriptor
    basis : np.array like
        The basis moment to use
    sigma : float
        The sigma value which is used when calcs moment
    cut_off : float
        cut off radius
    atom_info : None or dict (default is None)
        The information of component
        If you wanto to use atom information, please set
        If None, calc moment invariants only from crystal structure
    -------------------------------------------------------------------
    '''
    # Create structure class
    st = poscar.structure
    sites = st.sites

    # Create empty array to add result value
    if atom_info is None:
        result_arr = np.zeros((1, 1))
    else:
        result_arr = np.zeros((1, len(atom_info['Al'])))

    for each in sites:
        # Get neighbor atoms's coords information
        neighbors = st.get_neighbors(each, r=cut_off)
        # Conver coords for atom in attention to become origin
        neigh_coord = [a_file[0].coords - each.coords
                       for a_file in neighbors]
        trans_arr = np.array(neigh_coord)

        # Calc moment values of neighboring atoms
        _moments = [calc_each_ghm(basis, each_coords, sigma)
                    for each_coords in trans_arr]

        # If atom_info isn't set, use only struct information
        if atom_info is None:
            if _moments == []:
                result = np.zeros((1, 1))
            else:
                result = np.mean(_moments).reshape(1, 1)
            result_arr = np.vstack((result_arr, result))

        # If atom_info is set, use atom_info too
        else:
            if _moments == []:
                result = np.zeros((1, len(atom_info['Al'])))
            else:
                atom_num = len(_moments)
                arr = np.array(_moments).reshape(1, atom_num)/atom_num
                neigh_name = [a_file[0].species_string for a_file in neighbors]
                neigh_info = np.array([atom_info[i] for i in neigh_name])
                # Get weighed value
                result = np.dot(arr, neigh_info)
            result_arr = np.vstack((result_arr, result))
    return result_arr[1:]


def calc_each_ghm(basis, coords, sigma):
    '''
    This function calcs gaussian hermite moment
    input
    ---------------------------------------------------------------
    basis : np.array or list
        The list of exponent of polynomials
        The length must be 3
    coords : list
        The corrds list to calc
        Ex) [1, 2, 3] (length = 3)
    sigma : float
    ----------------------------------------------------------------
    '''
    value = [sp.hermite(mom)(x/sigma) * np.exp(-x ** 2 / (2 * (sigma ** 2)))
             for mom, x in list(zip(basis, coords))]
    result = functools.reduce(operator.mul, value)
    return result


if __name__ == '__main__':
    POSCAR_PATH = '/home/ryuhei/poscar_data/cohesive/descriptors/'\
                  'A2X3/2085_A2XY2_Bi2Te3/Al-O/POSCAR'
    from pymatgen.io.vasp import Poscar
    POSCAR = Poscar.from_file(POSCAR_PATH)

    ATOM_INFO_PATH = '/home/ryuhei/projection_method/data/working/'\
                     'std_pca_atominfo.npy'
    ATOMINFO = np.load(ATOM_INFO_PATH).item()
    print(ATOMINFO)

    RESULT = calc_ghm_value([2], 6, 2, POSCAR, atom_info=ATOMINFO)
    print(RESULT)
