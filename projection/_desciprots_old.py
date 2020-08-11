'''
This script makes descriptors from POSCAR file
'''

# Import modules
import os
import sys
import itertools
import functools
import operator
from joblib import Parallel
from joblib import delayed
import numpy as np

# Import modules which I made
from projection.moment import calc_ghm_value
from projection.projmatrix import ProjMatrix
from projection.spharmo import calc_product_sph_radius

class Descriptor():
    '''
    Make basis function
    '''
    def __init__(self, l_vals, exponents):
        ins = ThrDimInvs()
        ins.set_basis(l_vals, exponents)
        self.sph_basis = ins.sph_basis
        self.ghm_basis = ins.ghm_basis
        self.basis = ins.basis


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


def calc_each_struct_des(basis, sph_path, _sph_to_calc, ghm_path, max_mom,
                         _ghm_to_calc, eig_vec, i):
    '''
    Calc each atoms descritors
    Load values and calc product of spherical harmonics and GHM
    '''
    # Make spherical harmonics value dict
    sph_dict = {}
    # Get path info
    sph_path_li = [os.path.join(sph_path, str(each) + '.npy')
                   for each in _sph_to_calc]
    # Get info of i th structure
    data_li = [np.load(path)[i] for path in sph_path_li]
    # Set values to dict
    for _sph, sph_data in zip(_sph_to_calc, data_li):
        sph_dict[str(_sph)] = sph_data

    # Make Guassian Herimite Moment dict
    ghm_dict = {}
    ghm_path_li = [os.path.join(ghm_path, str(sum(each)), str(max_mom),
                                str(each) + '.npy')
                   for each in _ghm_to_calc]
    ghm_data_li = [np.load(path)[i] for path in ghm_path_li]
    # Set values
    for _ghm, ghm_data in zip(_ghm_to_calc, ghm_data_li):
        ghm_dict[str(_ghm)] = ghm_data

    # Calc product of spherical harmonics
    sph_val = [functools.reduce(operator.mul, [sph_dict[str(_sph)]
                                               for _sph in each[0]])
               for each in basis]
    ghm_val = [functools.reduce(operator.mul, [ghm_dict[str(_ghm)]
                                               for _ghm in each[1]])
               for each in basis]
    # Calc Product of them by broadcating
    # product_arr shape is len(basis) x atom_num x values
    product_val = [sph * ghm for sph, ghm in zip(sph_val, ghm_val)]


    # Calc decrtipros of each atom
    product_with_coeff = [val * coeff for val, coeff
                          in zip(product_val, eig_vec)]
    result = np.sum(np.array(product_with_coeff), axis=0)

    return result, i


if __name__ == '__main__':
    '''
    ------------------------------------------------------------------------
    Parameter Setting
    ------------------------------------------------------------------------
    '''
    ARGS = sys.argv
    L_VALS = list(map(int, ARGS[1].split('_')))
    EXPONENTS = list(map(int, ARGS[2].split('_')))
    CUT_OFF = int(ARGS[3])
    # Set parameters of radius function
    PARAMS = list(map(int, ARGS[4].split('_')))
    # set max moment value of sigma
    MAX_MOM = int(ARGS[5])
    WITH_ATOM = str(ARGS[6])
    INDEX = int(ARGS[7])

    print('L_VALS = ' + str(L_VALS) + ' , EXPONENTS = ' + str(EXPONENTS))
    print('PARAMS = ' + str(PARAMS))
    print('MAX_MOM = ' + str(MAX_MOM))
    print('WITH_ATOM = ' + str(WITH_ATOM))

    # Get basis info
    DES_INS = Descriptor(L_VALS, EXPONENTS)
    SPH_BASIS = DES_INS.sph_basis
    GHM_BASIS = DES_INS.ghm_basis
    BASIS = DES_INS.basis

    '''
    -----------------------------------------------------------------------
    Get save path info
    -----------------------------------------------------------------------
    '''
    SAVE_PATH = '/home/ryuhei/vega/projection_method/data/working/atom_des/'
    SAVE_PATH = os.path.join(SAVE_PATH, str(CUT_OFF), ARGS[1],
                             ARGS[4], ARGS[2], str(MAX_MOM), str(WITH_ATOM))

    '''
    ------------------------------------------------------------------------
    Load invariants data
    ------------------------------------------------------------------------
    '''
    INV_PATH = '/home/ryuhei/vega/projection_method/data/working/invariants/'
    INV_PATH = os.path.join(INV_PATH, str(L_VALS) + '_' +
                            str(EXPONENTS) + '.npy')
    INVS = np.load(INV_PATH)
    if INVS.size == 0:
        raise Exception('There is no invariants by this basis')

    EIG_VEC = INVS[:, INDEX].reshape(-1, 1)

    _SPH_TO_CALC = _make_calc_sph_lst(L_VALS)

    SPH_PATH = '/home/ryuhei/vega/projection_method/data/working/spharmo/'
    SPH_PATH = os.path.join(SPH_PATH, str(CUT_OFF), str(ARGS[4]))

    _GHM_TO_CALC = _make_calc_ghm_lst(EXPONENTS)

    _GHM_PATH = '/home/ryuhei/vega/projection_method/data/working/ghmoment/'\
                'cohesive/'
    if WITH_ATOM == 'on':
        GHM_PATH = os.path.join(_GHM_PATH, 'weighed', str(CUT_OFF))
    elif WITH_ATOM == 'off':
        GHM_PATH = os.path.join(_GHM_PATH, 'only_struct', str(CUT_OFF))

    '''
    -----------------------------------------------------------------------
    Start parallel calculation
    -----------------------------------------------------------------------
    '''
    RESULT = Parallel(n_jobs=-1, verbose=10)([delayed(calc_each_struct_des)
                                              (BASIS, SPH_PATH, _SPH_TO_CALC,
                                               GHM_PATH, MAX_MOM,
                                               _GHM_TO_CALC, EIG_VEC, i)
                                              for i in range(10)])
    RESULT.sort(key=lambda x: x[1])
    RESULT_LST = [t[0] for t in RESULT]
    SAVE_PATH = os.path.join(SAVE_PATH, str(INDEX) + 'test.npy')
    np.save(SAVE_PATH, RESULT_LST)
