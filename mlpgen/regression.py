#!/usr/bin/env python
"""
Program for the regression of mlp about paramagnetic FCC Fe
"""

# import standard modules
import copy
import argparse
#import time
#import tqdm
import random
import numpy as np

# from mlptools import some modules
from mlptools.common.fileio import InputParams
from mlptools.mlpgen.regression import PotEstimation
from mlptools.common.structure import Structure
from mlptools.mlpgen.io import ReadFeatureParams
from mlptools.mlpgen.model import Terms

def rearange_L(array, index_array):
    """Move designated columns to the head of array.

    Args:
        array (ndarray): input array(2D)
        index_array (list): list of index by which columns are designated

    Returns:
        ndarray: changed array
    """
    rest = np.delete(array, index_array, 1)
    return np.hstack((array[:, index_array], rest))

class VirtualDataInput:
    """Generate a virtual DataInput from normal DataInput
    """
    def __init__(self, di):
        self.vdi = copy.deepcopy(di)
        self.vdi.n_type = 2

    def get_data_input(self):
        """Return a newly generated DataInput.

        Returns:
            DataInput: virtual DataInput
        """
        return self.vdi

class MagneticStructuralFeatures:
    """Data structure including magnetic structural features
    """
    def __init__(self, tr, spin_array, vdi):
        st_set_all_train = self.get_virtual_structures(tr.train, spin_array)
        n_st_dataset = [len(data.st_set) for data in tr.train]
        term = Terms(st_set_all_train, vdi, n_st_dataset, vdi.train_force)
        self.train_x = np.hstack((tr.train_x, term.get_x()))
        st_set_all_test = self.get_virtual_structures(tr.test, spin_array)
        n_st_dataset = [len(data.st_set) for data in tr.test]
        force_dataset = [vdi.wforce for v in vdi.test_names]
        term = Terms(st_set_all_test, vdi, n_st_dataset, force_dataset)
        self.test_x = np.hstack((tr.test_x, term.get_x()))

    def get_x(self):
        """Return the X matrices for regression

        Returns:
            ndarray: X matrices needed for training and test
        """
        return self.train_x, self.test_x

    def get_virtual_structures(self, dataset, spin_array):
        """Generate virtual structures from dataset based on spin_array and return the
           list of them

        Args:
            dataset (dr_array): array of the instances, DataRegression
            spin_array (ndarray): which spin each atom has

        Returns:
            list: list of rewrited structures
        """
        index_array = np.nonzero(spin_array==1)[0]
        n_atom_1 = len(index_array)
        n_atom_2 = sum(dataset[0].st_set[0].n_atoms) - len(index_array)
        n_atoms = [n_atom_1, n_atom_2]
        specie1 = ["A" for i in range(n_atom_1)]
        specie2 = ["B" for i in range(n_atom_2)]
        elements = specie1.extend(specie2)
        type1 = [0 for i in range(n_atom_1)]
        type2 = [1 for i in range(n_atom_2)]
        types = type1.extend(type2)
        st_list = [Structure(st.axis, rearange_L(st.positions, index_array), n_atoms, elements, \
                   types=types, comment=st.comment) for data in dataset for st in data.st_set]
        return st_list

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, \
        help='Input file name. Training is performed from vasprun files.')
    parser.add_argument('-p', '--pot', type=str, \
        default='mlp.pkl', help='Potential file name for mlptools')
    args = parser.parse_args()

    # prepare temporary spin array
    spin_array = np.array(random.choices([-1, 1], k=32))

    p = InputParams(args.infile)
    di = ReadFeatureParams(p).get_params()
    vdi = VirtualDataInput(di).get_data_input()

    # calculate structural features
    tr = PotEstimation(di=di)
    # calculate magnetic structural features
    tr.train_x, tr.test_x = MagneticStructuralFeatures(tr, spin_array, vdi).get_x()
    tr.set_regression_data()

    # start regression
    if args.noreg is False:
        reg_method, alpha_min, alpha_max, n_alpha = read_regression_params(p)
        if (reg_method == 'ridge' or reg_method == 'lasso'):
            pot = tr.regularization_reg(method=reg_method,alpha_min=alpha_min,\
                alpha_max=alpha_max,n_alpha=n_alpha,svd=args.svd)
        elif reg_method == 'normal':
            pot = tr.normal_reg()

        pot.save_pot(file_name=args.pot)
        pot.save_pot_for_lammps(file_name=args.lammps)

        print(' --- input parameters ----')
        pot.di.model_e.print()
        print(' --- best model ----')
        if (reg_method == 'ridge' or reg_method == 'lasso'):
            print(' alpha = ', tr.best_alpha)

        rmse_train_e, rmse_test_e, rmse_train_f, files_train, \
            rmse_test_f, rmse_train_s, rmse_test_s, files_test \
            = tr.get_best_rmse()

        print(' -- Prediction Error --')
        for re, rf, rs, f in \
            zip(rmse_train_e, rmse_train_f, rmse_train_s, files_train):
            print(' structures :', f)
            print(' rmse (energy, train) = ', re * 1000, ' (meV/atom)')
            if rf is not None:
                print(' rmse (force, train) = ', rf, ' (eV/ang)')
                print(' rmse (stress, train) = ', rs, ' (GPa)')
        for re, rf, rs, f in \
            zip(rmse_test_e, rmse_test_f, rmse_test_s, files_test):
            print(' structures :', f)
            print(' rmse (energy, test) = ', re * 1000, ' (meV/atom)')
            if rf is not None:
                print(' rmse (force, test) = ', rf, ' (eV/ang)')
                print(' rmse (stress, test) = ', rs, ' (GPa)')
